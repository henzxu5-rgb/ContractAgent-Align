"""Paired comparisons and contract-cluster bootstrap; no judge self-scoring."""
import random
from pathlib import Path

from .scoring import summarize
from .settings import ARTIFACTS, SEED
from .storage import read_json, write_json, now


def compare(a, b, iterations=1000):
    left = {r["task_id"]: r for r in a}
    right = {r["task_id"]: r for r in b}
    common = sorted(left.keys() & right.keys())
    if set(left) != set(right):
        raise ValueError("Comparison task sets must match; failures must not be silently dropped")
    if not common:
        raise ValueError("No paired tasks")
    contracts = sorted({left[tid]["doc_id"] for tid in common})
    grouped = {doc: [t for t in common if left[t]["doc_id"] == doc] for doc in contracts}
    rng = random.Random(SEED)
    differences = []
    for _ in range(iterations):
        selected = [t for doc in rng.choices(contracts, k=len(contracts)) for t in grouped[doc]]
        differences.append(summarize([right[t] for t in selected])["macro_f1"] - summarize([left[t] for t in selected])["macro_f1"])
    differences.sort()
    la, lb = summarize(a), summarize(b)
    ev_a, ev_b = la["evidence"]["f1"], lb["evidence"]["f1"]
    invalid_a, invalid_b = la["invalid_citation_rate"], lb["invalid_citation_rate"]
    return {"paired_tasks": len(common), "contracts": len(contracts), "left": la, "right": lb,
            "macro_f1_difference": lb["macro_f1"] - la["macro_f1"],
            "cluster_bootstrap_95_interval": [differences[int(iterations * .025)], differences[min(iterations - 1, int(iterations * .975))]],
            "bootstrap_iterations": iterations, "seed": SEED,
            "continue_signal": lb["macro_f1"] - la["macro_f1"] >= .03 and ev_a is not None and ev_b is not None and ev_b >= ev_a - .02 and invalid_a is not None and invalid_b is not None and invalid_b <= invalid_a,
            "limitations": ["Single training seed; bootstrap covers task sampling, not retraining variance.", "Fixed ContractNLI hypotheses; no claim of arbitrary legal question generalization."]}


def report_batch(path, root=ARTIFACTS):
    batch = read_json(path)
    modes = sorted({r["mode"] for r in batch["runs"]})
    report = {"created": now(), "batch_id": batch["id"], "config": batch["config"],
              "conditions": {m: summarize([r for r in batch["runs"] if r["mode"] == m]) for m in modes},
              "failures": [{"run_id": r["id"], "task_id": r["task_id"], "mode": r["mode"],
                            "status": r["status"], "score": r["score"], "error": r.get("error")}
                           for r in batch["runs"] if not r["score"]["strict_success"]],
              "claim_boundary": "These are project protocol results, not official ContractNLI model rankings."}
    write_json(Path(root) / "reports" / f"{batch['id']}.json", report)
    return report


def diagnose(path, root=ARTIFACTS):
    from .data import DatasetStore
    from .environment import ContractEnvironment
    from collections import Counter
    store = DatasetStore(root)
    batch = read_json(path)
    rows, counts = [], Counter()
    for run in batch["runs"]:
        if run["mode"] != "agent": continue
        gold = store.gold(run["task_id"])
        expected = set(gold["evidence_ids"])
        seen = set(run["observed_ids"])
        cited = set((run.get("submission") or {}).get("evidence_ids", []))
        env = ContractEnvironment(store.public(run["task_id"]))
        env.search(store.index[run["task_id"]]["hypothesis"], 5)
        tags = []
        if expected - seen: tags.append("reference_evidence_never_observed")
        if expected & seen - cited: tags.append("observed_reference_evidence_not_cited")
        if not run["score"]["label_correct"]: tags.append("wrong_label_or_failed_submission")
        if run["score"]["strict_success"]: tags.append("exact_reference_success")
        counts.update(tags)
        rows.append({"task_id": run["task_id"], "run_id": run["id"], "tags": tags,
                     "gold_evidence_ids": sorted(expected), "never_seen": sorted(expected - seen),
                     "seen_not_cited": sorted((expected & seen) - cited),
                     "initial_bm25_recall_at_5": len(expected & env.observed) / len(expected) if expected else None,
                     "agent_observation_recall": len(expected & seen) / len(expected) if expected else None})
    report = {"created": now(), "batch": batch["id"], "counts": dict(counts), "tasks": rows,
              "boundary": "Missed reference evidence is a diagnostic proxy. Alternative valid evidence may not be annotated; observed does not imply retained in the rolling context."}
    write_json(Path(root) / "reports" / f"{batch['id']}-diagnosis.json", report)
    return report
