"""Durable GPU candidate production and independent, GPU-free Judge jobs."""
import argparse
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor

from .data import DatasetStore
from .preferences import (replay_prefix, continue_candidates, judge_once,
                          verified_preference, export_pairs)
from .runner import context_messages
from .scoring import score_run
from .settings import ARTIFACTS, SEED, MODEL_REVISION
from .storage import read_json, write_json, digest, identifier, now


def checkpoint(folder, name, callback):
    path = folder / (name + ".json")
    if path.exists(): return read_json(path)
    value = callback()
    write_json(path, value, exclusive=True)
    return value


def state_schedule(source, task_order):
    """Disperse contracts and early/middle/final decisions without using outcomes."""
    runs = {r["task_id"]: r for r in source["runs"] if r["mode"] == "agent"}
    if len(runs) != sum(r["mode"] == "agent" for r in source["runs"]):
        raise ValueError("Duplicate source task")
    if set(runs) - set(task_order): raise ValueError("Source task outside frozen training selection")
    ordered = [runs[tid] for tid in task_order if tid in runs]
    depths = []
    for index, run in enumerate(ordered):
        n = len(run["steps"])
        if not n:
            depths.append([]); continue
        preferred = list(dict.fromkeys([0, n // 2, n - 1]))
        offset = index % len(preferred)
        preferred = preferred[offset:] + preferred[:offset]
        depths.append(preferred + [i for i in range(n) if i not in preferred])
    return [(run, order[round_index]) for round_index in range(max(map(len, depths), default=0))
            for run, order in zip(ordered, depths) if round_index < len(order)]


def generate(source_path, policy, run_id, limit=50, excluded_path=None, resume=False, root=ARTIFACTS, parallel=1, temperature=.7):
    if not __import__("re").fullmatch(r"[A-Za-z0-9_-]+", run_id): raise ValueError("Invalid candidate ID")
    root = Path(root)
    if not 1 <= parallel <= 4: raise ValueError("Candidate concurrency must be between 1 and 4")
    if not 0 <= temperature <= 2: raise ValueError("Candidate temperature must be between 0 and 2")
    store = DatasetStore(root)
    source = read_json(source_path)
    excluded_rows = read_json(excluded_path) if excluded_path else []
    excluded = {tuple(k) for k in excluded_rows}
    hard_cap = 400 - len(excluded)
    if not 1 <= limit <= hard_cap: raise ValueError("Total candidate cap includes legacy pairs")
    folder = root / "candidate_batches" / run_id
    if folder.exists() and not resume: raise FileExistsError("Explicit --resume required")
    config = {"source_sha256": digest(source), "excluded_sha256": digest(excluded_rows), "model_revision": MODEL_REVISION,
              "adapter": getattr(policy, "adapter", None), "temperature": temperature, "max_tokens": 384,
              "scheduler": "manifest_tasks_rotating_early_middle_final_v2", "hard_cap": hard_cap}
    if (folder / "config.json").exists():
        if read_json(folder / "config.json") != config: raise ValueError("Candidate configuration changed")
    else: write_json(folder / "config.json", config, exclusive=True)
    records = [read_json(p) for p in sorted((folder / "items").glob("*.json"))]
    done = {(r["source_run"], r["decision_index"]) for r in records} | excluded
    segment = identifier("candidate-segment")
    write_json(folder / "segments" / (segment + ".json"), {"started": now(), "saved_pairs": len(records), "requested_limit": limit,
               "sampling_seed": getattr(policy, "sampling_seed", SEED), "resume": resume,
               "parallel_contexts": parallel, "gpu_batch_cap": 2 if parallel == 1 else 8,
               "sampling_note": "Concurrency can change RNG request order; saved candidates, not exact resampling, are the reproducibility artifact"}, exclusive=True)
    schedule = [(r, i) for r, i in state_schedule(source, store.select("train_initial")) if (r["id"], i) not in excluded]
    pending = [(position, run, index) for position, (run, index) in enumerate(schedule[:limit]) if (run["id"], index) not in done]

    def process(item, worker_policy):
        position, run, index = item
        key = (run["id"], index)
        if store.index[run["task_id"]]["split"] != "train": raise ValueError("Training split required")
        task = store.public(run["task_id"])
        env, history = replay_prefix(task, run, index)
        prompt = context_messages(history)
        attempt = folder / "attempts" / digest(list(key))
        checkpoint(attempt, "started", lambda: {"prompt_sha256": digest(prompt), "source_run": run["id"],
                   "decision_index": index, "segment": segment, "started": now()})
        responses = checkpoint(attempt, "candidates", lambda: worker_policy.generate_many([prompt, prompt], temperature=temperature, max_tokens=384))
        if any("error" in r for r in responses): raise RuntimeError("Candidate error preserved; no silent replacement")
        identical = responses[0]["text"].strip() == responses[1]["text"].strip()
        branches = [] if identical else checkpoint(attempt, "branches", lambda: continue_candidates(task, worker_policy, env, history, responses, root, run["id"]))
        for branch in branches: branch["score"] = score_run(branch, store.gold(run["task_id"]))
        row = {"id": f"{run_id}-{position:04d}", "task_id": run["task_id"], "split": "train", "pilot_only": False,
               "source_run": run["id"], "decision_index": index, "prompt": prompt, "prompt_sha256": digest(prompt),
               "candidates": responses, "branches": branches, "generation_segment": segment, "protocol": config["scheduler"],
               "candidate_status": "identical_actions" if identical else "executed_branches"}
        write_json(folder / "items" / (row["id"] + ".json"), row, exclusive=True)
        return row

    def record(row):
        records.append(row)
        write_json(folder / "progress.json", {"candidate_pairs": len(records), "last_task": row["task_id"], "updated": now()})
        print(json.dumps({"saved_candidate": row["id"], "task": row["task_id"], "labels_correct": [b["score"]["label_correct"] for b in row["branches"]]}), flush=True)

    if parallel == 1:
        for item in pending: record(process(item, policy))
    else:
        from concurrent.futures import as_completed
        from .batching import BatchingModel
        with BatchingModel(policy, 8) as batched, ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [pool.submit(process, item, batched) for item in pending]
            for future in as_completed(futures): record(future.result())
    records.sort(key=lambda r: r["id"])
    batch = {"id": run_id, "config": config, "records": records, "candidate_pairs": len(records),
             "hypotheses": len({r["task_id"].split(":")[1] for r in records}),
             "contracts": len({r["task_id"].split(":")[0] for r in records}), "updated": now()}
    write_json(folder / "batch.json", batch)
    return {k: v for k, v in batch.items() if k != "records"}


def label(batch_path, judge, output_id, legacy_dir=None, root=ARTIFACTS, reuse_legacy_annotations=False, workers=4):
    if not __import__("re").fullmatch(r"[A-Za-z0-9_-]+", output_id): raise ValueError("Invalid preference ID")
    root = Path(root)
    if not 1 <= workers <= 4: raise ValueError("Judge workers must be 1..4")
    store = DatasetStore(root)
    batch = read_json(batch_path)
    out = root / "preferences" / output_id
    legacy = [read_json(p) for p in sorted(Path(legacy_dir).glob("*.json")) if "raw_judge" in read_json(p)] if legacy_dir else []
    inputs = [*legacy, *batch["records"]]
    keys, ids = set(), set()
    for row in inputs:
        key = (row["source_run"], row["decision_index"])
        if key in keys or row["id"] in ids: raise ValueError("Duplicate preference context/ID across legacy and new candidates")
        keys.add(key); ids.add(row["id"])
        if row["split"] != "train" or store.index[row["task_id"]]["split"] != "train": raise ValueError("Training split required")
        if row["prompt_sha256"] != digest(row["prompt"]): raise ValueError("Candidate prompt hash mismatch")

    def label_one(row):
        destination = out / "items" / (row["id"] + ".json")
        if destination.exists():
            saved = read_json(destination)
            if saved["prompt_sha256"] != row["prompt_sha256"] or saved["candidates"] != row["candidates"]:
                raise ValueError("Saved Judge candidate changed")
            return saved
        if "raw_judge" in row and not reuse_legacy_annotations:
            # The provider's flash alias was updated on Oct 17. Keep the original
            # annotations, but label the common pool with the current service.
            prior = {k: row[k] for k in ("raw_judge", "swapped_judge", "verified_winner", "verified_basis") if k in row}
            row = {k: v for k, v in row.items() if k not in prior}
            row["prior_annotations"] = prior
            row["relabel_reason"] = "Provider flash alias changed; preserve old labels and use one current labeling batch"
        if row.get("candidate_status") == "identical_actions":
            row = {**row, "raw_judge": {"winner": "tie", "reason": "Identical next-action text; no Judge call"},
                   "swapped_judge": {"winner": "tie", "reason": "Identical next-action text; no Judge call"},
                   "verified_winner": "abstain", "verified_basis": "identical_actions"}
        elif "raw_judge" not in row:
            for branch in row["branches"]: branch["score"] = score_run(branch, store.gold(row["task_id"]))
            attempt = out / "attempts" / row["id"]
            raw = checkpoint(attempt, "judge_raw", lambda: judge_once(judge, row["prompt"], *row["branches"]))
            swapped = checkpoint(attempt, "judge_swapped", lambda: judge_once(judge, row["prompt"], *row["branches"], swap=True))
            winner, basis = verified_preference(*row["branches"], raw, swapped)
            row = {**row, "raw_judge": raw, "swapped_judge": swapped, "verified_winner": winner, "verified_basis": basis}
        write_json(destination, row, exclusive=True)
        print(json.dumps({"labeled_pair": row["id"], "verified": row["verified_winner"]}), flush=True)
        return row

    # Calls for each candidate pair retain their raw->swapped order. Different
    # pairs can proceed independently; SQLite reservations remain transactional.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(label_one, inputs))
    snapshot = out / "exports" / f"pairs-{len(records):04d}"
    summary = read_json(snapshot / "summary.json") if (snapshot / "summary.json").exists() else export_pairs(records, snapshot)
    summary = {**summary, "candidate_batch": str(batch_path), "legacy_pairs": len(legacy), "candidate_source_config": batch["config"],
               "identical_action_pairs": sum(r.get("candidate_status") == "identical_actions" for r in records)}
    summary["judge_workers"] = workers
    summary["legacy_annotations_reused"] = reuse_legacy_annotations
    write_json(out / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--source", required=True); g.add_argument("--model-path", required=True)
    g.add_argument("--adapter", required=True); g.add_argument("--run-id", required=True)
    g.add_argument("--limit", type=int, default=50); g.add_argument("--excluded"); g.add_argument("--resume", action="store_true")
    g.add_argument("--parallel", type=int, default=1)
    g.add_argument("--temperature", type=float, default=.7)
    j = sub.add_parser("label")
    j.add_argument("--batch", required=True); j.add_argument("--run-id", required=True); j.add_argument("--legacy-dir")
    j.add_argument("--reuse-legacy-annotations", action="store_true")
    j.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.command == "generate":
        from .models import HFModel
        from transformers import set_seed
        count = len(list((ARTIFACTS / "candidate_batches" / args.run_id / "segments").glob("*.json")))
        policy = HFModel(args.model_path, args.adapter)
        policy.sampling_seed = SEED + count
        set_seed(policy.sampling_seed)
        result = generate(args.source, policy, args.run_id, args.limit, args.excluded, args.resume, parallel=args.parallel, temperature=args.temperature)
    else:
        from .models import DeepSeekModel
        result = label(args.batch, DeepSeekModel(phase="main"), args.run_id, args.legacy_dir, reuse_legacy_annotations=args.reuse_legacy_annotations, workers=args.workers)
    print(json.dumps(result), flush=True)


if __name__ == "__main__": main()
