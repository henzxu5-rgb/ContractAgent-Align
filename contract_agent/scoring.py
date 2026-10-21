"""Reference-based evaluation, deliberately separate from the acting model."""
from .settings import LABELS


def evidence_score(predicted, expected):
    p, g = set(predicted), set(expected)
    correct = len(p & g)
    precision = correct / len(p) if p else 0.0
    recall = correct / len(g) if g else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "exact": p == g}


def score_run(run, gold):
    report = run.get("submission")
    if not isinstance(report, dict):
        report = {}
    ids = report.get("evidence_ids", [])
    observed = set(run.get("observed_ids", []))
    invalid = sum(type(i) is not int or i not in observed for i in ids)
    # A rejected citation is still a model error, even if it later recovers.
    seen = set(run.get("observed_ids", [])) if run.get("mode") == "full_context" else set()
    attempts = invalid_attempts = 0
    for event in run.get("events", []):
        action, output = event.get("action", {}), event.get("result", {})
        args = action.get("arguments") if isinstance(action, dict) else None
        if isinstance(args, dict) and action.get("tool") == "submit_result":
            cited = args.get("evidence_ids", [])
            if isinstance(cited, list):
                attempts += len(cited)
                invalid_attempts += sum(type(i) is not int or i not in seen for i in cited)
        for span in output.get("matches", []) + output.get("spans", []):
            seen.add(span["span_id"])
    label = report.get("label")
    correct = label == gold["label"] and run.get("status") == "completed"
    applicable = gold["label"] != "NotMentioned"
    evidence = evidence_score(ids, gold["evidence_ids"]) if applicable else None
    evidence_ok = evidence["exact"] if applicable else ids == []
    return {"gold_label": gold["label"], "predicted_label": label, "label_correct": correct,
            "evidence": evidence, "invalid_citations": invalid, "citation_count": len(ids),
            "invalid_citation_attempts": invalid_attempts, "citation_attempt_count": attempts,
            "tool_errors": len(run.get("tool_errors", [])),
            "strict_success": bool(correct and evidence_ok and invalid == 0),
            "boundary": "Exact annotation agreement is not an independent proof of explanation semantics."}


def summarize(scored_runs):
    confusion = {label: {p: 0 for p in (*LABELS, "INVALID")} for label in LABELS}
    ev = []
    for run in scored_runs:
        s = run["score"]
        pred = s["predicted_label"] if run.get("status") == "completed" else "INVALID"
        confusion[s["gold_label"]][pred if pred in LABELS else "INVALID"] += 1
        if s["evidence"] is not None:
            ev.append(s["evidence"])
    classes = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label].values()) - tp
        denom = 2 * tp + fp + fn
        classes[label] = {"f1": 2 * tp / denom if denom else 0.0, "support": sum(confusion[label].values())}
    n = len(scored_runs)
    cited = sum(r["score"]["citation_count"] for r in scored_runs)
    attempted = sum(r["score"].get("citation_attempt_count", r["score"]["citation_count"]) for r in scored_runs)
    return {"n": n, "macro_f1": sum(c["f1"] for c in classes.values()) / len(LABELS),
            "accuracy": sum(r["score"]["label_correct"] for r in scored_runs) / max(n, 1),
            "strict_success_rate": sum(r["score"]["strict_success"] for r in scored_runs) / max(n, 1),
            "classes": classes, "confusion_matrix": confusion,
            "evidence_n": len(ev),
            "evidence": {k: sum(x[k] for x in ev) / len(ev) if ev else None for k in ("precision", "recall", "f1")},
            "invalid_citation_rate": sum(r["score"].get("invalid_citation_attempts", r["score"]["invalid_citations"]) for r in scored_runs) / attempted if attempted else None,
            "accepted_invalid_citation_rate": sum(r["score"]["invalid_citations"] for r in scored_runs) / cited if cited else None,
            "tool_errors": sum(r["score"]["tool_errors"] for r in scored_runs),
            "action_attempts": sum(len(r.get("events", [])) for r in scored_runs),
            "tool_error_rate": (sum(r["score"]["tool_errors"] for r in scored_runs) / sum(len(r.get("events", [])) for r in scored_runs)) if sum(len(r.get("events", [])) for r in scored_runs) else None,
            "tasks_with_tool_errors": sum(bool(r["score"]["tool_errors"]) for r in scored_runs),
            "mean_tool_calls": sum(r.get("tool_calls", 0) for r in scored_runs) / max(n, 1),
            "mean_seconds": sum(r.get("seconds", 0) for r in scored_runs) / max(n, 1),
            "mean_amortized_inference_seconds": sum(r.get("amortized_inference_seconds", r.get("seconds", 0)) for r in scored_runs) / max(n, 1),
            "inference_batch_size_counts": {str(size): sum(step["response"].get("inference_batch_size", 1) == size for r in scored_runs for step in r.get("steps", [])) for size in sorted({step["response"].get("inference_batch_size", 1) for r in scored_runs for step in r.get("steps", [])})},
            "input_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in scored_runs),
            "output_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in scored_runs),
            "status_counts": {status: sum(r.get("status") == status for r in scored_runs) for status in sorted({r.get("status", "unknown") for r in scored_runs})}}
