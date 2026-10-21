"""Post-test diagnostics; these records never feed the acting model or training."""
from collections import Counter, defaultdict
from pathlib import Path

from .data import DatasetStore
from .scoring import summarize
from .storage import now, read_json, write_json, write_jsonl


def classify(run, gold):
    """Observable proxies, deliberately not an automated legal-error judgment."""
    expected = set(gold["evidence_ids"])
    seen = set(run.get("observed_ids", []))
    submitted = run.get("submission") or {}
    cited = set(submitted.get("evidence_ids", []))
    tags = []
    if run.get("status") != "completed":
        tags.append("execution_not_completed")
    if "context_overflow" in str(run.get("error", "")):
        tags.append("context_overflow")
    if run.get("error_type") == "OutOfMemoryError" or "exhausted GPU memory" in str(run.get("error", "")):
        tags.append("out_of_memory")
    if not run["score"]["label_correct"]:
        tags.append("label_wrong_or_execution_failed")
        if expected and expected <= seen:
            tags.append("all_reference_evidence_observed_but_label_failed")
    if expected:
        if expected - seen:
            tags.append("reference_evidence_not_observed")
        if (expected & seen) - cited:
            tags.append("observed_reference_evidence_not_cited")
        if cited - expected:
            tags.append("citation_outside_reference_set")
    elif submitted.get("label") == "NotMentioned" and cited:
        tags.append("not_mentioned_with_citations")
    if run["score"].get("invalid_citation_attempts", 0):
        tags.append("unobserved_or_invalid_citation_attempt")
    if run.get("tool_errors"):
        tags.append("tool_error")
    if run["score"]["strict_success"]:
        tags.append("exact_annotation_success")
    return tags


def build(result, root):
    root = Path(root)
    store = DatasetStore(root)
    conditions, rows, by_condition = {}, [], {}
    for name, source in result["sources"].items():
        batch = read_json(source)
        mode = "full_context" if name == "full_context" else "agent"
        selected = [r for r in batch["runs"] if r["mode"] == mode]
        groups, contracts = defaultdict(list), defaultdict(list)
        counts, examples = Counter(), defaultdict(list)
        for run in sorted(selected, key=lambda r: r["task_id"]):
            tags = classify(run, store.gold(run["task_id"]))
            counts.update(tags)
            task = store.index[run["task_id"]]
            groups[run["task_id"].split(":", 1)[1]].append(run)
            contracts[run["doc_id"]].append(run)
            row = {"condition": name, "task_id": run["task_id"], "doc_id": run["doc_id"],
                   "run_id": run["id"], "hypothesis": task["hypothesis"],
                   "status": run["status"], "score": run["score"], "tags": tags,
                   "observed_ids": run.get("observed_ids", []),
                   "reference_evidence_ids": store.gold(run["task_id"])["evidence_ids"],
                   "submission": run.get("submission"), "error": run.get("error"),
                   "tool_errors": run.get("tool_errors", []), "tool_calls": run.get("tool_calls", 0)}
            rows.append(row)
            for tag in tags:
                if len(examples[tag]) < 3:
                    examples[tag].append({"task_id": run["task_id"], "run_id": run["id"]})
        by_condition[name] = {r["task_id"]: r for r in selected}
        conditions[name] = {"n": len(selected), "tag_counts": dict(counts), "examples": dict(examples),
                            "by_hypothesis": {k: summarize(v) for k, v in sorted(groups.items())},
                            "by_contract": {k: summarize(v) for k, v in sorted(contracts.items())}}
    paired = {}
    for name, right in by_condition.items():
        if name in ("base", "full_context"):
            continue
        left = by_condition["base"]
        if set(left) != set(right):
            raise ValueError("Post-test paired analysis cannot drop tasks")
        wins, losses = [], []
        for tid in sorted(left):
            a, b = left[tid]["score"]["label_correct"], right[tid]["score"]["label_correct"]
            if b and not a: wins.append(tid)
            if a and not b: losses.append(tid)
        paired["base_to_" + name] = {"label_gains": wins, "label_regressions": losses,
                                           "both_correct": sum(left[t]["score"]["label_correct"] and right[t]["score"]["label_correct"] for t in left),
                                           "both_wrong_or_failed": sum(not left[t]["score"]["label_correct"] and not right[t]["score"]["label_correct"] for t in left)}
    capacity = None
    if "full_context" in by_condition:
        full = by_condition["full_context"]
        excluded = {tid: classify(run, store.gold(tid)) for tid, run in full.items()
                    if "context_overflow" in str(run.get("error", "")) or run.get("error_type") == "OutOfMemoryError" or "exhausted GPU memory" in str(run.get("error", ""))}
        kept = sorted(set(full) - set(excluded))
        capacity = {"excluded_full_context_task_tags": excluded, "same_subset_task_ids": kept,
                    "same_subset_metrics": {name: summarize([runs[t] for t in kept]) for name, runs in by_condition.items()} if kept else {},
                    "rule": "Exclude only explicit full-context input overflow or GPU OOM, never label errors; apply identical task subset to every condition. Primary metrics still include all 340 tasks."}
    report = {"created": now(), "condition_diagnostics": conditions, "paired_label_changes": paired,
              "full_context_capacity_diagnostic": capacity,
              "selection": "post-test descriptive analysis; no training or tuning consumes these outputs",
              "example_selection": "first three task IDs in lexical order for each tag; all cases included in per_case_analysis.jsonl",
              "limitations": ["Tags overlap and must not be added as disjoint categories.",
                              "Observed at some point does not imply present in the final rolling context.",
                              "Unannotated citations may still be semantically valid; no expert review is claimed.",
                              "These diagnostics do not establish that retrieval or reasoning caused an error."]}
    write_json(root / "reports/failure_analysis.json", report)
    write_jsonl(root / "reports/per_case_analysis.jsonl", rows)
    lines = ["# 独立测试后的逐例诊断", "", "本报告只描述保存记录，不用于本轮训练或调参。标签可重叠；观察过参考证据不等于最后上下文仍保留它。证据不在参考集合中也不等于法律上错误。", "",
             "| 条件 | 任务 | 未完成 | 标签错误或失败 | 参考证据未全观察 | 已观察但未引用 | 工具错误任务 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, data in conditions.items():
        counts = data["tag_counts"]
        keys = ["execution_not_completed", "label_wrong_or_execution_failed", "reference_evidence_not_observed", "observed_reference_evidence_not_cited", "tool_error"]
        lines.append("| " + " | ".join([name, str(data["n"]), *[str(counts.get(k, 0)) for k in keys]]) + " |")
    lines += ["", "## 与基础 Agent 的逐任务变化", ""]
    for name, data in paired.items():
        lines.append(f"- {name}：新增标签正确 {len(data['label_gains'])} 例，原先正确而变错/失败 {len(data['label_regressions'])} 例。完整任务编号见 failure_analysis.json。")
    if capacity:
        lines += ["", "## 全文输入容量的补充诊断", "",
                  f"全文配置的显式超长/显存失败共 {len(capacity['excluded_full_context_task_tags'])} 任务。以下仅为对所有条件采用相同剩余任务的补充统计，主结果仍保留全部340任务；不是按模型答对与否筛样。全文配置为本实现的16384 token上限，不代表基础模型的架构上限。", "",
                  "| 条件 | 相同子集任务 | Macro-F1 | 证据F1 |", "|---|---:|---:|---:|"]
        for name, metrics in capacity["same_subset_metrics"].items():
            ev = metrics["evidence"]["f1"]
            lines.append(f"| {name} | {metrics['n']} | {metrics['macro_f1']:.4f} | {ev:.4f} |" if ev is not None else f"| {name} | {metrics['n']} | {metrics['macro_f1']:.4f} | 不适用 |")
    lines += ["", "17类陈述和20份合同的分项指标、按固定规则选取的回放例子见 failure_analysis.json；全部条件、全部任务的预测与诊断见 per_case_analysis.jsonl。不得根据本报告筛选任务后重新宣称独立测试成绩。", ""]
    (root / "reports/FAILURE_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    return report
