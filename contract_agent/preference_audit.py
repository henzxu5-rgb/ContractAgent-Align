"""Describe the frozen labeling funnel without changing a single preference."""
from collections import Counter, defaultdict
from pathlib import Path

from .data import DatasetStore
from .preferences import tool_behavior_difference
from .storage import now, read_json, write_json


def build(summary_path, root):
    root = Path(root)
    summary_path = Path(summary_path)
    store = DatasetStore(root)
    records = [read_json(p) for p in sorted((summary_path.parent / "items").glob("*.json"))]
    summary = read_json(summary_path)
    if len(records) != summary["candidate_pairs"]:
        raise ValueError("Preference audit requires a complete frozen item set")
    reasons, cross, outcomes, models = Counter(), Counter(), Counter(), Counter()
    by_label, by_hypothesis = defaultdict(Counter), defaultdict(Counter)
    switched = decisive_both = raw_bad = raw_decisive = verified_only = 0
    for row in records:
        raw, verified, swapped = row["raw_judge"]["winner"], row["verified_winner"], row["swapped_judge"]["winner"]
        matched = raw in ("A", "B") and verified in ("A", "B")
        tool = matched and tool_behavior_difference(row["candidates"])
        reasons[row["verified_basis"]] += 1
        cross[raw + " -> " + verified] += 1
        branches = row["branches"]
        if not branches:
            outcomes["identical_actions_not_executed_twice"] += 1
        else:
            correct = sum(b["score"]["label_correct"] for b in branches)
            outcomes[{0: "both_wrong_or_failed", 1: "one_correct", 2: "both_correct"}[correct]] += 1
        if raw in ("A", "B"):
            raw_decisive += 1
            raw_bad += not branches[0 if raw == "A" else 1]["score"]["label_correct"]
        if raw in ("A", "B") and swapped in ("A", "B"):
            decisive_both += 1
            switched += raw != swapped
        verified_only += verified in ("A", "B") and raw not in ("A", "B")
        for annotation in (row["raw_judge"], row["swapped_judge"]):
            if "response" in annotation:
                models[annotation["response"].get("model", "unavailable")] += 1
        for target, key in ((by_label, store.gold(row["task_id"])["label"]),
                            (by_hypothesis, store.index[row["task_id"]]["hypothesis_id"])):
            target[key].update({"candidates": 1, "matched": int(matched), "tool_pairs": int(tool)})
    audit = {"created": now(), "candidate_pairs": len(records), "summary_path": str(summary_path),
             "verification_basis_counts": dict(reasons), "raw_to_verified": dict(cross),
             "branch_label_outcomes": dict(outcomes), "by_reference_label": {k: dict(v) for k, v in by_label.items()},
             "by_hypothesis": {k: dict(v) for k, v in by_hypothesis.items()},
             "judge_returned_model_names": dict(models),
             "raw_incorrect_winner": {"numerator": raw_bad, "denominator": raw_decisive},
             "a_b_winner_reversal": {"numerator": switched, "denominator": decisive_both},
             "verified_decisive_but_raw_not_decisive": verified_only,
             "minimum_data_pass": summary["minimum_data_pass"],
             "boundary": "All labels are AI feedback or original reference annotations, not new human judgments. Tool differences use normalized action/argument comparisons, not expert semantic certification."}
    write_json(root / "reports/preference_audit.json", audit)
    lines = ["# 偏好数据生产与质量诊断", "", f"本轮共 {len(records)} 对候选，共同保留 {summary['matched_pairs']} 对；工具行为差异 {summary['tool_pairs']} 对，raw/verified 标签差异 {summary['preference_differences']} 对。", "",
             f"正式DPO最低要求80对共同偏好及20对工具差异；本轮通过：{summary['minimum_data_pass']}。不因拒判降低门槛，不用训练损失下降替代能力证据。", "",
             "| 参考类别 | 候选 | 共同保留 | 工具差异 |", "|---|---:|---:|---:|"]
    for label, counts in sorted(by_label.items()):
        lines.append(f"| {label} | {counts['candidates']} | {counts['matched']} | {counts['tool_pairs']} |")
    lines += ["", "## 分母与解释", "",
              f"- 普通Judge选中的最终结果标签不正确或执行失败：{raw_bad}/{raw_decisive} 个明确A/B判定。此项不是律师对语义推理的准确率。",
              f"- 交换前后均明确A/B时，优胜者反转：{switched}/{decisive_both}。原summary的order_flip_rate还包括tie/abstain变化，分母为全部候选。",
              f"- 验证后可判定、普通Judge仍为tie/abstain：{verified_only} 对。为保持两训练臂相同候选池，这些没有进入共同保留集。",
              "- 相同动作也计入候选总上限，按平局记录，不重复续跑或请求Judge。两候选都错时验证后拒判。",
              "- 候选混合了历史温度0.7与新温度1.2，且上下文不同；不能用保留率差异证明温度的独立作用。",
              "- 工具差异为规范化动作/参数差异，不等于已经人工确认具有法律意义的差异。原始材料与全部拒判均保留。", "",
              "核验依据、原始/验证交叉表、17个陈述覆盖及分支结果分布见 preference_audit.json。", ""]
    (root / "reports/PREFERENCE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return audit
