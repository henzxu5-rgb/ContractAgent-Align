"""Build a Chinese report only from complete, frozen independent test batches."""
import argparse
from pathlib import Path

from .analysis import compare
from .budget import Ledger
from .data import DatasetStore
from .scoring import summarize
from .settings import ARTIFACTS, MODEL_ID
from .storage import now, read_json, write_json


def build(conditions, root=ARTIFACTS, preference_summary_path=None, sft_summary_path=None):
    root = Path(root)
    expected = set(DatasetStore(root).select("test"))
    rows, results, sources, inference_batches = {}, {}, {}, {}
    for name, path in conditions.items():
        batch = read_json(path)
        if batch["config"].get("selection") != "test" or batch["config"].get("phase") != "main":
            raise ValueError("Final report requires formal independent test batches")
        mode = "full_context" if name == "full_context" else "agent"
        runs = [r for r in batch["runs"] if r["mode"] == mode]
        if len(runs) != len(expected) or {r["task_id"] for r in runs} != expected:
            raise ValueError(f"{name}: incomplete or different frozen test tasks; preserve failures and finish pending tasks")
        if any(r["model"] != MODEL_ID for r in runs):
            raise ValueError("Final comparison must use the same selected base model")
        if name in ("base", "full_context") and any(r.get("adapter") for r in runs):
            raise ValueError("Base condition unexpectedly uses an adapter")
        if name not in ("base", "full_context") and any(not r.get("adapter") or "pilot-" in Path(r["adapter"]).name for r in runs):
            raise ValueError("Trained condition is missing a formal adapter")
        rows[name], results[name], sources[name] = runs, summarize(runs), str(path)
        inference_batches[name] = batch["config"].get("inference_batch_size", 1)
    if not {"base", "sft", "full_context"}.issubset(rows):
        raise ValueError("A complete report needs base Agent, SFT Agent and full-context baseline")
    if len(set(inference_batches.values())) != 1:
        raise ValueError("Formal comparisons must use the same inference batch cap")
    comparisons = {f"base_to_{k}": compare(rows["base"], v) for k, v in rows.items() if k not in ("base", "full_context")}
    for name in ("dpo_raw", "dpo_verified"):
        if name in rows:
            comparisons[f"sft_to_{name}"] = compare(rows["sft"], rows[name])
    if "dpo_raw" in rows and "dpo_verified" in rows:
        comparisons["raw_to_verified"] = compare(rows["dpo_raw"], rows["dpo_verified"])
    budget = Ledger(root / "budget.sqlite").summary()
    uid = Path(sources["base"]).stem.removesuffix("-test-base")
    preference_path = Path(preference_summary_path) if preference_summary_path else root / "preferences" / (uid + "-preferences") / "summary.json"
    sft_path = Path(sft_summary_path) if sft_summary_path else root / "datasets" / (uid + "-sft.summary.json")
    preference = read_json(preference_path) if preference_path.exists() else None
    training_data = read_json(sft_path) if sft_path.exists() else None
    result = {"created": now(), "status": "complete_test_batches", "test_tasks": len(expected),
              "sources": sources, "conditions": results, "comparisons": comparisons,
              "inference_batch_caps": inference_batches,
              "preference_diagnostics": preference, "training_data": training_data,
              "research_question_status": "matched_label_flows_compared" if "dpo_raw" in rows and "dpo_verified" in rows else "verified_dpo_only_no_raw_control" if "dpo_verified" in rows else "no_formal_dpo_data_insufficient",
              "cost_upper": {k: v for k, v in budget.items() if k != "entries"},
              "limitations": ["单次训练种子；合同聚类bootstrap不能代替多次重训。", "17个固定假设，仅检验未见合同上的表现。", "原数据由研究者及众包标注，本项目无新增法律专家审核。", "不排除公开数据进入模型预训练；不代表法律咨询能力。", "工具交互为本项目适配协议，不是ContractNLI官方Agent榜单。", "没有Judge与纯程序偏好的独立消融，不能证明Judge本身优于纯规则；若未运行原始Judge DPO对照，不能回答两套偏好标签的因果差异。"]}
    recovery = root / "reports/detached-clone-launch-plan.json"
    if recovery.exists():
        result["evaluation_recovery"] = read_json(recovery)
        result["limitations"].append("评测中断后迁移GPU：基础/SFT及raw DPO前142任务在RTX6000D，raw剩余198与verified全部340在RTX PRO6000 Blackwell；权重哈希、固定软件与批量上限一致，但硬件及实际批次组成不同，小幅差异不能作为严格的硬件受控因果证据。")
    sft_delta = results["sft"]["macro_f1"] - results["base"]["macro_f1"]
    result["findings"] = {"sft_macro_f1_gain": sft_delta,
                          "sft_completion_counts": {k: results[k]["status_counts"].get("completed", 0) for k in ("base", "sft")},
                          "verified_dpo_continue_signal": comparisons.get("sft_to_dpo_verified", {}).get("continue_signal"),
                          "verified_dpo_beats_raw_point_estimate": results["dpo_verified"]["macro_f1"] > results["dpo_raw"]["macro_f1"] if {"dpo_verified", "dpo_raw"} <= results.keys() else None}
    write_json(root / "reports/final_results.json", result)
    lines = ["# ContractAgent-Align 独立测试报告", "", f"生成：{result['created']}。固定20份合同、{len(expected)}个任务；失败计入主指标。", "",
             "| 条件 | Macro-F1 | 证据F1 | 无效引用尝试率 | 完成数 | 平均延迟秒 | 推理摊销秒 |", "|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda x: "不适用" if x is None else f"{x:.4f}"
    for name, metrics in results.items():
        lines.append(f"| {name} | {fmt(metrics['macro_f1'])} | {fmt(metrics['evidence']['f1'])} | {fmt(metrics['invalid_citation_rate'])} | {metrics['status_counts'].get('completed', 0)}/{metrics['n']} | {metrics['mean_seconds']:.2f} | {metrics['mean_amortized_inference_seconds']:.2f} |")
    lines += ["", "全文基线使用更长上下文。显存/超长失败必须结合逐例记录解释，不能直接算作工具使用的能力优势。", "", "## 配对差异", ""]
    lines += ["工具错误率以执行事件数（包含提交和无效动作）为分母；引用错误率以所有提交尝试中的引用数为分母。各类F1、混淆矩阵、token用量和状态计数见 final_results.json；逐例与超长/显存诊断见 FAILURE_ANALYSIS.md。", ""]
    lines += [f"所有条件使用相同推理批量上限：{next(iter(inference_batches.values()))}。批处理可能产生浮点舍入导致的解码差异；开发速度测试不作为任务效果证据。病例耗时为并发下延迟，另记录推理摊销时间。", ""]
    for name, comparison in comparisons.items():
        lo, hi = comparison["cluster_bootstrap_95_interval"]
        lines.append(f"- {name}：Macro-F1差值 {comparison['macro_f1_difference']:+.4f}；按合同聚类bootstrap区间 [{lo:+.4f}, {hi:+.4f}]。")
    lines += ["", "## 本轮结论", "",
              f"SFT相对基础Agent的Macro-F1差值为{sft_delta * 100:+.2f}个百分点；成功提交由{results['base']['status_counts'].get('completed', 0)}增加至{results['sft']['status_counts'].get('completed', 0)}/340。该提升包含协议遵循及提交完成率的改善，不能全部归因为法律推理能力增强。"]
    if "dpo_verified" in results and "dpo_raw" in results:
        delta = results["dpo_verified"]["macro_f1"] - results["dpo_raw"]["macro_f1"]
        decision = comparisons["sft_to_dpo_verified"]["continue_signal"]
        lines += ["", f"验证后DPO相对普通Judge DPO的Macro-F1点估计差值为{delta * 100:+.2f}个百分点。验证后DPO是否达到预设继续信号：{decision}。应同时查看配对区间、证据指标及恢复限制；不能仅凭标注核验通过或训练loss下降声称对齐更有效。"]
    if training_data:
        lines += ["", "## 数据生产与Judge诊断", "", f"SFT：{training_data['accepted_trajectories']}条通过检查的真实轨迹，{training_data['state_action_samples']}条状态动作。保留标签分布：{training_data.get('labels', {})}。"]
    if preference:
        lines += ["", f"自然候选{preference['candidate_pairs']}对，共同保留{preference['matched_pairs']}对，其中工具行为差异{preference['tool_pairs']}对；两套标签不同{preference['preference_differences']}对。",
                  f"顺序翻转率{preference['order_flip_rate']:.4f}，验证后拒判率{preference['verified_abstention_rate']:.4f}；候选覆盖{preference.get('task_coverage')}个任务，保留后{preference.get('retained_task_coverage')}个任务。",
                  f"原始Judge DPO对照是否达到预设标签差异门槛：{preference['train_raw_control']}。研究问题可回答范围：{result['research_question_status']}。"]
    lines += ["", "## 结论边界", "", *["- " + x for x in result["limitations"]], "", f"费用保守累计/预留：{budget['committed_cny']:.2f}元，不等于供应商实际账单。", ""]
    (root / "reports/FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    from .failure_analysis import build as build_failure_analysis
    build_failure_analysis(result, root)
    if preference:
        from .preference_audit import build as build_preference_audit
        build_preference_audit(preference_path, root)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", action="append", required=True, help="name=/absolute/path/to/batch.json")
    args = parser.parse_args()
    conditions = dict(value.split("=", 1) for value in args.condition)
    report = build(conditions)
    print({"status": report["status"], "conditions": list(report["conditions"]), "test_tasks": report["test_tasks"]})
