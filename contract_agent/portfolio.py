"""Chinese portfolio materials derived strictly from completed test results."""
from pathlib import Path

from .storage import read_json


def generate(result, root):
    root = Path(root)
    output = root / "deliverables"
    output.mkdir(parents=True, exist_ok=True)
    uid = Path(result["sources"]["base"]).stem.removesuffix("-test-base")
    sft_path = root / "datasets" / (uid + "-sft.summary.json")
    prefs_path = root / "preferences" / (uid + "-preferences") / "summary.json"
    sft = result.get("training_data") or (read_json(sft_path) if sft_path.exists() else {})
    prefs = result.get("preference_diagnostics") or (read_json(prefs_path) if prefs_path.exists() else {})
    conditions = result["conditions"]
    base, trained = conditions["base"], conditions["sft"]
    dpo = conditions.get("dpo_verified")
    lines = ["# 简历材料（只含本次实测）", "", "**ContractAgent-Align：合同证据核查 Agent 的偏好数据与对齐实验**", "",
             "- 基于公开 ContractNLI 数据构建合同级隔离的工具交互环境，接入 BM25 检索、证据读取与结构化提交，记录完整执行轨迹并保留失败。",
             f"- 筛选 {sft.get('accepted_trajectories', '见数据报告')} 条真实示范，导出 {sft.get('state_action_samples', '见数据报告')} 条状态动作 SFT 样本；采用 Qwen3-4B 的 QLoRA 训练，仅监督模型动作。",
             f"- 实现 Rubrics Judge、候选顺序交换与标准证据核验；{prefs.get('candidate_pairs', '见偏好报告')} 组自然候选共同保留 {prefs.get('matched_pairs', '见偏好报告')} 组偏好，区分 AI 反馈与人类标注。",
             f"- 在固定20份未见合同、340个任务上独立评测；基础 Agent / SFT 的 Macro-F1 分别为 {base['macro_f1']:.4f} / {trained['macro_f1']:.4f}，同时报告证据F1、失败率与按合同聚类的配对区间。"]
    if dpo:
        delta = dpo["macro_f1"] - trained["macro_f1"]
        lines.append(f"- 完成动作级 DPO；验证偏好版本 Macro-F1 为 {dpo['macro_f1']:.4f}，相对 SFT 差值 {delta * 100:+.2f} 个百分点。")
    else:
        lines += ["", "本次偏好数据未达到预设DPO门槛，因此不能写成已完成正式DPO训练。"]
    lines += ["", "面试中需说明：这是工具交互适配实验；没有新增专家审核，不是通用法律服务，也不是PPO式RLHF。若未提升，应如实解释失败分析，不写成性能优化成功。", ""]
    (output / "RESUME.md").write_text("\n".join(lines), encoding="utf-8")
    demo = ["# 五分钟演示脚本", "", "0:00–0:40：说明问题——在相同策略模型与候选池下，证据核验能否改善偏好标签及训练效果。展示来源、许可证、合同划分与去重清单。", "",
            "0:40–1:40：启动只读界面，选择一条真实轨迹，展示检索、已观察片段、提交和独立参考标注；再展示一条证据遗漏或未提交的失败。运行上下文看不到标准标签。", "",
            f"1:40–2:40：打开偏好记录，展示共享上下文、两种下一动作及各自后续执行。候选 {prefs.get('candidate_pairs', '见报告')} 对，保留 {prefs.get('matched_pairs', '见报告')} 对，工具差异 {prefs.get('tool_pairs', '见报告')} 对。解释平局/拒判、顺序检查和程序核验的边界。", "",
            "2:40–3:30：展示真实训练receipt、动作token损失掩码、策略参数变化与参考参数冻结、适配器重载证明。明确微型验收不计入正式效果。", "",
            "3:30–4:30：展示FINAL_REPORT.md中的完整340任务比较、全文基线、Macro-F1与证据F1，以及配对区间。报告负结果和显存/超长限制。", "",
            "4:30–5:00：说明可复用的数据SOP、失败记录、模型版本、费用上限及继续/停止条件；提出下一步必须依据实际失败类型。", "",
            "复现入口：解压后在项目根目录运行 `python -m contract_agent.cli prepare` 和 `python -m contract_agent.server --port 8766`；真实训练仍需固定模型、CUDA环境及环境变量凭据。", ""]
    (output / "DEMO_5_MIN.md").write_text("\n".join(demo), encoding="utf-8")
    delivery = ["# 交付索引", "", "- 数据来源与限制：contract_agent/DATA_CARD.md", "- 操作与质检流程：contract_agent/SOP.md", "- 复现命令：contract_agent/README.md", "- 独立测试：artifacts/reports/FINAL_REPORT.md 与 final_results.json", "- 训练产物与重载证据：artifacts/models/ 与 training_logs/", "- 真实示范与偏好：artifacts/datasets/ 与 preferences/", "- 简历表述：本目录 RESUME.md", "- 五分钟演示：本目录 DEMO_5_MIN.md", "", "完整数据、运行日志和适配器在本地工作目录；密钥仅在忽略的.env中，不属于可公开交付内容。", ""]
    (output / "PROJECT_DELIVERY.md").write_text("\n".join(delivery), encoding="utf-8")
    return {"folder": str(output), "files": ["RESUME.md", "DEMO_5_MIN.md", "PROJECT_DELIVERY.md"]}
