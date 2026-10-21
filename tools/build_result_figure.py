"""Plot the saved independent results, with a zero-based axis and no new evaluation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
result = json.loads((ROOT / "contract_agent/artifacts/reports/final_results.json").read_text(encoding="utf-8"))
keys = ["base", "sft", "dpo_raw", "dpo_verified", "full_context"]
names = ["Base Agent", "SFT Agent", "Judge DPO", "Verified DPO", "Full-context base"]
with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none", "font.size": 11}):
    fig, ax = plt.subplots(figsize=(10.5, 4.6), facecolor="#fbfaf6")
    ax.set_facecolor("#fbfaf6")
    bars = ax.barh(names, [result["conditions"][k]["macro_f1"] for k in keys],
                   color=["#a6b7af", "#176a5c", "#827799", "#b29054", "#b6beb9"], height=.58)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Macro-F1 (all failures retained)", labelpad=10, color="#5c6965")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e1e5df", linewidth=.8)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=8, colors="#30483e")
    ax.bar_label(bars, labels=[f"{result['conditions'][k]['macro_f1']*100:.2f}%" for k in keys], padding=7, color="#203e32")
    fig.text(.025, .95, "A measurable SFT gain. No additional DPO gain.", fontsize=17, weight="bold", color="#173d34")
    fig.text(.025, .895, "20 held-out contracts / 340 tasks per condition / Qwen3-4B", color="#63736b", fontsize=11)
    fig.text(.025, .025, "Single training seed; fixed hypotheses; GPU migration during evaluation. See methodology for limits.", color="#63736b", fontsize=9)
    fig.subplots_adjust(left=.20, right=.95, top=.83, bottom=.19)
    output = ROOT / "assets/evaluation.svg"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, format="svg", facecolor=fig.get_facecolor(), metadata={"Date": None})
    fig.savefig(output.with_suffix(".png"), dpi=150, facecolor=fig.get_facecolor())
    print(output)
