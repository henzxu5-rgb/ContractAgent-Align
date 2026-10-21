"""Evidence-checked pilot gate, with conservative measured cost extrapolation."""
import math
from pathlib import Path

from .settings import ARTIFACTS
from .storage import read_json, write_json, now
from .budget import Ledger


def evaluate(root=ARTIFACTS, hourly_cny=3.0):
    root = Path(root)
    receipts = [read_json(p) for p in (root / "training_logs").glob("*/receipt.json")]
    pilot = {r["kind"]: r for r in receipts if r.get("pilot_only") and r.get("status") == "completed" and r.get("optimizer_steps", 0) >= 20 and r.get("adapter_reload_verified")}
    batches = [read_json(p) for p in (root / "batches").glob("*.json")]
    baseline = next((b for b in batches if b["config"].get("backend") in ("hf", "remote") and b["config"].get("selection") == "dev_pilot" and len(b["runs"]) >= 36 and not b["config"].get("adapter")), None)
    reasons = []
    if "sft" not in pilot: reasons.append("SFT: 20 optimizer steps + adapter reload not yet passed")
    if "dpo" not in pilot: reasons.append("DPO: 20 optimizer steps + frozen reference + reload not yet passed")
    if baseline is None: reasons.append("18 paired development tasks on the actual trainable base model not complete")
    costs = Ledger(root / "budget.sqlite").summary()
    predicted = None
    if baseline is not None and len(pilot) == 2:
        agent = baseline["summary"]["agent"]
        full = baseline["summary"]["full_context"]
        # Formal accumulation is 16, whereas the optimizer-step smoke uses accumulation 1.
        sft_seconds = pilot["sft"]["seconds"] / 20 * 16 * math.ceil(120 * 5 / 16) * 2
        dpo_seconds = pilot["dpo"]["seconds"] / 20 * 16 * math.ceil(160 / 16) * 2
        # Up to four Agent conditions, one full baseline, training rollouts and branching.
        inference_seconds = agent["mean_seconds"] * (340 * 4 + 300 + 400 * 2) + full["mean_seconds"] * 340
        predicted = 1.5 * (sft_seconds + dpo_seconds + inference_seconds) / 3600 * hourly_cny
        if predicted > 100 - costs["categories"]["gpu"]:
            reasons.append("Measured full-experiment GPU projection exceeds remaining category cap")
        if predicted + costs["committed_cny"] + 40 > 200:
            reasons.append("Conservative full projection exceeds total cap")
        if agent["status_counts"].get("completed", 0) < 12:
            reasons.append("Base model completed fewer than 12/18 tool tasks; diagnose harness/model before scaling")
    result = {"created": now(), "approved": not reasons, "reasons": reasons,
              "predicted_gpu_cny_upper": predicted, "costs_upper": {k: v for k, v in costs.items() if k != "entries"},
              "pilot_training": {k: v["id"] for k, v in pilot.items()}, "baseline_batch": baseline["id"] if baseline else None,
              "extrapolation": "Measured step times; accumulation correction x16; 50% runtime contingency. A forecast, not a guarantee."}
    write_json(root / "reports/stage_gate.json", result)
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
