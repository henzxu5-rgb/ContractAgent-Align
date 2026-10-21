"""Resume from the real V100 SFT checkpoint on the user-provided RTX instance."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import time

from .pipeline import Controller, remote_json, push_json, REMOTE_ARTIFACTS, MODEL_PATH
from .remote import download
from .settings import ARTIFACTS
from .storage import read_json, write_json, now, digest

UID = "align-v1-rtx6000d-v2"
SFT = REMOTE_ARTIFACTS + "/models/align-v1-sft"
CANDIDATES = UID + "-candidates"
PREFERENCES = UID + "-preferences"


def wait_existing(controller, jid):
    while True:
        controller.lease()
        receipt = controller.network_read(lambda: remote_json("reports/jobs/" + jid + ".json"))
        if receipt and receipt["status"] == "superseded":
            jid = receipt["replacement_job"]
            continue
        if receipt and receipt["status"] in ("failed", "cancelled"):
            raise RuntimeError("Existing setup job failed: " + jid)
        if receipt and receipt["status"] == "completed": return receipt
        time.sleep(30)


def fetch_candidates(controller, name=CANDIDATES):
    batch = controller.network_read(lambda: remote_json("candidate_batches/" + name + "/batch.json"))
    if batch is None: raise FileNotFoundError("Candidate batch is missing")
    path = ARTIFACTS / "candidate_batches" / name / "batch.json"
    write_json(path, batch)
    # Preserve independent raw branch replay artifacts locally as well.
    for row in batch["records"]:
        destination = path.parent / "items" / (row["id"] + ".json")
        if destination.exists():
            if digest(read_json(destination)) != digest(row): raise ValueError("Saved raw candidate changed")
        else: write_json(destination, row, exclusive=True)
        for branch in row["branches"]:
            destination = ARTIFACTS / "runs" / (branch["id"] + ".json")
            if not destination.exists(): write_json(destination, {k: v for k, v in branch.items() if k != "score"}, exclusive=True)
    return str(path)


def produce_and_label(controller, limit):
    command = ["contract_agent.preference_jobs", "generate", "--source", REMOTE_ARTIFACTS + "/batches/align-v1-sft-rollouts.json",
               "--model-path", MODEL_PATH, "--adapter", SFT, "--run-id", CANDIDATES,
               "--limit", str(limit), "--excluded", REMOTE_ARTIFACTS + "/legacy_preference_keys.json"]
    if limit > 50: command += ["--resume", "--parallel", "4"]
    controller.cloud("candidates-" + str(limit), command, 3600)
    path = fetch_candidates(controller)
    controller.local("judge-" + str(limit), ["contract_agent.preference_jobs", "label", "--batch", path,
                     "--run-id", PREFERENCES, "--legacy-dir", str(ARTIFACTS / "preferences/align-v1-preferences")], 7200)
    summary = read_json(ARTIFACTS / "preferences" / PREFERENCES / "summary.json")
    controller.state["preference_summary"] = summary
    controller.save()
    return summary


def run(controller, stage):
    if not read_json(ARTIFACTS / "reports/stage_gate.json").get("approved"):
        raise RuntimeError("Original pilot gate not passed")
    controller.step("bootstrap", lambda: wait_existing(controller, "rtx6000d-bootstrap-v1"))
    controller.step("model-download", lambda: wait_existing(controller, "rtx6000d-model-download-v2"))
    controller.network_read(lambda: push_json(read_json(ARTIFACTS / "reports/stage_gate.json"), "reports/stage_gate.json"))
    controller.cloud("reload-dev", ["contract_agent.cli", "run", "--backend", "hf", "--selection", "dev_pilot", "--limit", "4",
                     "--mode", "agent", "--phase", "pilot", "--model-path", MODEL_PATH, "--adapter", SFT,
                     "--batch-id", "pilot-rtx-reload-v1", "--inference-batch-size", "4"], 600)
    reload_batch = controller.fetch_batch("pilot-rtx-reload-v1")
    if any(r["status"] == "failed" for r in read_json(reload_batch)["runs"]):
        raise RuntimeError("New GPU adapter reload/forward failed; do not expand")
    controller.cloud("compat-sft", ["contract_agent.training", "sft", "--data", REMOTE_ARTIFACTS + "/datasets/pilot-sft.jsonl",
                     "--model-path", MODEL_PATH, "--pilot", "--max-steps", "25", "--run-id", "pilot-rtx-sft-v1"], 900)
    controller.cloud("compat-dpo", ["contract_agent.training", "dpo", "--data", REMOTE_ARTIFACTS + "/datasets/pilot-dpo.jsonl",
                     "--model-path", MODEL_PATH, "--sft-adapter", REMOTE_ARTIFACTS + "/models/pilot-rtx-sft-v1",
                     "--pilot", "--max-steps", "25", "--run-id", "pilot-rtx-dpo-v1"], 1200)
    produce_and_label(controller, 50)
    if stage == "first-batch":
        controller.state.update(status="first_batch_ready_for_data_gate", finished=now())
        controller.save(); return
    summary = produce_and_label(controller, 383)
    finish(controller, summary, PREFERENCES)


def run_temperature(controller, complete=False):
    if controller.state["steps"].get("judge-50", {}).get("status") != "completed":
        raise ValueError("Original 50-pair data gate must finish before the documented temperature probe")
    name, prefname = CANDIDATES + "-temperature12", PREFERENCES + "-temperature12"
    for limit in ([20, 333] if complete else [20]):
        command = ["contract_agent.preference_jobs", "generate", "--source", REMOTE_ARTIFACTS + "/batches/align-v1-sft-rollouts.json",
                   "--model-path", MODEL_PATH, "--adapter", SFT, "--run-id", name, "--limit", str(limit),
                   "--excluded", REMOTE_ARTIFACTS + "/preference_keys_before_temperature_probe.json",
                   "--parallel", "4", "--temperature", "1.2"]
        if limit > 20: command.append("--resume")
        controller.cloud("candidates-temperature12-" + str(limit), command, 3600)
        path = fetch_candidates(controller, name)
        controller.local("judge-temperature12-" + str(limit), ["contract_agent.preference_jobs", "label", "--batch", path,
                         "--run-id", prefname, "--legacy-dir", str(ARTIFACTS / "preferences" / PREFERENCES / "items"),
                         "--reuse-legacy-annotations"], 7200)
        summary = read_json(ARTIFACTS / "preferences" / prefname / "summary.json")
        controller.state.update(preference_summary=summary, preference_output=prefname)
        controller.save()
    if not complete:
        controller.state.update(status="temperature_probe_ready_for_data_gate", finished=now())
        controller.save(); return
    finish(controller, summary, prefname)


def finish(controller, summary, preference_name):
    arms = []
    if summary["minimum_data_pass"]:
        from .remote import upload
        for kind in (["raw", "verified"] if summary["train_raw_control"] else ["verified"]):
            target = ARTIFACTS / "datasets" / (UID + "-" + kind + ".jsonl")
            source = Path(summary[kind + "_file"])
            if target.exists():
                if target.read_bytes() != source.read_bytes(): raise ValueError("Frozen DPO dataset changed")
            else: shutil.copyfile(source, target)
            arms.append(kind)
        controller.step("deploy-dpo", upload)
        for kind in arms:
            controller.cloud("dpo-" + kind, ["contract_agent.training", "dpo", "--data", REMOTE_ARTIFACTS + "/datasets/" + UID + "-" + kind + ".jsonl",
                             "--model-path", MODEL_PATH, "--sft-adapter", SFT, "--run-id", UID + "-dpo-" + kind], 10800)
    controller.state.update(training_arms=["sft", *["dpo-" + kind for kind in arms]], test_config_frozen_at=now())
    controller.save()
    conditions = {}
    for condition in ["base", "sft", *["dpo-" + kind for kind in arms]]:
        batch_id = UID + "-test-" + condition
        command = ["contract_agent.cli", "run", "--backend", "hf", "--selection", "test", "--mode", "both" if condition == "base" else "agent",
                   "--phase", "main", "--model-path", MODEL_PATH, "--batch-id", batch_id, "--inference-batch-size", "16"]
        if condition != "base": command += ["--adapter", SFT if condition == "sft" else REMOTE_ARTIFACTS + "/models/" + UID + "-" + condition]
        controller.cloud("test-" + condition, command, 18000)
        conditions[condition.replace("-", "_")] = controller.fetch_batch(batch_id)
    conditions["full_context"] = conditions["base"]
    from .final_report import build
    controller.step("independent-report", lambda: {"status": build(conditions,
                    preference_summary_path=ARTIFACTS / "preferences" / preference_name / "summary.json",
                    sft_summary_path=ARTIFACTS / "datasets/align-v1-sft.summary.json")["status"]})
    controller.step("download-artifacts", download)
    controller.state.update(status="completed" if arms else "completed_with_insufficient_preference_data", finished=now())
    controller.save()
    deadline = {"deadline_utc": (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat(), "reason": "All local results saved; stop idle billing"}
    push_json(deadline, "cloud_deadline.json"); write_json(ARTIFACTS / "cloud_deadline.json", deadline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("first-batch", "complete", "temperature-probe", "complete-temperature"), default="first-batch")
    args = parser.parse_args()
    controller = Controller(UID)
    controller.state.update(training_origin="align-v1-sft on V100S; checkpoint restored and hash verified",
                            inference_hardware="NVIDIA RTX 6000D", candidate_protocol="manifest_tasks_rotating_early_middle_final_v2")
    controller.save()
    try:
        if args.stage in ("temperature-probe", "complete-temperature"):
            run_temperature(controller, args.stage == "complete-temperature")
        else: run(controller, args.stage)
    except Exception as exc:
        controller.state.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:300])
        controller.save(); raise


if __name__ == "__main__": main()
