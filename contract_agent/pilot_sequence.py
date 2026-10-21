"""Bounded first-phase jobs. No main-phase work is launched by this module."""
from pathlib import Path
import argparse
import subprocess
import sys
import time

from .settings import ARTIFACTS
from .storage import now, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--steps", type=int, default=25)
    args = parser.parse_args()
    version = args.version
    start = time.monotonic()
    marker = ARTIFACTS / "cache/qwen3-4b/DOWNLOAD_COMPLETE"
    while not marker.exists():
        if time.monotonic() - start > 1800:
            raise TimeoutError("Weights not ready within the pilot wait cap")
        time.sleep(15)
    model = str(marker.parent)
    commands = [
        [sys.executable, "-u", "-m", "contract_agent.cli", "run", "--backend", "hf", "--selection", "dev_pilot", "--mode", "both", "--limit", "18", "--model-path", model, "--batch-id", "pilot-qwen-base-v1"],
        [sys.executable, "-u", "-m", "contract_agent.training", "sft", "--data", str(ARTIFACTS / "datasets/pilot-sft.jsonl"), "--model-path", model, "--run-id", "pilot-sft-v1", "--pilot", "--max-steps", "20"],
        [sys.executable, "-u", "-m", "contract_agent.training", "dpo", "--data", str(ARTIFACTS / "datasets/pilot-dpo.jsonl"), "--model-path", model, "--sft-adapter", str(ARTIFACTS / "models/pilot-sft-v1"), "--run-id", "pilot-dpo-v1", "--pilot", "--max-steps", "20"],
    ]
    commands = [[part.replace("-v1", "-" + version) for part in command] for command in commands]
    for command in commands:
        if "--max-steps" in command:
            command[command.index("--max-steps") + 1] = str(args.steps)
    if args.skip_baseline:
        commands = commands[1:]
    receipt = {"created": now(), "status": "running", "commands": commands, "completed": []}
    path = ARTIFACTS / "reports" / ("pilot_sequence-" + version + ".json")
    write_json(path, receipt)
    for command in commands:
        print("PILOT_COMMAND", command, flush=True)
        try:
            subprocess.run(command, check=True, timeout=1500)
        except Exception as exc:
            receipt.update(status="failed", error_type=type(exc).__name__, stopped_command=command)
            write_json(path, receipt)
            raise
        receipt["completed"].append(command)
        write_json(path, receipt)
    receipt.update(status="completed", seconds=time.monotonic() - start)
    write_json(path, receipt)
    print("PILOT_SEQUENCE_COMPLETED", flush=True)


if __name__ == "__main__": main()
