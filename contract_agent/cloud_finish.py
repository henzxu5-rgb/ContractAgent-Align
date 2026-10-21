"""Finite, SSH-independent completion of the frozen evaluation; no API key needed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .settings import ARTIFACTS, MODEL_REVISION
from .storage import now, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    if not __import__("re").fullmatch(r"[A-Za-z0-9_-]+", args.id): raise ValueError("Invalid run ID")
    folder = ARTIFACTS / "pipelines" / args.id
    path = folder / "state.json"
    state = {"id": args.id, "created": now(), "status": "running", "pid": os.getpid(),
             "independent_of_local_computer": True, "training_performed": False, "stages": {}}
    write_json(path, state, exclusive=True)
    project = Path(__file__).resolve().parent.parent
    models = ARTIFACTS / "models"
    base = ARTIFACTS / "cache/qwen3-4b"
    try:
        if (base / "DOWNLOAD_COMPLETE").read_text().strip() != MODEL_REVISION:
            raise ValueError("Pinned base model receipt missing")
        uid = "align-v1-rtx6000d-v2"
        for arm in ("raw", "verified"):
            adapter = models / (uid + "-dpo-" + arm)
            if not (adapter / "adapter_model.safetensors").is_file():
                raise FileNotFoundError("Formal DPO adapter missing")
            batch_id = uid + "-test-dpo-" + arm
            batch_path = ARTIFACTS / "batches" / (batch_id + ".json")
            before = len(read_json(batch_path)["runs"]) if batch_path.exists() else 0
            stage = {"started": now(), "status": "running", "previous_saved_runs": before}
            state.update(current_step="test-dpo-" + arm)
            state["stages"][arm] = stage
            write_json(path, state)
            command = [sys.executable, "-u", "-m", "contract_agent.cli", "run", "--backend", "hf",
                       "--selection", "test", "--mode", "agent", "--phase", "main", "--model-path", str(base),
                       "--adapter", str(adapter), "--batch-id", batch_id, "--inference-batch-size", "16"]
            with (folder / (arm + ".log")).open("x") as stream:
                result = subprocess.run(command, cwd=project, stdout=stream, stderr=subprocess.STDOUT, timeout=3000)
            if result.returncode: raise RuntimeError("Evaluation failed: " + arm)
            batch = read_json(batch_path)
            from .data import DatasetStore
            expected = set(DatasetStore().select("test"))
            if len(batch["runs"]) != 340 or {r["task_id"] for r in batch["runs"]} != expected:
                raise ValueError("Incomplete or mismatched evaluation tasks")
            stage.update(status="completed", finished=now(), saved_runs=340)
            write_json(path, state)
        state.update(status="completed", finished=now(), results_saved_on_cloud=True)
        write_json(path, state)
        print(json.dumps(state), flush=True)
    except Exception as exc:
        state.update(status="failed", finished=now(), error_type=type(exc).__name__, error=str(exc)[:300])
        write_json(path, state)
        raise
    finally:
        # Results are durable on the instance; local download happens after the user returns.
        time.sleep(10)
        subprocess.run(["/bin/bash", "-c", "shutdown"], check=True)


if __name__ == "__main__": main()
