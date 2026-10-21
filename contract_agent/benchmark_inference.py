"""Bounded development-only latency check after the ongoing rollout finishes."""
import argparse
import time

from .data import DatasetStore
from .environment import ContractEnvironment
from .models import HFModel
from .runner import initial_messages, observation_message
from .settings import ARTIFACTS
from .storage import read_json, write_json, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter", required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + 1200
    job = ARTIFACTS / "reports/jobs/align-v1-sft-rollouts.json"
    while read_json(job)["status"] == "running":
        if time.monotonic() >= deadline: raise TimeoutError("Rollout wait expired")
        time.sleep(10)
    if read_json(job)["status"] != "completed": raise RuntimeError("Rollout job failed")
    model = HFModel(args.model_path, args.adapter)
    store = DatasetStore()
    groups = []
    for index, tid in enumerate(store.select("dev_pilot", 8)):
        task = store.public(tid)
        messages = initial_messages(task)
        if index % 2:
            env = ContractEnvironment(task)
            action = {"tool": "search_contract", "arguments": {"query": task["hypothesis"], "k": 5}}
            import json
            messages += [{"role": "assistant", "content": json.dumps(action)}, observation_message(env.execute(action), 5)]
        groups.append(messages)
    model.generate(groups[0], max_tokens=16)
    start = time.monotonic()
    serial = [model.generate(g, max_tokens=128) for g in groups]
    serial_seconds = time.monotonic() - start
    start = time.monotonic()
    batched = model.generate_many(groups, max_tokens=128)
    batch_seconds = time.monotonic() - start
    result = {"created": now(), "scope": "8 development contexts, one next action; not a task quality benchmark",
              "serial_seconds": serial_seconds, "batch_seconds": batch_seconds,
              "speedup": serial_seconds / batch_seconds,
              "greedy_text_agreement": sum(a["text"] == b.get("text") for a, b in zip(serial, batched)),
              "n": len(groups), "serial": serial, "batched": batched,
              "batch_errors": sum("error" in r for r in batched),
              "max_memory_allocated_gb": model.torch.cuda.max_memory_allocated() / 2**30}
    write_json(ARTIFACTS / "reports/inference_batch_benchmark.json", result)
    print({k: v for k, v in result.items() if k not in ("serial", "batched")}, flush=True)


if __name__ == "__main__": main()
