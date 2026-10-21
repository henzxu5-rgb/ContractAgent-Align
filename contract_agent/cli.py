import argparse
import json
import time
from pathlib import Path

from .settings import ARTIFACTS, MODEL_ID
from .storage import identifier, read_json, write_json, now


def main():
    parser = argparse.ArgumentParser(description="ContractAgent-Align: real executions and reproducible data")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("budget")
    run = sub.add_parser("run")
    run.add_argument("--backend", choices=("deepseek", "hf", "remote"), default="deepseek")
    run.add_argument("--selection", choices=("train", "train_initial", "dev_pilot", "test"), default="dev_pilot")
    run.add_argument("--task")
    run.add_argument("--limit", type=int)
    run.add_argument("--mode", choices=("agent", "full_context", "both"), default="both")
    run.add_argument("--phase", choices=("pilot", "main"), default="pilot")
    run.add_argument("--model-path")
    run.add_argument("--adapter")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--batch-id")
    run.add_argument("--target-successes", type=int)
    run.add_argument("--inference-batch-size", type=int, default=1)
    export = sub.add_parser("export-sft")
    export.add_argument("--batch", action="append", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--pilot", action="store_true")
    pair = sub.add_parser("pairs")
    pair.add_argument("--batch", required=True)
    pair.add_argument("--adapter")
    pair.add_argument("--limit", type=int, default=400, help="Hard candidate-pair cap, including abstentions")
    pair.add_argument("--target", type=int, default=160, help="Stop after this many retained pairs and >=20 tool pairs")
    pair.add_argument("--pilot", action="store_true")
    pair.add_argument("--run-id")
    pair.add_argument("--resume", action="store_true", help="Explicitly resume saved candidate/branch/judge stages after diagnosis")
    report = sub.add_parser("report")
    report.add_argument("--batch", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare
        print(json.dumps(prepare(), ensure_ascii=False, default=dict)); return
    if args.command == "budget":
        from .budget import Ledger
        print(json.dumps(Ledger().summary(), ensure_ascii=False, indent=2)); return
    if args.command == "export-sft":
        from .preferences import export_sft
        print(json.dumps(export_sft(args.batch, args.output, args.pilot))); return
    if args.command == "pairs":
        from .preferences import produce_pairs
        from .models import RemoteHFModel, DeepSeekModel
        if not args.pilot and not read_json(ARTIFACTS / "reports/stage_gate.json").get("approved"):
            raise RuntimeError("Formal preference production requires a passed pilot gate")
        if not args.pilot and not args.adapter:
            raise ValueError("Formal preferences must use the SFT policy, not the base model")
        policy = RemoteHFModel(args.adapter)
        try:
            print(json.dumps(produce_pairs(args.batch, policy, DeepSeekModel(phase="pilot" if args.pilot else "main"), args.limit, args.pilot, target=args.target, output_id=args.run_id, resume=args.resume)))
        finally:
            policy.close()
        return
    if args.command == "report":
        from .analysis import report_batch
        print(json.dumps(report_batch(args.batch))); return
    if args.command == "run":
        from .data import DatasetStore
        from .models import DeepSeekModel, HFModel, RemoteHFModel
        from .runner import run_episode
        from .scoring import score_run, summarize
        store = DatasetStore()
        if args.phase == "main":
            gate = read_json(ARTIFACTS / "reports/stage_gate.json")
            if not gate.get("approved"):
                raise RuntimeError("Main-phase sampling/evaluation requires a passed pilot gate")
        if args.target_successes and (args.mode != "agent" or args.selection not in ("train", "train_initial")):
            raise ValueError("Target successes is for agent training-data collection only")
        if not 1 <= args.inference_batch_size <= 16:
            raise ValueError("Inference batch size must be 1..16")
        if args.inference_batch_size > 1 and (args.backend != "hf" or args.target_successes):
            raise ValueError("Batched evaluation requires local HF inference and no target-based early stop")
        if args.selection == "test" and args.phase == "pilot":
            raise ValueError("Pilot cannot use frozen test tasks")
        if args.task and args.task not in store.select(args.selection):
            raise ValueError("Task must belong to the declared selection")
        ids = [args.task] if args.task else store.select(args.selection, args.limit)
        modes = ("full_context", "agent") if args.mode == "both" else (args.mode,)
        model = DeepSeekModel(phase=args.phase) if args.backend == "deepseek" else (RemoteHFModel(args.adapter) if args.backend == "remote" else HFModel(args.model_path, args.adapter, context_limit=16384 if "full_context" in modes else 4096))
        batch_id = args.batch_id or identifier("batch")
        path = ARTIFACTS / "batches" / f"{batch_id}.json"
        batch = read_json(path) if path.exists() else {"id": batch_id, "created": now(), "config": vars(args), "runs": []}
        old_config = {"inference_batch_size": 1, **batch["config"]}
        if path.exists() and old_config != vars(args):
            raise ValueError("A batch ID cannot be resumed with a changed configuration")
        done = {(r["task_id"], r["mode"]) for r in batch["runs"]}
        wall_start, previous_wall = time.monotonic(), batch.get("wall_seconds", 0)

        def record(run):
            run["score"] = score_run(run, store.gold(run["task_id"]))
            batch["runs"].append(run)
            batch["wall_seconds"] = previous_wall + time.monotonic() - wall_start
            batch["summary"] = {m: summarize([r for r in batch["runs"] if r["mode"] == m]) for m in modes}
            write_json(path, batch)
            print(json.dumps({"batch": batch_id, "task": run["task_id"], "mode": run["mode"], "status": run["status"], "label_correct": run["score"]["label_correct"], "evidence_f1": (run["score"]["evidence"] or {}).get("f1"), "seconds": run["seconds"]}), flush=True)
            if run.get("error") and "Budget" in run.get("error_type", ""):
                raise RuntimeError("Budget exhausted; batch checkpoint retained")

        if args.inference_batch_size > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from .batching import BatchingModel
            for mode in modes:
                model.context_limit = 16384 if mode == "full_context" else 4096
                pending = [tid for tid in ids if (tid, mode) not in done]
                with BatchingModel(model, args.inference_batch_size) as batched:
                    with ThreadPoolExecutor(max_workers=args.inference_batch_size) as pool:
                        futures = [pool.submit(run_episode, store.public(tid), batched, mode=mode, temperature=args.temperature) for tid in pending]
                        for future in as_completed(futures): record(future.result())
            print(json.dumps(batch["summary"], ensure_ascii=False))
            return
        for tid in ids:
            if args.target_successes and sum(r.get("score", {}).get("strict_success", False) for r in batch["runs"]) >= args.target_successes:
                break
            for mode in modes:
                if (tid, mode) in done: continue
                if hasattr(model, "context_limit"): model.context_limit = 16384 if mode == "full_context" else 4096
                run = run_episode(store.public(tid), model, mode=mode, temperature=args.temperature)
                record(run)
        print(json.dumps(batch["summary"], ensure_ascii=False))
        if hasattr(model, "close"): model.close()


if __name__ == "__main__": main()
