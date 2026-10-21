"""Versioned QLoRA SFT/DPO, single-action loss and mandatory adapter reload."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time

from .settings import ARTIFACTS, MODEL_ID, MODEL_REVISION, SEED
from .storage import read_jsonl, write_json, identifier, now


def render_action(tokenizer, prompt, completion):
    prefix = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(prompt + completion, tokenize=False, add_generation_prompt=False)
    if not full.startswith(prefix):
        raise ValueError("Chat template changed the prompt at completion boundary")
    return prefix, full[len(prefix):].rstrip()


def prepare_rows(rows, tokenizer, kind, pilot=False, max_length=4096):
    kept, excluded = [], []
    for row in rows:
        if row.get("split") != "train" and not (pilot and row.get("split") == "dev" and row.get("pilot_only")):
            raise ValueError("Training/test leakage or unmarked development smoke data")
        if row.get("pilot_only") and not pilot:
            raise ValueError("Pilot-only examples cannot enter formal training")
        prompt, completion = render_action(tokenizer, row["prompt"], row["completion"] if kind == "sft" else row["chosen"])
        item = {"prompt": prompt, "completion" if kind == "sft" else "chosen": completion}
        if kind == "dpo":
            other_prompt, item["rejected"] = render_action(tokenizer, row["prompt"], row["rejected"])
            if other_prompt != prompt:
                raise ValueError("DPO prompt mismatch")
        parts = [item["completion"]] if kind == "sft" else [item["chosen"], item["rejected"]]
        lengths = [len(tokenizer(prompt + c, add_special_tokens=False)["input_ids"]) for c in parts]
        plen = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if max(lengths) > max_length or min(lengths) <= plen:
            excluded.append({"id": row["id"], "lengths": lengths, "reason": "overlength_or_no_action_tokens"})
            continue
        kept.append(item)
    if not kept:
        raise ValueError("No valid action-supervision examples remain")
    return kept, excluded


def parameter_fingerprint(model, reference=False):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if ((".reference." in name) if reference else p.requires_grad):
            h.update(name.encode())
            h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def train(args):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer
    from .attention import configure_attention
    attention_policy = configure_attention()

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device; refusing to silently change model or training mode")
    set_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    uid = args.run_id or identifier("train-" + args.kind)
    output = ARTIFACTS / "models" / uid
    if output.exists():
        raise FileExistsError("Training output exists; use a new run ID rather than overwrite history")
    output.mkdir(parents=True)
    logdir = ARTIFACTS / "training_logs" / uid
    logdir.mkdir(parents=True)
    receipt = {"id": uid, "created": now(), "kind": args.kind, "pilot_only": args.pilot,
               "config": vars(args), "base_model": MODEL_ID, "revision": MODEL_REVISION,
               "status": "running", "gpu": torch.cuda.get_device_name(0), "attention_policy": attention_policy,
               "fingerprint_scope": "full_trainable_and_reference_tensors"}
    write_json(logdir / "receipt.json", receipt)
    from .provenance import model_source
    source, revision = model_source(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    source_rows = read_jsonl(args.data)
    rows, excluded = prepare_rows(source_rows, tokenizer, args.kind, args.pilot)
    if not args.pilot:
        from .storage import read_json
        gate = read_json(ARTIFACTS / "reports/stage_gate.json")
        if not gate.get("approved"):
            raise RuntimeError("Formal training requires a passed pilot gate")
        if args.kind == "dpo" and len(rows) < 80:
            raise ValueError("Formal DPO requires at least 80 matched valid action pairs")
        excluded_ids = {r["id"] for r in excluded}
        retained_source = [r for r in source_rows if r["id"] not in excluded_ids]
        if args.kind == "sft" and len({r.get("source_run") for r in retained_source if r.get("source_run")}) < 80:
            raise ValueError("Formal SFT requires at least 80 accepted demonstration trajectories")
        if args.kind == "dpo":
            from .preferences import tool_behavior_difference
            tools = sum(tool_behavior_difference([{"text": r["chosen"][0]["content"]}, {"text": r["rejected"][0]["content"]}]) for r in retained_source)
            if tools < 20:
                raise ValueError("Formal DPO requires >=20 pairs with differing tool behavior")
            source_receipt = read_json(ARTIFACTS / "training_logs" / Path(args.sft_adapter or "missing").name / "receipt.json")
            if source_receipt.get("pilot_only") or source_receipt.get("status") != "completed":
                raise ValueError("Formal DPO requires a completed formal SFT checkpoint")
    receipt.update(samples=len(rows), excluded=excluded)
    write_json(logdir / "receipt.json", receipt)
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(source, revision=revision, torch_dtype=torch.float16,
                    device_map={"": 0}, quantization_config=config, attn_implementation="sdpa")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.kind == "sft":
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lora)
    else:
        if not args.sft_adapter or not (Path(args.sft_adapter) / "adapter_config.json").is_file():
            raise FileNotFoundError("DPO requires the exact SFT adapter; base fallback forbidden")
        model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True, adapter_name="default")
        model.load_adapter(args.sft_adapter, adapter_name="reference", is_trainable=False)
        model.set_adapter("default")
        for name, p in model.named_parameters():
            if ".reference." in name:
                p.requires_grad_(False)
    initial = parameter_fingerprint(model)
    reference_before = parameter_fingerprint(model, reference=True) if args.kind == "dpo" else None
    successful_updates = 0

    class AuditCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            nonlocal successful_updates
            if not trainer.accelerator.optimizer_step_was_skipped:
                successful_updates += 1

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                item = {"step": state.global_step, **logs}
                for key, v in logs.items():
                    if isinstance(v, (int, float)) and not math.isfinite(v):
                        if key == "grad_norm" and trainer.accelerator.optimizer_step_was_skipped:
                            item[key] = None
                            item["fp16_overflow_skipped_update"] = True
                        else:
                            raise FloatingPointError("Non-finite training metric: " + key)
                with (logdir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(item) + "\n")
                print(json.dumps(item), flush=True)

    # Smoke runs use accumulation=1; formal runs use the preregistered 16.
    common = dict(output_dir=str(output), per_device_train_batch_size=1,
                  gradient_accumulation_steps=1 if args.pilot else 16,
                  gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                  fp16=True, bf16=False, logging_steps=1, save_strategy="no", report_to="none",
                  seed=SEED, data_seed=SEED, optim="adamw_torch", max_grad_norm=1.0,
                  warmup_ratio=.05, lr_scheduler_type="cosine", dataloader_num_workers=0,
                  max_steps=args.max_steps if args.max_steps is not None else -1,
                  max_length=4096, remove_unused_columns=True)
    if args.kind == "sft":
        training_args = SFTConfig(**common, learning_rate=1e-4, num_train_epochs=2,
                                 completion_only_loss=True, packing=False)
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=Dataset.from_list(rows),
                             processing_class=tokenizer, callbacks=[AuditCallback()])
        # Inspect a real trainer batch, not just the source format.
        batch = next(iter(trainer.get_train_dataloader()))
        if not (batch["labels"] == -100).any() or not (batch["labels"] != -100).any():
            raise ValueError("Missing prompt mask or missing assistant supervision")
        receipt["masked_prompt_tokens"] = int((batch["labels"] == -100).sum())
        receipt["supervised_action_tokens"] = int((batch["labels"] != -100).sum())
        first_target = int((batch["labels"][0] != -100).nonzero()[0])
        receipt["first_batch_target_start"] = first_target
        audit_row = trainer.train_dataset[0]
        audit_batch = trainer.data_collator([audit_row])
        expected_mask = audit_row["completion_mask"]
        labels = audit_batch["labels"][0].tolist()[:len(expected_mask)]
        if any((label != -100) != bool(mask) for label, mask in zip(labels, expected_mask)):
            raise ValueError("Trainer loss mask differs from the assistant completion boundary")
        receipt["completion_mask_verified"] = True
        del audit_batch
        del batch
    else:
        training_args = DPOConfig(**common, learning_rate=5e-6, num_train_epochs=1, beta=.1,
                                 max_prompt_length=4096, loss_type="sigmoid", precompute_ref_log_probs=True,
                                 model_adapter_name="default", ref_adapter_name="reference")
        trainer = DPOTrainer(model=model, args=training_args, train_dataset=Dataset.from_list(rows),
                             processing_class=tokenizer, callbacks=[AuditCallback()])
        for row in trainer.train_dataset:
            if any(len(row["prompt_input_ids"]) + len(row[key]) > 4096 for key in ("chosen_input_ids", "rejected_input_ids")):
                raise ValueError("DPO preprocessing exceeded the prevalidated context; refusing internal truncation")
        receipt["dpo_no_internal_truncation_verified"] = True
    # TRL 0.21 prepares an already-quantized SFT model again, freezing existing
    # adapters. Reactivate only the policy adapter after trainer construction.
    model.set_adapter("default")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" in name and ".default." in name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not trainable:
        raise RuntimeError("Trainer initialization left no trainable policy parameters")
    receipt["trainable_policy_parameters"] = trainable
    if trainer.accelerator.scaler is not None:
        scaler_state = trainer.accelerator.scaler.state_dict()
        scaler_state["scale"] = 4096.0
        trainer.accelerator.scaler.load_state_dict(scaler_state)
        receipt["fp16_initial_loss_scale"] = 4096.0
    write_json(logdir / "receipt.json", receipt)
    start = time.monotonic()
    try:
        result = trainer.train()
        if args.pilot and successful_updates < 20:
            raise RuntimeError("Fewer than 20 actual optimizer updates after FP16 scaler skips")
        final = parameter_fingerprint(model)
        if initial == final:
            raise RuntimeError("Trainable adapter did not change")
        if args.kind == "dpo" and any(p.requires_grad for n, p in model.named_parameters() if ".reference." in n):
            raise RuntimeError("Reference adapter became trainable")
        reference_after = parameter_fingerprint(model, reference=True) if args.kind == "dpo" else None
        if reference_after != reference_before:
            raise RuntimeError("Reference adapter weights changed during DPO")
        model.save_pretrained(output, selected_adapters=["default"])
        tokenizer.save_pretrained(output)
        elapsed = time.monotonic() - start
        receipt.update(status="trained", metrics=result.metrics, seconds=elapsed,
                       optimizer_steps=successful_updates, attempted_steps=trainer.state.global_step, initial_fingerprint=initial,
                       final_fingerprint=final, max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                       output_adapter=str(output), reference_fingerprint_before=reference_before,
                       reference_fingerprint_after=reference_after)
        write_json(logdir / "receipt.json", receipt)
        del trainer, model
        gc.collect(); torch.cuda.empty_cache()
        from .models import HFModel
        reloaded = HFModel(args.model_path, adapter=output)
        response = reloaded.generate(read_jsonl(args.data)[0]["prompt"], max_tokens=64)
        receipt.update(status="completed", adapter_reload_verified=True, reload_response=response)
        write_json(logdir / "receipt.json", receipt)
        print(json.dumps(receipt), flush=True)
        return receipt
    except Exception as e:
        receipt.update(status="failed", error_type=type(e).__name__, error=str(e)[:300],
                       optimizer_steps=successful_updates, attempted_steps=trainer.state.global_step)
        write_json(logdir / "receipt.json", receipt)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("sft", "dpo"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--sft-adapter")
    parser.add_argument("--run-id")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.pilot and (args.max_steps is None or args.max_steps < 20):
        raise ValueError("Pilot must explicitly request at least 20 optimizer steps")
    args.run_id = args.run_id or identifier("train-" + args.kind)
    try:
        train(args)
    except Exception as e:
        from .storage import read_json
        receipt_path = ARTIFACTS / "training_logs" / args.run_id / "receipt.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            receipt.update(status="failed", error_type=type(e).__name__, error=str(e)[:300])
            write_json(receipt_path, receipt)
        raise


if __name__ == "__main__": main()
