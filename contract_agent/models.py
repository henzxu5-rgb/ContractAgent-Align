"""One JSON action protocol across providers; secrets never enter records."""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .budget import Ledger
from .settings import INPUT_PRICE, OUTPUT_PRICE, MODEL_ID, MODEL_REVISION, load_env
from .storage import identifier, now, write_json


class DeepSeekModel:
    def __init__(self, ledger=None, phase="pilot", log_dir=None):
        load_env()
        self.key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.key:
            raise ValueError("DEEPSEEK_API_KEY is not configured")
        self.ledger = ledger or Ledger()
        self.phase = phase
        self.log_dir = Path(log_dir) if log_dir else self.ledger.path.parent / "api_calls"
        self.name = "deepseek-flash"

    def generate(self, messages, temperature=0.0, max_tokens=384):
        payload = {"model": self.name, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, "response_format": {"type": "json_object"},
                   "thinking": {"type": "disabled"}, "stream": False}
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        # UTF-8 byte count plus provider framing is a conservative token upper bound.
        input_upper = len(encoded) + 4096
        reserved = input_upper * INPUT_PRICE / 1e6 + max_tokens * OUTPUT_PRICE / 1e6
        rid = self.ledger.reserve(reserved, "api", self.phase, {"requested_model": self.name})
        call_id = identifier("api")
        start = time.monotonic()
        record = {"id": call_id, "created": now(), "messages": messages, "request": {k: v for k, v in payload.items() if k != "messages"},
                  "prices_cny_per_million": {"input_uncached": INPUT_PRICE, "output": OUTPUT_PRICE}, "budget_reservation": rid}
        try:
            req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=encoded,
                    headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as response:
                body = json.load(response)
            usage = body.get("usage", {})
            if "prompt_tokens" not in usage or "completion_tokens" not in usage:
                raise ValueError("Provider did not return usage; reservation retained as charge")
            cost = (usage["prompt_tokens"] * INPUT_PRICE + usage["completion_tokens"] * OUTPUT_PRICE) / 1e6
            self.ledger.settle(rid, cost, {"actual_model": body.get("model"), "usage": usage})
            record.update(status="completed", response=body, seconds=time.monotonic() - start, cost_cny_upper=cost)
            write_json(self.log_dir / f"{call_id}.json", record, exclusive=True)
            choice = body["choices"][0]
            return {"text": choice["message"].get("content", ""), "usage": usage,
                    "model": body.get("model"), "call_id": call_id, "finish_reason": choice.get("finish_reason"),
                    "seconds": record["seconds"], "cost_cny_upper": cost}
        except Exception as e:
            if isinstance(e, urllib.error.HTTPError):
                record["http_status"] = e.code
                # Provider errors are diagnostic data, never instructions.
                # Never persist Authorization headers or a reflected key.
                detail = e.read(2048).decode("utf-8", errors="replace")
                record["provider_error"] = detail.replace(self.key, "[REDACTED]")
            # A timeout may have been charged. Never release an uncertain reservation for free.
            summary = self.ledger.summary()
            pending = next((r for r in summary["entries"] if r["id"] == rid), None)
            if pending and pending["status"] == "reserved":
                self.ledger.settle(rid, metadata={"uncertain_charge": True, "error_type": type(e).__name__})
            record.update(status="failed", error_type=type(e).__name__, seconds=time.monotonic() - start)
            write_json(self.log_dir / f"{call_id}.json", record, exclusive=True)
            raise RuntimeError(f"API request failed ({type(e).__name__}); see {call_id}") from None


class HFModel:
    def __init__(self, model_path=None, adapter=None, four_bit=True, context_limit=4096):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        from .attention import configure_attention
        self.attention_policy = configure_attention()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for the selected model; no silent CPU fallback")
        self.torch = torch
        self.context_limit = context_limit
        from .provenance import model_source
        path, revision = model_source(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True) if four_bit else None
        self.model = AutoModelForCausalLM.from_pretrained(path, revision=revision, torch_dtype=torch.float16,
                        device_map={"": 0}, quantization_config=config, attn_implementation="sdpa")
        self.name = MODEL_ID
        self.adapter = str(adapter) if adapter else None
        if adapter:
            if not (Path(adapter) / "adapter_config.json").is_file():
                raise FileNotFoundError(f"Requested adapter is missing: {adapter}")
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False)
        self.model.eval()

    def generate_many(self, message_groups, temperature=0.0, max_tokens=384):
        """One GPU batch; each output retains its own prompt length and failure."""
        torch = self.torch
        rendered = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in message_groups]
        lengths = [len(self.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in rendered]
        results = [None] * len(rendered)
        valid = []
        for index, length in enumerate(lengths):
            if length + max_tokens > self.context_limit:
                results[index] = {"error_type": "ValueError", "error": f"context_overflow:{length}+{max_tokens}>{self.context_limit}"}
            else:
                valid.append(index)
        if not valid:
            return results
        start = time.monotonic()
        previous_side, previous_pad = self.tokenizer.padding_side, self.tokenizer.pad_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            encoded = self.tokenizer([rendered[i] for i in valid], padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
        finally:
            self.tokenizer.padding_side, self.tokenizer.pad_token = previous_side, previous_pad
        width = encoded["input_ids"].shape[1]
        args = {"max_new_tokens": max_tokens, "do_sample": temperature > 0, "pad_token_id": self.tokenizer.eos_token_id}
        args.update({"temperature": temperature, "top_p": .9} if temperature > 0 else {"temperature": 1.0, "top_p": 1.0, "top_k": 50})
        oom = False
        try:
            with torch.inference_mode():
                output = self.model.generate(**encoded, **args)
        except torch.OutOfMemoryError:
            oom = True
        if oom:
            # Capacity batching must not manufacture failures in the full baseline.
            del encoded
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            if len(valid) == 1:
                results[valid[0]] = {"error_type": "OutOfMemoryError", "error": "Single input exhausted GPU memory"}
                return results
            midpoint = len(valid) // 2
            for group in (valid[:midpoint], valid[midpoint:]):
                for i, response in zip(group, self.generate_many([message_groups[j] for j in group], temperature, max_tokens)):
                    results[i] = response
            return results
        seconds = time.monotonic() - start
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for row, index in enumerate(valid):
            ids = output[row, width:].tolist()
            end = next((j + 1 for j, token in enumerate(ids) if token in eos), len(ids))
            ids = ids[:end]
            results[index] = {"text": self.tokenizer.decode(ids, skip_special_tokens=True),
                "model": self.name, "adapter": self.adapter, "model_revision": MODEL_REVISION,
                "attention_policy": self.attention_policy, "inference_batch_size": len(valid),
                "usage": {"prompt_tokens": lengths[index], "completion_tokens": len(ids)},
                "seconds": seconds, "amortized_seconds": seconds / len(valid)}
        return results

    def generate(self, messages, temperature=0.0, max_tokens=384):
        torch = self.torch
        start = time.monotonic()
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        length = encoded["input_ids"].shape[1]
        if length + max_tokens > self.context_limit:
            raise ValueError(f"context_overflow:{length}+{max_tokens}>{self.context_limit}")
        encoded = encoded.to("cuda")
        args = {"max_new_tokens": max_tokens, "do_sample": temperature > 0, "pad_token_id": self.tokenizer.eos_token_id}
        if temperature > 0:
            args.update(temperature=temperature, top_p=.9)
        else:
            args.update(temperature=1.0, top_p=1.0, top_k=50)
        with torch.inference_mode():
            output = self.model.generate(**encoded, **args)
        ids = output[0, length:]
        return {"text": self.tokenizer.decode(ids, skip_special_tokens=True),
                "model": self.name, "adapter": self.adapter, "model_revision": MODEL_REVISION,
                "attention_policy": self.attention_policy,
                "usage": {"prompt_tokens": length, "completion_tokens": len(ids)},
                "seconds": time.monotonic() - start}


class RemoteHFModel:
    """Inference over an SSH channel, without publishing a model port."""
    def __init__(self, adapter=None, port=9003, context_limit=4096):
        from .remote import connect
        self.client = connect()
        self.port = port
        self.context_limit = context_limit
        self.name = MODEL_ID
        self.adapter = adapter
        info = self.request("GET", "/info")
        actual = Path(info["adapter"]).name if info.get("adapter") else None
        expected = Path(adapter).name if adapter else None
        if actual != expected or info.get("model_revision") != MODEL_REVISION:
            self.client.close()
            raise ValueError(f"Remote model identity mismatch: expected adapter={expected}, returned adapter={actual}")

    def request(self, method, path, body=None):
        import http.client
        transport = self.client.get_transport()
        channel = transport.open_channel("direct-tcpip", ("127.0.0.1", self.port), ("127.0.0.1", 0), timeout=180)
        channel.settimeout(180)
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=180)
        connection.sock = channel
        try:
            connection.request(method, path, body=json.dumps(body).encode() if body is not None else None,
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            result = json.load(response)
            if response.status != 200:
                raise RuntimeError(result.get("error", "remote_model_error"))
            return result
        finally:
            connection.close()

    def generate(self, messages, temperature=0.0, max_tokens=384):
        return self.request("POST", "/generate", {"messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens, "context_limit": self.context_limit})

    def shutdown(self):
        return self.request("POST", "/shutdown", {})

    def generate_many(self, message_groups, temperature=0.0, max_tokens=384):
        return self.request("POST", "/generate_batch", {"message_groups": message_groups, "temperature": temperature,
                            "max_tokens": max_tokens, "context_limit": self.context_limit})

    def close(self):
        self.client.close()
