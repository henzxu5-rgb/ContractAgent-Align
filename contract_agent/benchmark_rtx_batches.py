"""Development-context throughput probe; never reads test tasks or scores labels."""
import argparse
import time
from .data import DatasetStore
from .models import HFModel
from .runner import initial_messages
from .settings import ARTIFACTS
from .storage import write_json, now


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--adapter", required=True)
    args = p.parse_args()
    store = DatasetStore()
    prompts = [initial_messages(store.public(tid)) for tid in store.select("dev_pilot")[:16]]
    model = HFModel(args.model_path, args.adapter)
    results = []
    for size in (8, 16):
        try:
            model.generate_many(prompts[:size], max_tokens=1)
            model.torch.cuda.synchronize()
            model.torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            responses = []
            for i in range(0, len(prompts), size): responses.extend(model.generate_many(prompts[i:i+size], max_tokens=96))
            model.torch.cuda.synchronize()
            seconds = time.monotonic() - start
            tokens = sum(r.get("usage", {}).get("completion_tokens", 0) for r in responses)
            results.append({"batch_cap": size, "seconds": seconds, "output_tokens": tokens,
                            "tokens_per_second": tokens / seconds, "errors": sum("error" in r for r in responses),
                            "peak_gb": model.torch.cuda.max_memory_allocated() / 2**30})
        except Exception as exc:
            results.append({"batch_cap": size, "error_type": type(exc).__name__, "error": str(exc)[:300]})
    write_json(ARTIFACTS / "reports/rtx6000d-batch-probe.json", {"created": now(), "development_contexts": len(prompts),
               "results": results, "scope": "Single next-action throughput; not model quality or end-to-end speed guarantee"})
    print(results, flush=True)


if __name__ == "__main__": main()
