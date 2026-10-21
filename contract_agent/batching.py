"""Batch only inference calls; each original episode keeps its own environment."""
from dataclasses import dataclass, field
import queue
import threading
import time


@dataclass
class Request:
    messages: list
    temperature: float
    max_tokens: int
    ready: threading.Event = field(default_factory=threading.Event)
    response: object = None
    error: object = None


class BatchingModel:
    scheduler = "dynamic_gpu_batch_v1"
    def __init__(self, model, batch_size=8, gather_seconds=.02):
        self.model = model
        self.name, self.adapter = model.name, getattr(model, "adapter", None)
        self.batch_size, self.gather_seconds = batch_size, gather_seconds
        self.requests = queue.Queue()
        self.closed = False
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def generate(self, messages, temperature=0.0, max_tokens=384):
        if self.closed: raise RuntimeError("Inference batcher is closed")
        item = Request(messages, temperature, max_tokens)
        self.requests.put(item)
        if not item.ready.wait(timeout=600):
            raise TimeoutError("Inference batch did not return")
        if item.error: raise item.error
        return item.response

    def generate_many(self, message_groups, temperature=0.0, max_tokens=384):
        """Enqueue a pair together; only the single GPU worker calls the model."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(message_groups)) as pool:
            futures = [pool.submit(self.generate, group, temperature, max_tokens) for group in message_groups]
            return [f.result() for f in futures]

    def _serve(self):
        backlog = []
        while True:
            first = backlog.pop(0) if backlog else self.requests.get()
            if first is None: return
            group = [first]
            end = time.monotonic() + self.gather_seconds
            while len(group) < self.batch_size:
                remaining = end - time.monotonic()
                if remaining <= 0: break
                try:
                    item = self.requests.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    backlog.append(None); break
                if (item.temperature, item.max_tokens) == (first.temperature, first.max_tokens):
                    group.append(item)
                else:
                    backlog.append(item)
            try:
                responses = self.model.generate_many([r.messages for r in group], temperature=first.temperature, max_tokens=first.max_tokens)
                if len(responses) != len(group): raise RuntimeError("Batch response count mismatch")
                for item, response in zip(group, responses):
                    if "error" in response:
                        item.error = RuntimeError(response.get("error_type", "InferenceError") + ":" + response["error"])
                    else:
                        item.response = response
            except Exception as exc:
                for item in group: item.error = exc
            finally:
                for item in group: item.ready.set()

    def close(self):
        self.closed = True
        self.requests.put(None)
        self.thread.join(timeout=600)
        if self.thread.is_alive(): raise TimeoutError("Inference worker did not stop")

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
