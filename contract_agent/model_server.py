"""Cloud loopback inference endpoint, accessed only through SSH forwarding."""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from .models import HFModel
from .settings import MODEL_ID, MODEL_REVISION, SEED


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--adapter")
    p.add_argument("--port", type=int, default=9003)
    args = p.parse_args()
    model = HFModel(args.model_path, args.adapter)
    from transformers import set_seed
    set_seed(SEED)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def send(self, status, value):
            b = json.dumps(value).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            if self.path != "/info": return self.send(404, {"error": "not_found"})
            self.send(200, {"model": MODEL_ID, "model_revision": MODEL_REVISION, "adapter": model.adapter, "batch_support": True, "sampling_seed": SEED})
        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size < 500000:
                return self.send(400, {"error": "invalid_request_size"})
            body = json.loads(self.rfile.read(size))
            if self.path == "/shutdown":
                self.send(200, {"shutting_down": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start(); return
            if self.path not in ("/generate", "/generate_batch"): return self.send(404, {"error": "not_found"})
            try:
                limit = body.get("context_limit", 4096)
                if limit not in (4096, 16384): raise ValueError("unsupported_context_limit")
                model.context_limit = limit
                if self.path == "/generate_batch":
                    groups = body["message_groups"]
                    if not isinstance(groups, list) or not 1 <= len(groups) <= 8: raise ValueError("batch_size_must_be_1_to_8")
                    result = model.generate_many(groups, temperature=body.get("temperature", 0), max_tokens=min(384, int(body.get("max_tokens", 384))))
                else:
                    result = model.generate(body["messages"], temperature=body.get("temperature", 0), max_tokens=min(384, int(body.get("max_tokens", 384))))
                self.send(200, result)
            except Exception as e:
                self.send(422, {"error": str(e)[:240], "error_type": type(e).__name__})

    print(json.dumps({"status": "MODEL_SERVER_READY", "adapter": model.adapter, "revision": MODEL_REVISION}), flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__": main()
