"""Read-only loopback UI. Opening the page never triggers a paid model call."""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .settings import ROOT, ARTIFACTS
from .storage import read_json
from .budget import Ledger


def state():
    runs = []
    for path in sorted((ARTIFACTS / "batches").glob("*.json")):
        batch = read_json(path)
        for r in batch["runs"]:
            runs.append({"id": r["id"], "batch": batch["id"], "selection": batch["config"].get("selection"),
                         "task_id": r["task_id"], "doc_id": r["doc_id"], "mode": r["mode"],
                         "model": r.get("actual_model", r["model"]), "adapter": r.get("adapter"),
                         "status": r["status"], "score": r["score"], "seconds": r["seconds"],
                         "tool_calls": r["tool_calls"]})
    receipts = [read_json(p) for p in sorted((ARTIFACTS / "training_logs").glob("*/receipt.json"))]
    reports = [read_json(p) for p in sorted((ARTIFACTS / "reports").glob("*.json"))
               if p.name != "api-balance-after-judge-error.json"]
    audit = next((r for r in reports if "counts" in r and "sha256" in r), None)
    # A portable read-only delivery has the report snapshot, not the private ledger.
    final = ARTIFACTS / "reports/final_results.json"
    budget = (Ledger().summary() if (ARTIFACTS / "budget.sqlite").exists() or not final.exists()
              else read_json(final)["cost_upper"])
    preference_sets = [{"id": p.parent.name, "summary": read_json(p)}
                       for p in sorted((ARTIFACTS / "preferences").glob("*/summary.json"), key=lambda x: x.stat().st_mtime)]
    return {"runs": runs, "training": receipts, "reports": reports, "audit": audit,
            "preference_sets": preference_sets,
            "budget": {k: v for k, v in budget.items() if k != "entries"},
            "scope": "基于 ContractNLI 的工具交互适配；无新增法律专家审核；开发结果不等于独立测试结果。"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def respond(self, status, value, content_type="application/json; charset=utf-8"):
        body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.headers.get("Host") not in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"):
            return self.respond(403, {"error": "loopback only"})
        path = urlparse(self.path).path
        try:
            if path == "/":
                return self.respond(200, (ROOT / "web/index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/state":
                return self.respond(200, state())
            if path.startswith("/api/preferences/") or path.startswith("/api/preference/"):
                parts = path.strip("/").split("/")
                if len(parts) not in (3, 4) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in parts[2:]):
                    raise ValueError()
                folder = ARTIFACTS / "preferences" / parts[2]
                if parts[1] == "preferences" and len(parts) == 3:
                    items = sorted((folder / "items").glob("*.json")) or sorted(folder.glob(parts[2] + "-*.json"))
                    pairs = []
                    for item in items:
                        row = read_json(item)
                        if "raw_judge" in row:
                            pairs.append({"id": row["id"], "task_id": row["task_id"],
                                          "raw": row["raw_judge"]["winner"], "verified": row["verified_winner"]})
                    return self.respond(200, {"pairs": pairs})
                if parts[1] == "preference" and len(parts) == 4:
                    file = folder / "items" / (parts[3] + ".json")
                    if not file.exists(): file = folder / (parts[3] + ".json")
                    row = read_json(file)
                    from .data import DatasetStore
                    store = DatasetStore()
                    return self.respond(200, {"id": row["id"], "task_id": row["task_id"],
                        "hypothesis": store.public(row["task_id"])["hypothesis"],
                        "decision_index": row["decision_index"], "prompt": row["prompt"],
                        "reference": store.gold(row["task_id"]),
                        "candidates": [c["text"] for c in row["candidates"]],
                        "branches": [{k: b.get(k) for k in ("id", "submission", "status", "tool_calls", "tool_errors", "score", "steps")} for b in row["branches"]],
                        "raw": {k: row["raw_judge"].get(k) for k in ("winner", "reason")},
                        "swapped": {k: row["swapped_judge"].get(k) for k in ("winner", "reason")},
                        "verified": row["verified_winner"], "verified_basis": row["verified_basis"]})
                raise ValueError()
            if path.startswith("/api/run/"):
                uid = path.removeprefix("/api/run/")
                if not re.fullmatch(r"run-[A-Za-z0-9-]+", uid): raise ValueError()
                record = read_json(ARTIFACTS / "runs" / f"{uid}.json")
                from .data import DatasetStore
                store = DatasetStore()
                record["task"] = store.public(record["task_id"])
                record["reference"] = store.gold(record["task_id"])
                return self.respond(200, record)
            if path == "/favicon.ico": return self.respond(204, b"")
            return self.respond(404, {"error": "not found"})
        except (FileNotFoundError, ValueError, KeyError):
            return self.respond(404, {"error": "record not found"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--port", type=int, default=8766); args = p.parse_args()
    print(f"ContractAgent-Align: http://127.0.0.1:{args.port} (read-only; no API calls)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
