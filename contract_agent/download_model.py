"""Pin and verify every model weight against original Hugging Face LFS hashes."""
import hashlib
import json
from pathlib import Path
import urllib.request
import argparse

from .download import fetch_ranges
from .settings import MODEL_ID, MODEL_REVISION, ARTIFACTS
from .storage import write_json, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32: raise ValueError("Download workers must be between 1 and 32")
    root = ARTIFACTS / "cache/qwen3-4b"
    root.mkdir(parents=True, exist_ok=True)
    # Manifest was fetched from the original pinned HF API locally and deployed verbatim.
    files = json.loads((Path(__file__).parent / "model_manifest.json").read_text())
    receipt = {"repo": MODEL_ID, "revision": MODEL_REVISION, "created": now(), "files": []}
    for item in files:
        name = item["path"]
        if item.get("type") != "file" or not name.endswith((".json", ".safetensors", ".jinja", ".txt")):
            continue
        # Public mirror transport is accepted only if content matches original pinned HF hashes.
        source = f"https://modelscope.cn/models/{MODEL_ID}/resolve/master/{name}"
        destination = root / name
        expected = (item.get("lfs") or {}).get("oid")
        if item["size"] > 512 * 1024:
            actual = fetch_ranges(source, destination, expected, workers=args.workers, chunk_size=1024 * 1024)
        else:
            if not destination.exists():
                destination.write_bytes(urllib.request.urlopen(source, timeout=120).read())
            actual = hashlib.sha256(destination.read_bytes()).hexdigest()
            if expected and actual != expected:
                raise ValueError("Model metadata/tokenizer hash mismatch: " + name)
        if not expected and item.get("oid"):
            data = destination.read_bytes()
            if hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() != item["oid"]:
                raise ValueError("Model metadata Git blob hash mismatch: " + name)
        receipt["files"].append({"file": name, "size": destination.stat().st_size, "sha256": actual, "expected_lfs_sha256": expected})
        write_json(root / "download_receipt.json", receipt)
    (root / "DOWNLOAD_COMPLETE").write_text(MODEL_REVISION)
    print("MODEL_DOWNLOAD_COMPLETED", flush=True)


if __name__ == "__main__": main()
