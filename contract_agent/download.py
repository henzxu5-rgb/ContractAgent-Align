"""Parallel, resumable HTTP ranges with per-range validation and final checksum."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
import time


def fetch_ranges(url, destination, sha256=None, workers=8, chunk_size=8 * 1024 * 1024):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        checksum = hashlib.file_digest(destination.open("rb"), "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(destination.read_bytes()).hexdigest()
        if sha256 and checksum != sha256:
            raise ValueError("Existing download checksum mismatch")
        return checksum
    req = urllib.request.Request(url + ("&" if "?" in url else "?") + "contract_agent_range=0-0", headers={"Range": "bytes=0-0", "User-Agent": "ContractAgent-Align/0.1"})
    with urllib.request.urlopen(req, timeout=60) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith("bytes 0-0/"):
            raise ValueError("Server does not provide validated byte ranges")
        size = int(content_range.split("/")[-1])
    chunks = destination.with_suffix(destination.suffix + f".parts-{chunk_size}")
    chunks.mkdir(exist_ok=True)
    def fetch(index):
        start = index * chunk_size; end = min(size, start + chunk_size) - 1
        target = chunks / str(index)
        if target.exists() and target.stat().st_size == end - start + 1:
            return
        for attempt in range(3):
            req = urllib.request.Request(url + ("&" if "?" in url else "?") + f"contract_agent_range={start}-{end}", headers={"Range": f"bytes={start}-{end}", "User-Agent": "ContractAgent-Align/0.1"})
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("Mismatched content range")
                    data = response.read()
                break
            except Exception as exc:
                if attempt == 2: raise
                print(json.dumps({"download": destination.name, "chunk": index, "network_retry": attempt + 1, "error": type(exc).__name__}), flush=True)
                time.sleep(2 ** attempt)
        if len(data) != end - start + 1:
            raise ValueError("Incomplete range")
        target.write_bytes(data)
        print(json.dumps({"download": destination.name, "chunk": index, "bytes": len(data)}), flush=True)
    indices = list(range((size + chunk_size - 1) // chunk_size))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, indices))
    checksum = hashlib.sha256()
    with destination.with_suffix(destination.suffix + ".partial").open("wb") as output:
        for i in indices:
            data = (chunks / str(i)).read_bytes(); checksum.update(data); output.write(data)
    actual = checksum.hexdigest()
    if sha256 and actual != sha256:
        raise ValueError("Final download checksum mismatch")
    destination.with_suffix(destination.suffix + ".partial").replace(destination)
    for i in indices: (chunks / str(i)).unlink()
    chunks.rmdir()
    return actual
