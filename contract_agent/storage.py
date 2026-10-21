import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def identifier(prefix):
    return f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:10]}"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
        return
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
