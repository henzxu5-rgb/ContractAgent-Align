"""Reject local model folders without pinned download provenance."""
from pathlib import Path

from .settings import MODEL_ID, MODEL_REVISION
from .storage import read_json


def model_source(model_path=None):
    source = model_path or MODEL_ID
    path = Path(source)
    if not path.exists():
        return source, MODEL_REVISION
    marker = path / "DOWNLOAD_COMPLETE"
    receipt = read_json(path / "download_receipt.json")
    if not marker.is_file() or marker.read_text().strip() != MODEL_REVISION or receipt.get("revision") != MODEL_REVISION or receipt.get("repo") != MODEL_ID:
        raise ValueError("Local model lacks the pinned verified download receipt")
    manifest = read_json(Path(__file__).parent / "model_manifest.json")
    required = {f["path"] for f in manifest if f.get("type") == "file" and f["path"].endswith((".json", ".safetensors", ".jinja", ".txt"))}
    if {f["file"] for f in receipt["files"]} != required:
        raise ValueError("Local model download is incomplete")
    for item in receipt["files"]:
        file = path / item["file"]
        if not file.is_file() or file.stat().st_size != item["size"]:
            raise ValueError("Local verified model file missing or resized: " + item["file"])
    return str(path), None
