"""Build a local, explicit-allowlist portfolio archive after independent testing."""
import argparse
import hashlib
from pathlib import Path
import zipfile

from .settings import ARTIFACTS, ROOT
from .storage import now, read_json, write_json


def build(root=ARTIFACTS):
    root = Path(root)
    result = read_json(root / "reports/final_results.json")
    if result["status"] != "complete_test_batches":
        raise ValueError("Complete independent test results required before packaging")
    project = ROOT.parent
    files = set(ROOT.glob("*.py")) | set(ROOT.glob("*.md")) | set(ROOT.glob("*.txt"))
    files.add(ROOT / "model_manifest.json")
    files.update((ROOT / "web").glob("*.html"))
    files.update((ROOT / "tests").glob("*.py"))
    for name in ("PROJECT_CHARTER.md", "PROJECT_STATUS.md", "PROJECT_BRIEF.md", "WORK_LOG.md"):
        files.add(project / "docs" / name)
    for directory, pattern in (("data", "*.json"), ("datasets", "*.jsonl"), ("datasets", "*.summary.json"),
                               ("reports", "*.json"), ("reports", "*.jsonl"), ("reports", "*.md"), ("reports", "*.txt"),
                               ("deliverables", "*.md")):
        files.update((root / directory).glob(pattern))
    batch_paths = {Path(p) for p in result["sources"].values()}
    batch_paths.update(root / "batches" / (name + ".json") for name in ("align-v1-teacher", "align-v1-sft-rollouts"))
    model_names = set()
    for path in batch_paths:
        files.add(path)
        for run in read_json(path)["runs"]:
            if run.get("adapter"):
                model_names.add(Path(run["adapter"]).name)
            # UI replay reads the independent episode file, not the scored batch.
            files.add(root / "runs" / (run["id"] + ".json"))
    for name in model_names:
        if name.startswith("pilot-"):
            raise ValueError("Pilot adapter in final delivery")
        adapter = root / "models" / name
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError("Formal adapter is missing locally: " + name)
        files.update(p for p in adapter.iterdir() if p.is_file())
    files.update((root / "training_logs").glob("*/receipt.json"))
    preference = result.get("preference_diagnostics")
    if preference:
        # raw_file points to an immutable exports/pairs-NNNN snapshot.
        pref_folder = Path(preference["raw_file"]).parents[2]
        files.add(pref_folder / "summary.json")
        files.update((pref_folder / "items").glob("*.json"))
        files.update(Path(preference["raw_file"]).parent.glob("*"))
    # Fail closed if the allowlist references an absent result or a secret path.
    files = sorted(p for p in files if p.name not in ("api-balance-after-judge-error.json", "delivery_archive.json"))
    for path in files:
        path.relative_to(project)
        if not path.is_file(): raise FileNotFoundError(path)
        if any(part in (".env", "cache", ".venv", ".git") for part in path.parts):
            raise ValueError("Forbidden path in delivery allowlist")
    manifest = {"created": now(), "files": [{"path": p.relative_to(project).as_posix(),
                 "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
                "not_included": ["credentials", "API authentication headers", "base model weights", "pilot adapters", "optimizer checkpoints"],
                "license": "ContractNLI CC BY 4.0; cite original authors and disclose project adaptation"}
    destination = root / "deliverables" / ("ContractAgent-Align-" + now().replace(":", "").replace("+", "_") + ".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        import json
        archive.writestr("DELIVERY_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for path in files:
            archive.write(path, path.relative_to(project).as_posix())
    with zipfile.ZipFile(destination) as archive:
        bad = archive.testzip()
        if bad is not None: raise ValueError("Archive integrity failure: " + bad)
    receipt = {"created": now(), "archive": str(destination), "files": len(files),
               "bytes": destination.stat().st_size, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
               "zip_integrity_verified": True, "published": False}
    write_json(root / "reports/delivery_archive.json", receipt)
    return receipt


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(build())
