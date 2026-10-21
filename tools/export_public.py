"""Export a reviewed allowlist to a standalone public repository, never parent history."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contract_agent.settings import load_env
from contract_agent.storage import read_json, now


def export(destination):
    destination = Path(destination).resolve()
    if destination == ROOT or ROOT.is_relative_to(destination):
        raise ValueError("Use a separate publication directory, not the source or its ancestor")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError("Publication directory must be empty")
    artifacts = ROOT / "contract_agent/artifacts"
    result = read_json(artifacts / "reports/final_results.json")
    assert result["status"] == "complete_test_batches"
    paths = {ROOT / p for p in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", ".env.example",
                              ".github/workflows/tests.yml", "docs/METHODOLOGY.md", "docs/REPRODUCE.md")}
    for pattern in ("*.py", "*.md", "*.txt", "model_manifest.json", "tests/*.py", "web/*.html"):
        paths.update((ROOT / "contract_agent").glob(pattern))
    paths.update((ROOT / "tools").glob("*.py"))
    paths.update((ROOT / "assets").glob("*.svg"))
    for name in ("documents.json", "annotations.json", "tasks.json", "manifest.json"):
        paths.add(artifacts / "data" / name)
    for name in ("align-v1-sft.jsonl", "align-v1-sft.summary.json", "align-v1-rtx6000d-v2-raw.jsonl", "align-v1-rtx6000d-v2-verified.jsonl"):
        paths.add(artifacts / "datasets" / name)
    report_names = ("FINAL_REPORT.md", "final_results.json", "FAILURE_ANALYSIS.md", "failure_analysis.json",
                    "per_case_analysis.jsonl", "PREFERENCE_AUDIT.md", "preference_audit.json", "data_audit.json",
                    "stage_gate.json", "dpo-arms-integrity.json", "final-preference-freeze.json", "model_checksums.json")
    paths.update(artifacts / "reports" / n for n in report_names)
    def locate(value):
        path = Path(value)
        return path if path.is_absolute() else ROOT / path
    batches = {locate(p) for p in result["sources"].values()}
    batches.update(artifacts / "batches" / (name + ".json") for name in ("align-v1-teacher", "align-v1-sft-rollouts"))
    for batch in batches:
        paths.add(batch)
        paths.update(artifacts / "runs" / (run["id"] + ".json") for run in read_json(batch)["runs"])
    pref = locate(result["preference_diagnostics"]["raw_file"]).parents[2]
    paths.add(pref / "summary.json")
    paths.update((pref / "items").glob("*.json"))
    paths.update((pref / "exports/pairs-0400").glob("*.jsonl"))
    for name in ("align-v1-sft", "align-v1-rtx6000d-v2-dpo-raw", "align-v1-rtx6000d-v2-dpo-verified"):
        paths.add(artifacts / "training_logs" / name / "receipt.json")

    def clean(value):
        if isinstance(value, list): return [clean(v) for v in value]
        if isinstance(value, dict):
            # Infrastructure login metadata is not needed to audit model behavior.
            return {k: clean(v) for k, v in value.items() if k not in ("evaluation_recovery",)}
        if not isinstance(value, str): return value
        value = value.replace(str(ROOT), ".").replace(ROOT.as_posix(), ".")
        value = value.replace("/root/autodl-tmp/contract-agent-align/", "")
        if value.startswith(".\\contract_agent\\"):
            value = value[2:].replace("\\", "/")
        if value.startswith("./contract_agent/"): value = value[2:]
        value = re.sub(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/]+", "<user-home>", value)
        value = re.sub(r"(?:connect\.[a-z]+\.seetacloud\.com|region-\d+\.autodl\.pro)", "<cloud-host>", value)
        return value

    load_env()
    secret_values = {v.encode() for k, v in os.environ.items() if ("PASS" in k or k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY")) and len(v) >= 6}
    records = []
    for source in sorted(paths):
        relative = source.relative_to(ROOT).as_posix()
        original = source.read_bytes()
        if source.suffix == ".json":
            published = (json.dumps(clean(json.loads(original.decode("utf-8-sig"))), ensure_ascii=False, indent=2) + "\n").encode()
        elif source.suffix == ".jsonl":
            published = ("".join(json.dumps(clean(json.loads(line)), ensure_ascii=False) + "\n" for line in original.decode("utf-8-sig").splitlines() if line.strip())).encode()
        else:
            published = original
        if relative == "contract_agent/README.md":
            published = "# ContractAgent-Align Python 包\n\n[项目主页与快速体验](../README.md) · [方法](../docs/METHODOLOGY.md) · [复现](../docs/REPRODUCE.md) · [数据卡](DATA_CARD.md) · [SOP](SOP.md)\n".encode()
        if any(secret in published for secret in secret_values):
            raise ValueError("Credential match in publication allowlist: " + relative)
        if len(published) >= 50 * 1024 * 1024:
            raise ValueError("Oversized public file: " + relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(published)
        records.append({"path": relative, "bytes": len(published), "source_sha256": hashlib.sha256(original).hexdigest(),
                        "public_sha256": hashlib.sha256(published).hexdigest()})
    (destination / ".gitignore").write_text(".env\n.env.*\n!.env.example\n.venv/\n__pycache__/\n*.py[cod]\ncontract_agent/artifacts/models/\ncontract_agent/artifacts/cache/\ncontract_agent/artifacts/raw/\ncontract_agent/artifacts/api_calls/\ncontract_agent/artifacts/budget.sqlite*\n*.download\n", encoding="utf-8")
    manifest = {"created": now(), "scope": "Current ContractAgent-Align only; data content retained, local paths normalized",
                "files": records, "current_credential_matches": 0,
                "excluded": ["credentials", "account balance responses", "personal work logs", "legacy projects", "base and adapter weights", "optimizer states"]}
    (destination / "PUBLICATION_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"destination": str(destination), "files": len(records), "bytes": sum(r["bytes"] for r in records), "credential_matches": 0}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    export(parser.parse_args().destination)
