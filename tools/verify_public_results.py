"""Re-score saved executions and validate the matched data pool. No model calls."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contract_agent.data import DatasetStore
from contract_agent.scoring import score_run, summarize
from contract_agent.storage import read_json, read_jsonl


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify():
    artifacts = ROOT / "contract_agent/artifacts"
    result = read_json(artifacts / "reports/final_results.json")
    gold = read_json(artifacts / "data/annotations.json")
    expected = set(DatasetStore(artifacts).select("test"))
    assert len(expected) == 340
    total = 0
    for condition, source in result["sources"].items():
        mode = "full_context" if condition == "full_context" else "agent"
        rows = [r for r in read_json(resolve(source))["runs"] if r["mode"] == mode]
        assert len(rows) == len(expected) and {r["task_id"] for r in rows} == expected, condition
        for row in rows:
            actual = score_run(row, gold[row["task_id"]])
            assert actual == row["score"], (condition, row["task_id"], "per-case score differs")
            row["score"] = actual
        actual = summarize(rows)
        stored = result["conditions"][condition]
        for key in ("n", "macro_f1", "accuracy", "evidence", "confusion_matrix", "status_counts"):
            assert actual[key] == stored[key], (condition, key)
        total += len(rows)
        print(f"{condition}: n={len(rows)}, Macro-F1={actual['macro_f1']:.6f}, evidence-F1={actual['evidence']['f1']:.6f}")
    assert total == 1700
    prefs = result["preference_diagnostics"]
    raw = read_jsonl(resolve(prefs["raw_file"]))
    verified = read_jsonl(resolve(prefs["verified_file"]))
    assert len(raw) == len(verified) == 88
    right = {r["id"]: r for r in verified}
    assert {r["id"] for r in raw} == set(right)
    flips = 0
    for row in raw:
        other = right[row["id"]]
        assert row["prompt"] == other["prompt"]
        canonical = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
        assert {canonical(row[k]) for k in ("chosen", "rejected")} == {canonical(other[k]) for k in ("chosen", "rejected")}
        flips += row["chosen"] != other["chosen"]
    assert flips == 17
    print(f"Verified {total} test executions; matched preference pool={len(raw)}, label differences={flips}. No inference performed.")


if __name__ == "__main__":
    verify()
