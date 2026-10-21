from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DATA_URL = "https://raw.githubusercontent.com/stanfordnlp/contract-nli/gh-pages/resources/contract-nli.zip"
DATA_SHA256 = "e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b"
SEED = 20261015
LABELS = ("Entailment", "Contradiction", "NotMentioned")
MAX_TOOL_CALLS = 6
# Conservative peak-time uncached CNY / million tokens; record the rate per call.
INPUT_PRICE = 2.0
OUTPUT_PRICE = 8.0
API_BUDGET = 40.0
GPU_BUDGET = 100.0
TOTAL_BUDGET = 200.0
PILOT_BUDGET = 20.0


def load_env():
    """Read local ignored credentials without exposing them in logs."""
    import os
    for path in (ROOT.parent / ".env", ROOT / ".env"):
        if path.exists():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.lstrip().startswith("#"):
                    continue
                key, sep, value = line.partition("=")
                if sep:
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def gpu_hourly_cny():
    """User-confirmed instance rate; defaults only to the original V100 upper bound."""
    import os
    import math
    load_env()
    value = float(os.environ.get("AUTODL_HOURLY_CNY", "3"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("AUTODL_HOURLY_CNY must be a positive finite hourly price")
    return value
