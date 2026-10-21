"""A bounded cloud subprocess with a durable success/failure receipt."""
import argparse
import subprocess
import sys

from .settings import ARTIFACTS
from .storage import now, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    path = ARTIFACTS / "reports/jobs" / (args.id + ".json")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    receipt = {"id": args.id, "created": now(), "status": "running", "command": command}
    write_json(path, receipt, exclusive=True)
    try:
        result = subprocess.run([sys.executable, "-u", "-m", *command], timeout=args.timeout)
        receipt.update(status="completed" if result.returncode == 0 else "failed", exit_code=result.returncode)
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:250])
    receipt["finished"] = now()
    write_json(path, receipt)
    if receipt["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__": main()
