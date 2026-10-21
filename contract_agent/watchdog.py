"""Instance deadline guard: AutoDL's documented shutdown stops idle GPU billing."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    path = Path(__file__).resolve().parent / "artifacts/cloud_deadline.json"
    while True:
        config = json.loads(path.read_text())
        deadline = datetime.fromisoformat(config["deadline_utc"].replace("Z", "+00:00"))
        if datetime.now(timezone.utc) >= deadline:
            print("Cloud resource deadline reached; preserve project files and shut down billing instance", flush=True)
            # AutoDL supplies a shell wrapper without a shebang on this image.
            # Calling it through Popen directly would raise ENOEXEC; use the
            # documented shell command, with no interpolated user input.
            subprocess.run(["/bin/bash", "-c", "shutdown"], check=True)
            return
        time.sleep(20)


if __name__ == "__main__": main()
