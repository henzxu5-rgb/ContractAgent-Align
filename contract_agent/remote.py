"""Explicit project-only SSH deployment. Never uploads parent folders or secrets."""
import argparse
import json
import os
import posixpath
import shlex
import time
from pathlib import Path

from .settings import ROOT, ARTIFACTS, load_env

REMOTE_DIR = "/root/autodl-tmp/contract-agent-align"


def connect():
    import paramiko
    load_env()
    required = ("AUTODL_HOST", "AUTODL_PORT", "AUTODL_PASS")
    if any(not os.environ.get(k) for k in required):
        raise ValueError("Current AUTODL_HOST/AUTODL_PORT/AUTODL_PASS required in ignored .env")
    for attempt in range(3):
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.WarningPolicy())
        try:
            client.connect(os.environ["AUTODL_HOST"], port=int(os.environ["AUTODL_PORT"]),
                           username=os.environ.get("AUTODL_USER", "root"), password=os.environ["AUTODL_PASS"],
                           timeout=20, auth_timeout=20, banner_timeout=20)
            client.get_transport().set_keepalive(30)
            return client
        except (paramiko.AuthenticationException, paramiko.BadHostKeyException):
            client.close(); raise
        except (paramiko.SSHException, OSError, EOFError):
            client.close()
            if attempt == 2: raise
            # No command has been submitted yet: this retries transport setup,
            # never a paid inference request, training job or launch command.
            time.sleep(2 ** attempt)


def run(command, timeout=60):
    client = connect()
    try:
        _, stdout, stderr = client.exec_command(command, timeout=timeout)
        output, error = stdout.read().decode(), stderr.read().decode()
        status = stdout.channel.recv_exit_status()
        return {"exit_code": status, "stdout": output, "stderr": error}
    finally:
        client.close()


def mkdirs(sftp, remote):
    parts = remote.strip("/").split("/")
    current = ""
    for part in parts:
        current += "/" + part
        try:
            sftp.stat(current)
        except IOError:
            sftp.mkdir(current)


def pull_file(sftp, remote, local):
    """Bounded prefetch and atomic promotion keep interrupted copies out of artifacts."""
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    pending = local.with_name(local.name + ".download")
    sftp.get(remote, str(pending), max_concurrent_prefetch_requests=32)
    expected = sftp.stat(remote).st_size
    if pending.stat().st_size != expected:
        raise IOError("Remote file changed or download incomplete: " + local.name)
    pending.replace(local)


def pull_verified(sftp, remote, local, expected_sha256):
    """Resume an immutable file; promote only after the source hash matches."""
    import hashlib
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and hashlib.sha256(local.read_bytes()).hexdigest() == expected_sha256:
        return
    pending = local.with_name(local.name + ".download")
    size = sftp.stat(remote).st_size
    offset = pending.stat().st_size if pending.exists() else 0
    if offset > size:
        raise ValueError("Saved partial file is larger than immutable source")
    with sftp.open(remote, "rb") as source, pending.open("ab") as target:
        source.seek(offset)
        source.prefetch(size, max_concurrent_requests=8)
        while offset < size:
            block = source.read(min(512 * 1024, size - offset))
            if not block: raise IOError("Immutable transfer ended before expected size")
            target.write(block)
            offset += len(block)
    h = hashlib.sha256()
    with pending.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != expected_sha256:
        raise ValueError("Immutable transfer hash mismatch; partial copy preserved for diagnosis")
    pending.replace(local)


def upload(include_data=True):
    client = connect()
    count = 0
    try:
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(60)
        files = list(ROOT.glob("*.py")) + list(ROOT.glob("*.txt")) + [ROOT / "model_manifest.json"]
        if include_data:
            files += list((ARTIFACTS / "data").glob("*.json"))
            files += list((ARTIFACTS / "datasets").glob("*.jsonl"))
        for path in files:
            remote = REMOTE_DIR + "/contract_agent/" + path.relative_to(ROOT).as_posix()
            mkdirs(sftp, posixpath.dirname(remote))
            if path.parent == ARTIFACTS / "data":
                try:
                    if sftp.stat(remote).st_size == path.stat().st_size:
                        continue  # Prepared data are immutable and manifest-hashed.
                except IOError:
                    pass
            sftp.put(str(path), remote)
            count += 1
        sftp.close()
    finally:
        client.close()
    return {"uploaded_files": count, "remote_directory": REMOTE_DIR, "credentials_uploaded": False}


def download():
    client = connect()
    count = 0
    try:
        import stat
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(60)
        def descend(remote, local):
            nonlocal count
            if local.parent == ARTIFACTS / "models":
                try:
                    with sftp.open(REMOTE_DIR + "/contract_agent/artifacts/training_logs/" + local.name + "/receipt.json") as stream:
                        receipt = json.load(stream)
                    if receipt.get("status") != "completed":
                        return  # Never fetch a checkpoint while it is being written.
                except IOError:
                    return
            try:
                entries = sftp.listdir_attr(remote)
            except IOError:
                return
            for entry in entries:
                rpath, lpath = remote + "/" + entry.filename, local / entry.filename
                if stat.S_ISDIR(entry.st_mode):
                    descend(rpath, lpath)
                else:
                    # Only completed run records are immutable. Progress reports and
                    # receipts can change without changing their byte length.
                    if local.is_relative_to(ARTIFACTS / "runs") and lpath.exists() and lpath.stat().st_size == entry.st_size:
                        continue
                    pull_file(sftp, rpath, lpath); count += 1
        for group in ("runs", "reports", "datasets", "training_logs", "models", "batches"):
            descend(REMOTE_DIR + "/contract_agent/artifacts/" + group, ARTIFACTS / group)
        sftp.close()
    finally:
        client.close()
    return {"downloaded_files": count}


def launch(module, args, log_name, use_venv=True, timeout=3600):
    # Popen closes the SSH descriptors; no shell background group can keep the channel open.
    python = REMOTE_DIR + "/.venv/bin/python" if use_venv else "/root/miniconda3/bin/python"
    argv = ["timeout", "--signal=TERM", str(timeout), python, "-u", "-m", module, *args]
    code = "import subprocess; f=open(" + repr(REMOTE_DIR + "/" + log_name) + ", 'x'); p=subprocess.Popen(" + repr(argv) + ",cwd=" + repr(REMOTE_DIR) + ",stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(p.pid)"
    return run("/root/miniconda3/bin/python -c " + shlex.quote(code))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("inspect", "upload", "download", "exec"))
    parser.add_argument("--command")
    args = parser.parse_args()
    if args.operation == "upload":
        result = upload()
    elif args.operation == "download":
        result = download()
    else:
        cmd = args.command or "ls -d /root/miniconda3/bin/python /root/miniconda3/envs/*/bin/python /usr/bin/python* 2>/dev/null; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader; date -u +%FT%TZ"
        result = run(cmd)
    print(json.dumps(result, ensure_ascii=False))
