"""Bounded execution of the approved experiment, with durable stage receipts.

No model search or automatic score-driven retries. A failed stage stops the run;
completed stages can be resumed. The cloud deadline limits charges if this local
controller disappears. API credentials remain local.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .budget import Ledger
from .remote import connect, upload, launch, REMOTE_DIR, download
from .settings import ARTIFACTS, gpu_hourly_cny
from .storage import now, read_json, write_json

REMOTE_ARTIFACTS = REMOTE_DIR + "/contract_agent/artifacts"
MODEL_PATH = REMOTE_ARTIFACTS + "/cache/qwen3-4b"


def remote_json(relative):
    client = connect()
    try:
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(60)
        try:
            with sftp.open(REMOTE_ARTIFACTS + "/" + relative) as stream:
                stream.prefetch(max_concurrent_requests=32)
                return json.load(stream)
        except FileNotFoundError:
            return None
        finally:
            sftp.close()
    finally:
        client.close()


def push_json(value, relative):
    from .remote import mkdirs
    import posixpath
    client = connect()
    try:
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(60)
        path = REMOTE_ARTIFACTS + "/" + relative
        mkdirs(sftp, posixpath.dirname(path))
        with sftp.open(path + ".new", "w") as stream:
            stream.write(json.dumps(value))
        sftp.posix_rename(path + ".new", path)
        sftp.close()
    finally:
        client.close()


class Controller:
    def __init__(self, uid):
        if not __import__("re").fullmatch(r"[A-Za-z0-9_-]+", uid):
            raise ValueError("Invalid pipeline ID")
        self.uid = uid
        self.folder = ARTIFACTS / "pipelines" / uid
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "state.json"
        self.state = read_json(self.path) if self.path.exists() else {"id": uid, "created": now(), "steps": {}}

    def save(self):
        self.state["updated"] = now()
        write_json(self.path, self.state)

    def stop_model_service(self, adapter, preserve_error=False):
        """Stop only the HTTP inference service, never the billing instance."""
        from .models import RemoteHFModel
        try:
            policy = RemoteHFModel(adapter)
            try: policy.shutdown()
            finally: policy.close()
        except Exception as exc:
            self.state.setdefault("cleanup_errors", []).append({"time": now(), "error_type": type(exc).__name__, "error": str(exc)[:300]})
            self.save()
            if not preserve_error: raise

    def resume_after_shutdown(self):
        """Explicitly recover the diagnosed preference interruption, not an unknown failed job."""
        if self.state.get("current_step") != "preferences" or self.state["steps"].get("preferences", {}).get("status") != "failed":
            raise ValueError("Shutdown recovery only applies to the diagnosed failed preference stage")
        for name in ("sft", "sft-rollouts"):
            if self.state["steps"].get(name, {}).get("status") != "completed":
                raise ValueError("Completed SFT and rollout receipts required for shutdown recovery")
            receipt = remote_json("reports/jobs/" + self.uid + "-" + name + ".json")
            if not receipt or receipt.get("status") != "completed":
                raise ValueError("Remote completed job receipt is missing: " + name)
        recovery = "recovery-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        write_json(self.folder / "recoveries" / (recovery + "-previous-state.json"), self.state, exclusive=True)
        start = datetime.now(timezone.utc)
        # Half an hour also covers the bounded reconnect/setup period before this
        # receipt. Old uncertain GPU reservations remain conservative upper bounds.
        rate = gpu_hourly_cny()
        rid = Ledger().reserve(2.5 * rate, "gpu", "main", {"pipeline": self.uid, "recovery": recovery,
                    "start": (start - timedelta(minutes=30)).isoformat(), "hours": 2.5, "hourly_cny_upper": rate})
        deadline = {"deadline_utc": (start + timedelta(hours=2)).isoformat(), "reservation": rid,
                    "hourly_cny_upper": rate,
                    "reason": "User resumed after balance shutdown; new bounded cloud session"}
        push_json(deadline, "cloud_deadline.json")
        write_json(ARTIFACTS / "cloud_deadline.json", deadline)
        guard = launch("contract_agent.watchdog", [], self.uid + "-" + recovery + "-guard.log", False, 172800)
        if guard["exit_code"]: raise RuntimeError("Recovery guard launch failed")
        deployed = upload()
        for name in ("preferences", "start-sft-server"):
            old = self.state["steps"].get(name, {})
            self.state["steps"][name] = {"status": "pending", "previous_attempts": [*old.get("previous_attempts", []), {k: v for k, v in old.items() if k != "previous_attempts"}]}
        self.state.setdefault("recoveries", []).append({"id": recovery, "time": now(), "reason": "User confirmed balance shutdown and explicitly requested continuation", "lease": deadline, "deployment": deployed})
        self.state.update(status="ready_to_resume")
        for key in ("error", "error_type", "network_status"): self.state.pop(key, None)
        self.save()
        print(json.dumps({"shutdown_recovery": recovery, "deadline": deadline["deadline_utc"]}), flush=True)

    def network_read(self, callback, max_wait=1800):
        """Wait for transient transport failures; never repeat a paid launch/API call."""
        import paramiko
        started = time.monotonic()
        waiting = False
        while True:
            try:
                value = callback()
                if waiting:
                    self.state.update(network_status="connected", network_recovered=now())
                    self.save()
                    print(json.dumps({"network_recovered": now()}), flush=True)
                return value
            except (paramiko.AuthenticationException, paramiko.BadHostKeyException):
                raise
            except (paramiko.SSHException, OSError, EOFError):
                deadline = datetime.fromisoformat(read_json(ARTIFACTS / "cloud_deadline.json")["deadline_utc"].replace("Z", "+00:00"))
                if time.monotonic() - started >= max_wait or datetime.now(timezone.utc) >= deadline:
                    raise TimeoutError("Network wait limit or reserved cloud deadline reached; cloud work is not relaunched") from None
                if not waiting:
                    self.state.update(network_status="waiting", network_wait_started=now())
                    self.save()
                    print(json.dumps({"waiting_for_network": now(), "max_wait_seconds": max_wait}), flush=True)
                    waiting = True
                time.sleep(20)

    def lease(self):
        path = ARTIFACTS / "cloud_deadline.json"
        data = read_json(path)
        deadline = datetime.fromisoformat(data["deadline_utc"].replace("Z", "+00:00"))
        current = datetime.now(timezone.utc)
        if current >= deadline:
            raise RuntimeError("Cloud lease expired; do not assume the instance is still running")
        if (deadline - current).total_seconds() >= 1800:
            return
        # Reserve before extending. Existing reservations remain charged at their
        # upper bound; this cannot undercount an unknown invoice.
        rate = gpu_hourly_cny()
        amount = 2 * rate
        rid = Ledger().reserve(amount, "gpu", "main", {"pipeline": self.uid, "start": deadline.isoformat(), "hours": 2, "hourly_cny_upper": rate})
        data.update(deadline_utc=(deadline + timedelta(hours=2)).isoformat(), reservation=rid,
                    hourly_cny_upper=rate,
                    reason="Bounded formal experiment lease; shutdown if the controller stops extending it")
        self.network_read(lambda: push_json(data, "cloud_deadline.json"))
        write_json(path, data)
        print(json.dumps({"cloud_lease_extended": data["deadline_utc"], "reserved_cny": amount}), flush=True)

    def local(self, name, command, timeout):
        def execute():
            log = self.folder / (name + ".log")
            started = time.monotonic()
            with log.open("a", encoding="utf-8") as stream:
                process = subprocess.Popen([sys.executable, "-u", "-m", *command], stdout=stream, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                try:
                    while process.poll() is None:
                        self.lease()
                        if time.monotonic() - started > timeout:
                            raise TimeoutError(name)
                        time.sleep(20)
                    if process.returncode:
                        raise RuntimeError(name + " failed; inspect " + str(log))
                finally:
                    if process.poll() is None:
                        process.terminate(); process.wait(timeout=15)
            return {"log": str(log)}
        return self.step(name, execute)

    def step(self, name, callback):
        if self.state["steps"].get(name, {}).get("status") == "completed":
            return self.state["steps"][name].get("result")
        if self.state["steps"].get(name, {}).get("status") == "failed":
            raise RuntimeError("A failed stage needs diagnosis before an explicit new run; no automatic paid retry: " + name)
        self.state.update(status="running", current_step=name)
        if self.state["steps"].get(name, {}).get("status") == "running":
            self.state["steps"][name].setdefault("resumed", []).append(now())
        else:
            previous = self.state["steps"].get(name, {}).get("previous_attempts", [])
            self.state["steps"][name] = {"status": "running", "started": now(), "previous_attempts": previous}
        self.save()
        print(json.dumps({"step_started": name}), flush=True)
        try:
            result = callback()
        except Exception as exc:
            self.state.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:300])
            self.state["steps"][name].update(status="failed", finished=now())
            self.save(); raise
        self.state["steps"][name].update(status="completed", finished=now(), result=result)
        self.save()
        return result

    def cloud(self, name, command, timeout):
        def execute():
            self.lease()
            jid = self.uid + "-" + name
            relative = "reports/jobs/" + jid + ".json"
            receipt = self.network_read(lambda: remote_json(relative))
            if receipt is None:
                launched = launch("contract_agent.cloud_job", ["--id", jid, "--timeout", str(timeout), "--", *command], jid + ".log", True, timeout + 120)
                if launched["exit_code"]:
                    raise RuntimeError("Cloud launch failed: " + str(launched))
            started = time.monotonic()
            while True:
                self.lease()
                receipt = self.network_read(lambda: remote_json(relative))
                if receipt and receipt["status"] in ("completed", "failed"):
                    write_json(self.folder / (name + "-receipt.json"), receipt)
                    if receipt["status"] != "completed":
                        raise RuntimeError("Cloud stage failed: " + jid + "; logs retained on instance")
                    return receipt
                if time.monotonic() - started > timeout + 180:
                    raise TimeoutError(jid)
                time.sleep(30)
        return self.step(name, execute)

    def fetch_batch(self, name):
        batch = self.network_read(lambda: remote_json("batches/" + name + ".json"))
        if batch is None:
            raise FileNotFoundError(name)
        path = ARTIFACTS / "batches" / (name + ".json")
        write_json(path, batch)
        # Each episode is also available independently for UI replay.
        for row in batch["runs"]:
            record = {k: v for k, v in row.items() if k != "score"}
            destination = ARTIFACTS / "runs" / (row["id"] + ".json")
            if not destination.exists(): write_json(destination, record, exclusive=True)
        return str(path)

    def run(self):
        gate = read_json(ARTIFACTS / "reports/stage_gate.json")
        if not gate.get("approved"):
            raise RuntimeError("Formal pipeline cannot start before the pilot gate passes")
        self.lease()
        self.step("deploy", lambda: upload())
        self.step("cloud-deadline-guard", lambda: launch("contract_agent.watchdog", [], self.uid + "-deadline-guard.log", False, 172800))
        self.network_read(lambda: push_json(gate, "reports/stage_gate.json"))
        teacher = self.uid + "-teacher"
        self.local("teacher", ["contract_agent.cli", "run", "--backend", "deepseek", "--selection", "train", "--limit", "300", "--mode", "agent", "--phase", "main", "--target-successes", "120", "--batch-id", teacher], 7200)
        sft_data = ARTIFACTS / "datasets" / (self.uid + "-sft.jsonl")
        self.local("export-sft", ["contract_agent.cli", "export-sft", "--batch", str(ARTIFACTS / "batches" / (teacher + ".json")), "--output", str(sft_data)], 120)
        summary = read_json(sft_data.with_suffix(".summary.json"))
        if summary["accepted_trajectories"] < 80:
            raise RuntimeError("Fewer than 80 accepted teacher trajectories; stop rather than expand beyond the 300-task cap")
        self.step("deploy-sft-data", lambda: upload())
        sft = self.uid + "-sft"
        adapter = REMOTE_ARTIFACTS + "/models/" + sft
        self.cloud("sft", ["contract_agent.training", "sft", "--data", REMOTE_ARTIFACTS + "/datasets/" + sft_data.name, "--model-path", MODEL_PATH, "--run-id", sft], 10800)
        rollout = self.uid + "-sft-rollouts"
        self.cloud("sft-rollouts", ["contract_agent.cli", "run", "--backend", "hf", "--selection", "train_initial", "--limit", "200", "--mode", "agent", "--phase", "main", "--model-path", MODEL_PATH, "--adapter", adapter, "--batch-id", rollout], 10800)
        rollout_file = self.fetch_batch(rollout)
        pair_id = self.uid + "-preferences"
        if self.state["steps"].get("preferences", {}).get("status") != "completed":
            self.lease()
            self.step("start-sft-server", lambda: launch("contract_agent.model_server", ["--model-path", MODEL_PATH, "--adapter", adapter], self.uid + "-sft-server.log", True, 21600))
            from .models import RemoteHFModel
            policy = None
            for _ in range(12):
                try:
                    policy = RemoteHFModel(adapter); break
                except Exception:
                    time.sleep(10)
            if policy is None: raise RuntimeError("SFT inference server did not become ready")
            policy.close()
            try:
                command = ["contract_agent.cli", "pairs", "--batch", rollout_file, "--adapter", adapter, "--limit", "400", "--target", "160", "--run-id", pair_id]
                if (ARTIFACTS / "preferences" / pair_id).exists(): command.append("--resume")
                self.local("preferences", command, 18000)
            finally:
                self.stop_model_service(adapter, preserve_error=sys.exc_info()[0] is not None)
        pref = read_json(ARTIFACTS / "preferences" / pair_id / "summary.json")
        arms = []
        if pref["minimum_data_pass"]:
            for kind in (["raw", "verified"] if pref["train_raw_control"] else ["verified"]):
                source = Path(pref[kind + "_file"])
                destination = ARTIFACTS / "datasets" / (self.uid + "-" + kind + ".jsonl")
                if not destination.exists(): shutil.copyfile(source, destination)
                arms.append(kind)
            self.step("deploy-dpo-data", lambda: upload())
            for kind in arms:
                self.cloud("dpo-" + kind, ["contract_agent.training", "dpo", "--data", REMOTE_ARTIFACTS + "/datasets/" + self.uid + "-" + kind + ".jsonl", "--model-path", MODEL_PATH, "--sft-adapter", adapter, "--run-id", self.uid + "-dpo-" + kind], 10800)
        self.state["preference_summary"] = pref
        self.state["training_arms"] = ["sft", *["dpo-" + arm for arm in arms]]
        self.save()
        conditions = {}
        for condition in ["base", "sft", *["dpo-" + arm for arm in arms]]:
            batch_id = self.uid + "-test-" + condition
            command = ["contract_agent.cli", "run", "--backend", "hf", "--selection", "test", "--mode", "both" if condition == "base" else "agent", "--phase", "main", "--model-path", MODEL_PATH, "--batch-id", batch_id, "--inference-batch-size", "8"]
            if condition != "base": command += ["--adapter", REMOTE_ARTIFACTS + "/models/" + self.uid + "-" + condition]
            self.cloud("test-" + condition, command, 18000)
            conditions[condition.replace("-", "_")] = self.fetch_batch(batch_id)
        conditions["full_context"] = conditions["base"]
        from .final_report import build
        self.step("independent-report", lambda: {"status": build(conditions)["status"]})
        self.step("download-artifacts", download)
        self.state.update(status="completed" if arms else "completed_with_insufficient_preference_data", finished=now())
        self.save()
        # All artifacts are local now; shorten the existing guard to avoid idle billing.
        deadline = {"deadline_utc": (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat(),
                    "reason": "Experiment finished and artifacts downloaded; stop idle billing"}
        push_json(deadline, "cloud_deadline.json")
        write_json(ARTIFACTS / "cloud_deadline.json", deadline)
        print(json.dumps({"pipeline_status": self.state["status"], "state": str(self.path)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume-after-shutdown", action="store_true", help="Explicitly recover the diagnosed preference interruption after reboot")
    args = parser.parse_args()
    controller = Controller(args.run_id)
    try:
        if args.resume_after_shutdown: controller.resume_after_shutdown()
        controller.run()
    except Exception as exc:
        controller.state.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:300])
        controller.save()
        raise
