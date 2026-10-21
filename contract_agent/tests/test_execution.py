"""Execution controls, without network calls or a paid GPU."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

import contract_agent.pipeline as pipeline
import contract_agent.final_report as final_report
from contract_agent.remote import pull_file
from contract_agent.storage import write_json


class ExecutionTests(unittest.TestCase):
    def test_deadline_shutdown_uses_shell_for_provider_wrapper(self):
        import contract_agent.watchdog as watchdog
        with tempfile.TemporaryDirectory() as folder, patch.object(watchdog, "__file__", str(Path(folder) / "watchdog.py")), patch.object(watchdog.subprocess, "run") as shutdown:
            write_json(Path(folder) / "artifacts/cloud_deadline.json", {"deadline_utc": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()})
            watchdog.main()
            shutdown.assert_called_once_with(["/bin/bash", "-c", "shutdown"], check=True)

    def test_cloud_lease_uses_the_current_instance_price(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "ARTIFACTS", Path(folder)), patch.object(pipeline, "gpu_hourly_cny", return_value=6.8), patch.object(pipeline, "Ledger") as ledger, patch.object(pipeline, "push_json"):
            ledger.return_value.reserve.return_value = "reservation"
            write_json(Path(folder) / "cloud_deadline.json", {"deadline_utc": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()})
            pipeline.Controller("test").lease()
            self.assertAlmostEqual(ledger.return_value.reserve.call_args.args[0], 13.6)
            self.assertEqual(ledger.return_value.reserve.call_args.args[3]["hourly_cny_upper"], 6.8)

    def test_interrupted_download_does_not_replace_an_existing_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "adapter.bin"
            path.write_bytes(b"previous complete copy")
            class Interrupted:
                def get(self, remote, target, **kwargs):
                    self.limit = kwargs["max_concurrent_prefetch_requests"]
                    Path(target).write_bytes(b"partial")
                    raise ConnectionResetError("offline")
            transfer = Interrupted()
            with self.assertRaises(ConnectionResetError): pull_file(transfer, "remote.bin", path)
            self.assertEqual(path.read_bytes(), b"previous complete copy")
            self.assertEqual(transfer.limit, 32)

    def test_transient_network_failure_retries_only_the_read_and_recovers(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "ARTIFACTS", Path(folder)):
            write_json(Path(folder) / "cloud_deadline.json", {"deadline_utc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()})
            controller = pipeline.Controller("test")
            calls = []
            def read():
                calls.append(1)
                if len(calls) == 1: raise ConnectionResetError("temporary outage")
                return {"status": "running"}
            with patch.object(pipeline.time, "sleep"):
                self.assertEqual(controller.network_read(read), {"status": "running"})
            self.assertEqual(len(calls), 2)
            self.assertEqual(controller.state["network_status"], "connected")

    def test_network_wait_does_not_extend_reserved_cloud_deadline(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "ARTIFACTS", Path(folder)):
            write_json(Path(folder) / "cloud_deadline.json", {"deadline_utc": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()})
            controller = pipeline.Controller("test")
            def read(): raise ConnectionResetError("offline")
            with patch.object(pipeline.time, "sleep") as sleep:
                with self.assertRaisesRegex(TimeoutError, "reserved cloud deadline"):
                    controller.network_read(read)
                sleep.assert_not_called()

    def test_completed_stage_survives_controller_restart_without_rerun(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "ARTIFACTS", Path(folder)):
            calls = []
            first = pipeline.Controller("test")
            first.step("one", lambda: calls.append(1) or {"value": 1})
            second = pipeline.Controller("test")
            self.assertEqual(second.step("one", lambda: calls.append(2)), {"value": 1})
            self.assertEqual(calls, [1])

    def test_failed_stage_is_not_automatically_paid_for_again(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "ARTIFACTS", Path(folder)):
            controller = pipeline.Controller("test")
            def fail(): raise ValueError("diagnose me")
            with self.assertRaises(ValueError): controller.step("one", fail)
            with self.assertRaisesRegex(RuntimeError, "no automatic paid retry"):
                pipeline.Controller("test").step("one", lambda: self.fail("must not run"))

    def test_unpassed_gate_cannot_reserve_cloud_or_start_pipeline(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "ARTIFACTS", Path(folder)):
            write_json(Path(folder) / "reports/stage_gate.json", {"approved": False})
            with patch.object(pipeline.Controller, "lease") as lease:
                with self.assertRaisesRegex(RuntimeError, "pilot gate"):
                    pipeline.Controller("test").run()
                lease.assert_not_called()

    def test_final_report_rejects_dev_results_and_incomplete_test_batches(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "batch.json"
            with patch.object(final_report, "DatasetStore") as store:
                store.return_value.select.return_value = ["a", "b"]
                write_json(path, {"config": {"selection": "dev_pilot", "phase": "pilot"}, "runs": []})
                with self.assertRaisesRegex(ValueError, "independent test"):
                    final_report.build({"base": path}, folder)
                write_json(path, {"config": {"selection": "test", "phase": "main"}, "runs": []})
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    final_report.build({"base": path}, folder)


if __name__ == "__main__": unittest.main()
