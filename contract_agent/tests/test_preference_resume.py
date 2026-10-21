"""Interrupted preference production must reuse durable GPU and Judge results."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch, Mock

from contract_agent import preferences as pref
from contract_agent.storage import write_json, read_json, digest
from contract_agent.pipeline import Controller
import contract_agent.pipeline as pipeline


class PreferenceResumeTests(unittest.TestCase):
    def test_shutdown_recovery_keeps_completed_jobs_and_rearms_only_interruption(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "ARTIFACTS", Path(folder)), patch.object(pipeline, "remote_json", return_value={"status": "completed"}), patch.object(pipeline, "push_json"), patch.object(pipeline, "launch", return_value={"exit_code": 0}), patch.object(pipeline, "upload", return_value={"credentials_uploaded": False}), patch.object(pipeline, "Ledger") as ledger:
            controller = Controller("test")
            controller.state.update(current_step="preferences", status="failed", error="old error")
            controller.state["steps"] = {"sft": {"status": "completed"}, "sft-rollouts": {"status": "completed"}, "preferences": {"status": "failed"}, "start-sft-server": {"status": "completed"}}
            ledger.return_value.reserve.return_value = "test-reservation"
            controller.resume_after_shutdown()
            self.assertEqual(controller.state["steps"]["sft"]["status"], "completed")
            self.assertEqual(controller.state["steps"]["preferences"]["status"], "pending")
            self.assertEqual(controller.state["steps"]["start-sft-server"]["status"], "pending")
            self.assertNotIn("error", controller.state)
            ledger.return_value.reserve.assert_called_once()
            self.assertTrue(list((controller.folder / "recoveries").glob("*-previous-state.json")))

    def test_legacy_completed_pair_is_not_generated_or_judged_again(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.json"
            write_json(source, {"runs": [{"id": "run1", "mode": "agent", "task_id": "t1", "steps": [{}]}]})
            policy = Mock(spec=["adapter", "generate_many"])
            policy.adapter = "same-sft"
            context = [{"role": "user", "content": "same"}]
            row = {"id": "pairs-0000", "task_id": "t1", "split": "train", "source_run": "run1", "decision_index": 0,
                   "prompt": context, "prompt_sha256": digest(context), "candidates": [{"text": "A"}, {"text": "B"}],
                   "raw_judge": {"winner": "A"}, "swapped_judge": {"winner": "A"}, "verified_winner": "A", "pilot_only": False}
            path = root / "preferences/pairs/pairs-0000.json"
            write_json(path, row)
            original = path.read_bytes()
            store = Mock()
            store.public.return_value = {}
            with patch.object(pref, "DatasetStore", return_value=store), patch.object(pref, "replay_prefix", return_value=(None, context)), patch.object(pref, "context_messages", side_effect=lambda x: x), patch.object(pref, "judge_once") as judge:
                result = pref.produce_pairs(source, policy, None, limit=1, root=root, output_id="pairs", resume=True)
            self.assertEqual(result["candidate_pairs"], 1)
            self.assertEqual(path.read_bytes(), original)
            policy.generate_many.assert_not_called()
            judge.assert_not_called()

    def test_saved_candidates_and_branches_survive_judge_interruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.json"
            write_json(source, {"runs": [{"id": "run1", "mode": "agent", "task_id": "t1", "steps": [{}]}]})
            policy = Mock(spec=["adapter", "generate_many"])
            policy.adapter = "same-sft"
            policy.generate_many.return_value = [{"text": "A"}, {"text": "B"}]
            store = Mock()
            store.index = {"t1": {"split": "train"}}
            store.public.return_value = {}
            context = [{"role": "user", "content": "unchanged"}]
            branches = [{"steps": [], "status": "submitted"}, {"steps": [], "status": "submitted"}]
            verdict = {"winner": "A"}
            with patch.object(pref, "DatasetStore", return_value=store), patch.object(pref, "replay_prefix", return_value=(None, context)), patch.object(pref, "context_messages", side_effect=lambda x: x), patch.object(pref, "continue_candidates", return_value=branches) as rollout, patch.object(pref, "score_run", return_value={"label_correct": True}), patch.object(pref, "verified_preference", return_value=("A", "test")), patch.object(pref, "judge_once", side_effect=[verdict, ConnectionResetError("offline")]) as judge:
                with self.assertRaises(ConnectionResetError):
                    pref.produce_pairs(source, policy, None, limit=1, root=root, output_id="pairs")
                self.assertEqual(policy.generate_many.call_count, 1)
                self.assertEqual(rollout.call_count, 1)
                with patch.object(pref, "judge_once", return_value=verdict) as resumed_judge:
                    result = pref.produce_pairs(source, policy, None, limit=1, root=root, output_id="pairs", resume=True)
                self.assertEqual(resumed_judge.call_count, 1)  # only the unsaved swapped Judge
                self.assertEqual(policy.generate_many.call_count, 1)
                self.assertEqual(rollout.call_count, 1)
                self.assertEqual(result["candidate_pairs"], 1)

    def test_resume_rejects_changed_source_before_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.json"
            original = {"runs": []}
            write_json(source, original)
            policy = Mock(spec=["adapter", "generate_many"])
            policy.adapter = "same-sft"
            with patch.object(pref, "DatasetStore"):
                pref.produce_pairs(source, policy, None, root=root, output_id="pairs")
                write_json(source, {"runs": [], "changed": True})
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    pref.produce_pairs(source, policy, None, root=root, output_id="pairs", resume=True)
            policy.generate_many.assert_not_called()

    def test_cleanup_connection_error_does_not_replace_main_failure(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "ARTIFACTS", Path(folder)), patch("contract_agent.models.RemoteHFModel", side_effect=ConnectionResetError("cloud off")):
            controller = Controller("test")
            with self.assertRaisesRegex(ValueError, "original generation error"):
                try:
                    raise ValueError("original generation error")
                finally:
                    controller.stop_model_service("sft", preserve_error=True)
            self.assertEqual(controller.state["cleanup_errors"][0]["error_type"], "ConnectionResetError")


if __name__ == "__main__": unittest.main()
