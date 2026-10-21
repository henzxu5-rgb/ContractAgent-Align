import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from contract_agent import preference_jobs as jobs
from contract_agent.storage import write_json, digest


class SplitPreferenceJobTests(unittest.TestCase):
    def test_parallel_candidates_keep_unique_stable_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.json"
            ids = [f"{i}:h1" for i in range(1, 5)]
            write_json(source, {"runs": [{"id": f"r{i}", "task_id": tid, "mode": "agent", "steps": [{}]} for i, tid in enumerate(ids)]})
            store = Mock(); store.select.return_value = ids; store.index = {tid: {"split": "train"} for tid in ids}
            policy = Mock(); policy.name = "test"; policy.adapter = "sft"; policy.sampling_seed = 1
            policy.generate_many.side_effect = lambda groups, **kw: [{"text": "identical"} for _ in groups]
            with patch.object(jobs, "DatasetStore", return_value=store), patch.object(jobs, "replay_prefix", return_value=(None, [])), patch.object(jobs, "context_messages", return_value=[]):
                jobs.generate(source, policy, "parallel", 4, root=root, parallel=4)
                from contract_agent.storage import read_json
                rows = read_json(root / "candidate_batches/parallel/batch.json")["records"]
                self.assertEqual([r["id"] for r in rows], [f"parallel-{i:04d}" for i in range(4)])
                self.assertEqual([r["task_id"] for r in rows], ids)
                self.assertEqual(sum(len(call.args[0]) for call in policy.generate_many.call_args_list), 8)

    def test_identical_actions_count_toward_cap_without_branch_or_judge_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            write_json(source, {"runs": [{"id": "r1", "task_id": "1:h1", "mode": "agent", "steps": [{}]}]})
            store = Mock(); store.select.return_value = ["1:h1"]; store.index = {"1:h1": {"split": "train"}}
            policy = Mock(); policy.adapter = "sft"; policy.sampling_seed = 1
            policy.generate_many.return_value = [{"text": "same"}, {"text": "same"}]
            with patch.object(jobs, "DatasetStore", return_value=store), patch.object(jobs, "replay_prefix", return_value=(None, [])), patch.object(jobs, "context_messages", return_value=[]), patch.object(jobs, "continue_candidates") as branch, patch.object(jobs, "judge_once") as judge:
                self.assertEqual(jobs.generate(source, policy, "candidates", 1, root=root)["candidate_pairs"], 1)
                result = jobs.label(root / "candidate_batches/candidates/batch.json", None, "labels", root=root)
                self.assertEqual(result["identical_action_pairs"], 1)
                self.assertEqual(result["matched_pairs"], 0)
                branch.assert_not_called(); judge.assert_not_called()

    def test_schedule_visits_diverse_tasks_before_deeper_decisions(self):
        source = {"runs": [{"id": "a", "task_id": "t1", "mode": "agent", "steps": [{}, {}]},
                           {"id": "b", "task_id": "t2", "mode": "agent", "steps": [{}, {}]}]}
        self.assertEqual([(r["task_id"], i) for r, i in jobs.state_schedule(source, ["t2", "t1"])],
                         [("t2", 0), ("t1", 1), ("t2", 1), ("t1", 0)])
        with self.assertRaises(ValueError): jobs.state_schedule(source, ["t1"])

    def test_gpu_candidates_are_reused_after_branch_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            write_json(source, {"runs": [{"id": "r1", "task_id": "1:h1", "mode": "agent", "steps": [{}]}]})
            store = Mock()
            store.select.return_value = ["1:h1"]
            store.index = {"1:h1": {"split": "train"}}
            store.public.return_value = {}
            policy = Mock()
            policy.adapter = "sft"
            policy.sampling_seed = 1
            policy.generate_many.return_value = [{"text": "A"}, {"text": "B"}]
            with patch.object(jobs, "DatasetStore", return_value=store), patch.object(jobs, "replay_prefix", return_value=(None, [])), patch.object(jobs, "context_messages", return_value=[]), patch.object(jobs, "score_run", return_value={"label_correct": True}), patch.object(jobs, "continue_candidates", side_effect=[ConnectionResetError("interrupted"), [{}, {}]]):
                with self.assertRaises(ConnectionResetError): jobs.generate(source, policy, "candidates", 1, root=root)
                result = jobs.generate(source, policy, "candidates", 1, resume=True, root=root)
                self.assertEqual(result["candidate_pairs"], 1)
                self.assertEqual(policy.generate_many.call_count, 1)

    def test_local_judge_resumes_without_any_gpu_or_repeated_completed_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "candidates.json"
            prompt = [{"role": "user", "content": "contract"}]
            row = {"id": "c1", "task_id": "1:h1", "split": "train", "source_run": "r1", "decision_index": 0,
                   "prompt": prompt, "prompt_sha256": digest(prompt), "candidates": [{"text": "A"}, {"text": "B"}], "branches": [{}, {}]}
            write_json(source, {"config": {"fixed": True}, "records": [row]})
            store = Mock(); store.index = {"1:h1": {"split": "train"}}
            verdict = {"winner": "A"}
            with patch.object(jobs, "DatasetStore", return_value=store), patch.object(jobs, "score_run", return_value={"label_correct": True}), patch.object(jobs, "verified_preference", return_value=("A", "test")):
                with patch.object(jobs, "judge_once", side_effect=[verdict, ConnectionResetError("API offline")]):
                    with self.assertRaises(ConnectionResetError): jobs.label(source, None, "labels", root=root)
                with patch.object(jobs, "judge_once", return_value=verdict) as judge:
                    result = jobs.label(source, None, "labels", root=root)
                    self.assertEqual(judge.call_count, 1)
                    self.assertEqual(result["matched_pairs"], 1)
                with patch.object(jobs, "judge_once") as judge:
                    jobs.label(source, None, "labels", root=root)
                    judge.assert_not_called()


if __name__ == "__main__": unittest.main()
