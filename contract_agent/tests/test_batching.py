import json
from concurrent.futures import ThreadPoolExecutor
import unittest

from contract_agent.batching import BatchingModel
from contract_agent.runner import run_episode


class FakeBatchPolicy:
    name = "batch-test-double"
    def __init__(self, fail_one=False):
        self.sizes, self.fail_one = [], fail_one

    def generate_many(self, prompts, **kwargs):
        self.sizes.append(len(prompts))
        outputs = []
        for messages in prompts:
            public = json.loads(messages[1]["content"])
            index = 0 if public["task_id"].startswith("a:") else 1
            if self.fail_one and index == 1:
                outputs.append({"error_type": "ValueError", "error": "context_overflow"}); continue
            if len(messages) == 2:
                action = {"tool": "read_spans", "arguments": {"span_ids": [index]}}
            else:
                action = {"tool": "submit_result", "arguments": {"label": "Entailment", "evidence_ids": [index], "explanation": "Observed clause."}}
            outputs.append({"text": json.dumps(action), "model": self.name, "seconds": .1, "amortized_seconds": .1 / len(prompts)})
        return outputs


def public_task(doc):
    return {"task_id": doc + ":nda-1", "doc_id": doc, "hypothesis": "Disclosure permitted.",
            "text": "Clause A. Clause B.", "spans": [[0, 9], [10, 19]]}


class BatchingTests(unittest.TestCase):
    def test_original_runner_keeps_independent_observations_in_one_gpu_batch(self):
        policy = FakeBatchPolicy()
        with BatchingModel(policy, 2, .05) as batched, ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_episode, public_task(doc), batched, save=False) for doc in ("a", "b")]
            runs = [f.result() for f in futures]
        self.assertIn(2, policy.sizes)
        self.assertEqual([r["status"] for r in runs], ["completed", "completed"])
        self.assertEqual([r["observed_ids"] for r in runs], [[0], [1]])
        self.assertEqual([r["submission"]["evidence_ids"] for r in runs], [[0], [1]])
        self.assertTrue(all(r["inference_scheduler"] == "dynamic_gpu_batch_v1" for r in runs))

    def test_one_failed_input_does_not_drop_the_other_episode(self):
        with BatchingModel(FakeBatchPolicy(True), 2, .05) as batched, ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_episode, public_task(doc), batched, save=False) for doc in ("a", "b")]
            good, bad = [f.result() for f in futures]
        self.assertEqual(good["status"], "completed")
        self.assertEqual(bad["status"], "failed")
        self.assertIn("context_overflow", bad["error"])


if __name__ == "__main__": unittest.main()
