import json
from pathlib import Path
import tempfile
import threading
import unittest

from contract_agent.budget import BudgetExceeded, Ledger
from contract_agent.environment import ContractEnvironment
from contract_agent.runner import initial_messages, run_episode
from contract_agent.scoring import score_run, summarize


def task():
    text = "Recipient may share information with employees. Recipient must not share information with third parties."
    return {"task_id": "unit:nda-6", "doc_id": "unit", "hypothesis": "Recipient may share information with third parties.",
            "text": text, "spans": [[0, 46], [47, len(text)]]}


def submit(ids, label="Contradiction"):
    return {"tool": "submit_result", "arguments": {"label": label, "evidence_ids": ids, "explanation": "The contract prohibits this."}}


class ScriptedModel:
    name = "unit-test-double"
    def __init__(self, actions):
        self.actions = iter(actions)
    def generate(self, messages, **kwargs):
        return {"text": json.dumps(next(self.actions)), "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "model": self.name}


class EnvironmentTests(unittest.TestCase):
    def test_gold_never_enters_prompt_or_tool_results(self):
        public = task()
        text = json.dumps(initial_messages(public))
        self.assertNotIn('"choice"', text)
        self.assertNotIn('"annotation_sets"', text)
        self.assertNotIn('"gold_label"', text)
        env = ContractEnvironment(public)
        self.assertNotIn("annotations", json.dumps(env.execute({"tool": "read_spans", "arguments": {"span_ids": [0]}})))

    def test_unread_and_out_of_range_references_fail(self):
        env = ContractEnvironment(task())
        self.assertIn("error", env.execute(submit([1])))
        self.assertIn("error", env.execute({"tool": "read_spans", "arguments": {"span_ids": [-1]}}))
        self.assertIn("error", env.execute({"tool": "read_spans", "arguments": {"span_ids": [True]}}))
        env.execute({"tool": "read_spans", "arguments": {"span_ids": [1]}})
        self.assertTrue(env.execute(submit([1]))["submitted"])

    def test_invalid_calls_consume_budget(self):
        env = ContractEnvironment(task())
        for _ in range(6):
            self.assertEqual(env.execute({"tool": "missing", "arguments": {}})["error"], "unknown_tool")
        self.assertEqual(env.execute({"tool": "read_spans", "arguments": {"span_ids": [1]}})["error"], "tool_budget_exhausted")

    def test_empty_search_is_not_automatic_notmentioned(self):
        env = ContractEnvironment(task())
        self.assertEqual(env.execute({"tool": "search_contract", "arguments": {"query": "xyzq"}})["matches"], [])
        self.assertIsNone(env.submission)

    def test_state_reset_and_branch_are_independent(self):
        env = ContractEnvironment(task()); branch = env.clone()
        branch.execute({"tool": "read_spans", "arguments": {"span_ids": [1]}})
        self.assertEqual(env.observed, set())
        self.assertEqual(ContractEnvironment(task()).observed, set())

    def test_full_context_has_all_spans_and_no_read_tools(self):
        env = ContractEnvironment(task(), full_context=True)
        self.assertEqual(env.observed, {0, 1})
        self.assertIn("error", env.execute({"tool": "read_spans", "arguments": {"span_ids": [1]}}))

    def test_real_runner_preserves_failures_and_observed_evidence(self):
        model = ScriptedModel([{"tool": "missing", "arguments": {}}, {"tool": "read_spans", "arguments": {"span_ids": [1]}}, submit([1])])
        run = run_episode(task(), model, save=False)
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["tool_errors"], ["unknown_tool"])
        s = score_run(run, {"label": "Contradiction", "evidence_ids": [1]})
        self.assertTrue(s["strict_success"])

    def test_not_submitted_is_failure_even_with_apparent_prediction(self):
        run = {"status": "not_submitted", "submission": {"label": "Contradiction", "evidence_ids": [1]}, "observed_ids": [1]}
        s = score_run(run, {"label": "Contradiction", "evidence_ids": [1]})
        self.assertFalse(s["label_correct"])

    def test_rejected_citation_is_counted_even_after_recovery(self):
        model = ScriptedModel([submit([1]), {"tool": "read_spans", "arguments": {"span_ids": [1]}}, submit([1])])
        run = run_episode(task(), model, save=False)
        run["score"] = score_run(run, {"label": "Contradiction", "evidence_ids": [1]})
        self.assertEqual(run["score"]["invalid_citation_attempts"], 1)
        self.assertEqual(run["score"]["citation_attempt_count"], 2)
        self.assertEqual(summarize([run])["invalid_citation_rate"], .5)

    def test_malformed_submission_arguments_do_not_crash_scoring(self):
        model = ScriptedModel([{"tool": "submit_result", "arguments": None}, submit([], "NotMentioned")])
        run = run_episode(task(), model, save=False)
        self.assertTrue(score_run(run, {"label": "NotMentioned", "evidence_ids": []})["label_correct"])

    def test_macro_f1_counts_failed_outputs(self):
        run = {"status": "failed", "submission": None, "observed_ids": []}
        run["score"] = score_run(run, {"label": "Entailment", "evidence_ids": [0]})
        summary = summarize([run])
        self.assertEqual(summary["macro_f1"], 0)
        self.assertEqual(summary["confusion_matrix"]["Entailment"]["INVALID"], 1)

    def test_notmentioned_has_no_evidence_f1(self):
        run = {"status": "completed", "submission": {"label": "NotMentioned", "evidence_ids": []}, "observed_ids": []}
        run["score"] = score_run(run, {"label": "NotMentioned", "evidence_ids": []})
        self.assertIsNone(summarize([run])["evidence"]["f1"])


class BudgetTests(unittest.TestCase):
    def test_pilot_cap_and_uncertain_requests(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = Ledger(Path(d) / "budget.sqlite")
            rid = ledger.reserve(19, "gpu")
            with self.assertRaises(BudgetExceeded): ledger.reserve(2)
            ledger.settle(rid)  # Unknown actual usage consumes full reservation.
            self.assertEqual(ledger.summary()["committed_cny"], 19)

    def test_concurrent_reservations_cannot_overspend(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = Ledger(Path(d) / "budget.sqlite"); outcomes = []
            def worker():
                try: ledger.reserve(12); outcomes.append(True)
                except BudgetExceeded: outcomes.append(False)
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for t in threads: t.start()
            for t in threads: t.join()
            self.assertEqual(sorted(outcomes), [False, True])


if __name__ == "__main__": unittest.main()
