import copy
import tempfile
from pathlib import Path
import unittest

from contract_agent.preferences import verified_preference, export_pairs, tool_behavior_difference
from contract_agent.storage import digest, read_jsonl
from contract_agent.training import prepare_rows


def branch(correct=True, f1=1):
    return {"score": {"label_correct": correct, "invalid_citations": 0, "evidence": {"f1": f1}}}


class JudgeTests(unittest.TestCase):
    def test_identical_read_behavior_is_not_counted_as_tool_difference(self):
        candidates = [{"text": '{"tool":"read_spans","arguments":{"span_ids":[1,2]}}'},
                      {"text": '{"tool":"read_spans","arguments":{"span_ids":[2,1]}}'}]
        self.assertFalse(tool_behavior_difference(candidates))
        candidates[1]["text"] = '{"tool":"search_contract","arguments":{"query":"restriction"}}'
        self.assertTrue(tool_behavior_difference(candidates))
        candidates = [{"text": '{"tool":"search_contract","arguments":{"query":"Confidential restriction"}}'},
                      {"text": '{"tool":"search_contract","arguments":{"query":"restriction confidential!","k":5}}'}]
        self.assertFalse(tool_behavior_difference(candidates))
    def test_confident_judge_cannot_override_known_wrong_answer(self):
        winner, reason = verified_preference(branch(True), branch(False), {"winner": "B"}, {"winner": "B"})
        self.assertEqual(winner, "A")
        self.assertEqual(reason, "reference_label_and_completed_execution")

    def test_both_wrong_are_not_forced_into_preference(self):
        self.assertEqual(verified_preference(branch(False), branch(False), {"winner": "A"}, {"winner": "A"})[0], "abstain")

    def test_order_unstable_tied_results_are_quarantined(self):
        self.assertEqual(verified_preference(branch(), branch(), {"winner": "A"}, {"winner": "B"})[0], "abstain")

    def test_matched_exports_change_labels_not_candidate_pool(self):
        prompt = [{"role": "user", "content": "question"}]
        record = {"id": "p1", "task_id": "1:nda-1", "split": "train", "prompt": prompt,
                  "prompt_sha256": digest(prompt), "candidates": [{"text": '{"tool":"search_contract","arguments":{"query":"x"}}'}, {"text": '{"tool":"submit_result","arguments":{}}'}],
                  "raw_judge": {"winner": "B"}, "swapped_judge": {"winner": "A"}, "verified_winner": "A"}
        with tempfile.TemporaryDirectory() as d:
            out = export_pairs([record], Path(d))
            a, b = read_jsonl(out["raw_file"])[0], read_jsonl(out["verified_file"])[0]
            self.assertEqual(a["prompt"], b["prompt"])
            self.assertEqual(a["chosen"], b["rejected"])
            self.assertEqual(a["rejected"], b["chosen"])


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return ''.join('<'+m['role']+'>'+m['content']+'</end>' for m in messages) + ('<assistant>' if add_generation_prompt else '')
    def __call__(self, text, **kwargs):
        return {"input_ids": list(text.encode())}


class TrainingDataTests(unittest.TestCase):
    def row(self):
        return {"id": "x", "split": "train", "prompt": [{"role": "user", "content": "q"}],
                "completion": [{"role": "assistant", "content": "action"}]}
    def test_test_data_cannot_enter_training(self):
        row = self.row(); row["split"] = "test"
        with self.assertRaisesRegex(ValueError, "leakage"): prepare_rows([row], FakeTokenizer(), "sft")
    def test_smoke_examples_cannot_become_formal_examples(self):
        row = self.row(); row.update(split="dev", pilot_only=True)
        with self.assertRaises(ValueError): prepare_rows([row], FakeTokenizer(), "sft", pilot=False)
        self.assertEqual(len(prepare_rows([row], FakeTokenizer(), "sft", pilot=True)[0]), 1)
    def test_action_and_prompt_are_separate(self):
        rows, excluded = prepare_rows([self.row()], FakeTokenizer(), "sft")
        self.assertNotIn('action', rows[0]['prompt'])
        self.assertEqual(rows[0]['completion'], 'action</end>')
    def test_overlength_is_excluded_not_silently_truncated(self):
        short = self.row(); long = copy.deepcopy(short); long['id'] = 'long'; long['completion'][0]['content'] = 'x' * 5000
        rows, excluded = prepare_rows([short, long], FakeTokenizer(), 'sft')
        self.assertEqual(len(rows), 1)
        self.assertEqual(excluded[0]['id'], 'long')


if __name__ == "__main__": unittest.main()
