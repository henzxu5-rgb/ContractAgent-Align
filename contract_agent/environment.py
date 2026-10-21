"""A deterministic, cloneable contract-only tool environment with no gold labels."""
import collections
import copy
import math
import re

from .settings import LABELS, MAX_TOOL_CALLS

TOOLS = [
    {"name": "search_contract", "arguments": {"query": "string", "k": "integer 1..6, default 5"},
     "description": "BM25 search inside this contract only; returns numbered original evidence spans."},
    {"name": "read_spans", "arguments": {"span_ids": "array of 1..8 integer IDs"},
     "description": "Read full numbered spans. IDs are zero-based, immutable within this contract."},
    {"name": "submit_result", "arguments": {"label": list(LABELS), "evidence_ids": "array of observed integer IDs", "explanation": "short string"},
     "description": "Submit conclusion. NotMentioned requires no evidence IDs; a failed search alone is not proof of absence."},
]


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class ContractEnvironment:
    def __init__(self, public_task, full_context=False):
        self.task = copy.deepcopy(public_task)
        self.span_texts = [self.task["text"][a:b] for a, b in self.task["spans"]]
        self.observed = set(range(len(self.span_texts))) if full_context else set()
        self.tool_calls = 0
        self.errors = []
        self.submission = None
        self.events = []
        self.full_context = full_context
        self._terms = [collections.Counter(tokens(t)) for t in self.span_texts]
        self._avglen = sum(sum(c.values()) for c in self._terms) / max(1, len(self._terms))
        self._df = collections.Counter(t for c in self._terms for t in c)

    def clone(self):
        return copy.deepcopy(self)

    def _span(self, i):
        a, b = self.task["spans"][i]
        return {"span_id": i, "start": a, "end": b, "text": self.span_texts[i]}

    def search(self, query, k):
        query_terms = set(tokens(query))
        scores = []
        n = len(self._terms)
        for i, counts in enumerate(self._terms):
            length = sum(counts.values())
            score = 0.0
            for term in query_terms:
                freq = counts.get(term, 0)
                idf = math.log(1 + (n - self._df[term] + .5) / (self._df[term] + .5))
                score += idf * freq * 2.5 / (freq + 1.5 * (.25 + .75 * length / max(1, self._avglen)))
            if score > 0:
                scores.append((-score, i))
        ids = [i for _, i in sorted(scores)[:k]]
        self.observed.update(ids)
        return {"matches": [self._span(i) for i in ids], "retriever": "BM25 k1=1.5 b=0.75"}

    def execute(self, action):
        try:
            if self.submission is not None:
                raise ValueError("already_submitted")
            if not isinstance(action, dict) or set(action) != {"tool", "arguments"}:
                raise ValueError("invalid_action_schema")
            name, args = action["tool"], action["arguments"]
            if not isinstance(args, dict):
                raise ValueError("arguments_must_be_object")
            if name == "submit_result":
                if set(args) != {"label", "evidence_ids", "explanation"}:
                    raise ValueError("invalid_submission_schema")
                if args["label"] not in LABELS:
                    raise ValueError("invalid_label")
                ids = args["evidence_ids"]
                if not isinstance(ids, list) or any(type(i) is not int for i in ids) or len(ids) != len(set(ids)):
                    raise ValueError("invalid_evidence_ids")
                if any(i not in self.observed for i in ids):
                    raise ValueError("unobserved_or_invalid_evidence")
                if args["label"] == "NotMentioned" and ids:
                    raise ValueError("notmentioned_requires_empty_evidence")
                if args["label"] != "NotMentioned" and not ids:
                    raise ValueError("non_neutral_requires_evidence")
                if not isinstance(args["explanation"], str) or len(args["explanation"]) > 1600:
                    raise ValueError("invalid_explanation")
                self.submission = copy.deepcopy(args)
                result = {"submitted": True}
            else:
                self.tool_calls += 1  # Invalid calls also consume the fixed action budget.
                if self.full_context:
                    raise ValueError("full_context_baseline_has_no_read_tools")
                if self.tool_calls > MAX_TOOL_CALLS:
                    raise ValueError("tool_budget_exhausted")
                if name == "search_contract":
                    if set(args) - {"query", "k"} or not isinstance(args.get("query"), str) or not args["query"].strip():
                        raise ValueError("invalid_search_parameters")
                    k = args.get("k", 5)
                    if type(k) is not int or not 1 <= k <= 6:
                        raise ValueError("invalid_search_k")
                    result = self.search(args["query"], k)
                elif name == "read_spans":
                    ids = args.get("span_ids")
                    if set(args) != {"span_ids"} or not isinstance(ids, list) or not 1 <= len(ids) <= 8:
                        raise ValueError("invalid_read_parameters")
                    if any(type(i) is not int or not 0 <= i < len(self.span_texts) for i in ids):
                        raise ValueError("invalid_span_id")
                    self.observed.update(ids)
                    result = {"spans": [self._span(i) for i in ids]}
                else:
                    raise ValueError("unknown_tool")
        except (ValueError, TypeError, KeyError) as e:
            result = {"error": str(e)}
            self.errors.append(str(e))
        self.events.append({"action": copy.deepcopy(action), "result": result})
        return result


def parse_action(text):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    action = __import__("json").loads(text)
    if not isinstance(action, dict):
        raise ValueError("action_not_object")
    return action
