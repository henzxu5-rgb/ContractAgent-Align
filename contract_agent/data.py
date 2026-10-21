"""Reproduce official splits; isolate model-visible content from annotations."""
import collections
import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from pathlib import Path

from .settings import ARTIFACTS, DATA_URL, DATA_SHA256, SEED
from .storage import digest, read_json, write_json, write_jsonl, now


def prepare(root=ARTIFACTS):
    root = Path(root)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "contract-nli.zip"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "ContractAgent-Align/0.1"})
        with urllib.request.urlopen(req, timeout=60) as response, partial.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        partial.replace(archive)
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != DATA_SHA256:
        raise ValueError(f"Dataset checksum differs from reviewed release: {actual}")
    all_docs, gold, split_tasks, counts = {}, {}, {}, {}
    duplicates, text_owners, invalid = [], {}, []
    with zipfile.ZipFile(archive) as z:
        for split in ("train", "dev", "test"):
            payload = json.loads(z.read(f"contract-nli/{split}.json"))
            counts[split] = {"documents": len(payload["documents"]), "tasks": 0, "labels": collections.Counter()}
            tasks = []
            for doc in payload["documents"]:
                doc_id = str(doc["id"])
                if doc_id in all_docs:
                    raise ValueError("Document ID appears in multiple splits")
                for start, end in doc["spans"]:
                    if not 0 <= start < end <= len(doc["text"]):
                        invalid.append([doc_id, start, end])
                key = hashlib.sha256(re.sub(r"\s+", " ", doc["text"]).strip().lower().encode()).hexdigest()
                if key in text_owners:
                    duplicates.append({"previous": text_owners[key], "current": [split, doc_id]})
                text_owners[key] = [split, doc_id]
                all_docs[doc_id] = {"doc_id": doc_id, "split": split, "text": doc["text"],
                                    "spans": doc["spans"], "source_url": doc.get("url"),
                                    "source_file": doc.get("file_name")}
                annotations = doc["annotation_sets"][0]["annotations"]
                for hid, label in payload["labels"].items():
                    tid = f"{doc_id}:{hid}"
                    annotation = annotations[hid]
                    if any(type(i) is not int or not 0 <= i < len(doc["spans"]) for i in annotation["spans"]):
                        raise ValueError(f"Invalid evidence IDs in {tid}")
                    tasks.append({"task_id": tid, "doc_id": doc_id, "split": split,
                                  "hypothesis_id": hid, "hypothesis": label["hypothesis"]})
                    gold[tid] = {"label": annotation["choice"], "evidence_ids": annotation["spans"]}
                    counts[split]["labels"][annotation["choice"]] += 1
            split_tasks[split] = tasks
            counts[split]["tasks"] = len(tasks)
    if invalid:
        raise ValueError(f"Invalid source offsets: {invalid[:3]}")
    # Quarantine exact duplicate documents from training/dev if they occur in a later split.
    excluded = {p["previous"][1] for p in duplicates if p["previous"][0] != p["current"][0]}
    for split in ("train", "dev"):
        split_tasks[split] = [t for t in split_tasks[split] if t["doc_id"] not in excluded]
    processed = root / "data"
    for name, value in (("documents", all_docs), ("annotations", gold), ("tasks", split_tasks)):
        path = processed / f"{name}.json"
        if path.exists() and read_json(path) != value:
            raise ValueError(f"Existing data artifact differs: {path}")
        if not path.exists():
            write_json(path, value, exclusive=True)
    # Document selection is label-blind and independent of model outcomes.
    rng = random.Random(SEED)
    test_docs = sorted({t["doc_id"] for t in split_tasks["test"]}, key=int)
    chosen_docs = set(rng.sample(test_docs, 20))
    test = [t["task_id"] for t in split_tasks["test"] if t["doc_id"] in chosen_docs]
    # Balanced training order is permitted on training labels; keep all hypotheses covered.
    groups = collections.defaultdict(list)
    for task in split_tasks["train"]:
        groups[(task["hypothesis_id"], gold[task["task_id"]]["label"])].append(task)
    for group in groups.values():
        rng.shuffle(group)
    ordered = []
    keys = sorted(groups)
    while groups:
        for key in keys:
            if groups.get(key):
                ordered.append(groups[key].pop()["task_id"])
                if not groups[key]:
                    del groups[key]
    training = ordered[:300]
    if len({tid.split(":")[0] for tid in training[:200]}) < 60:
        raise ValueError("Training task selection failed document diversity floor")
    # Pilot uses 18 class-balanced development tasks; test labels never guide selection.
    devgroups = collections.defaultdict(list)
    for task in split_tasks["dev"]:
        devgroups[gold[task["task_id"]]["label"]].append(task["task_id"])
    for group in devgroups.values():
        rng.shuffle(group)
    pilot = [tid for group in devgroups.values() for tid in group[:6]]
    manifests = {"seed": SEED, "dataset_sha256": actual, "train": training,
                 "train_initial": training[:200], "dev_pilot": pilot, "test": test,
                 "test_document_ids": sorted(chosen_docs, key=int),
                 "excluded_duplicate_document_ids": sorted(excluded)}
    mpath = processed / "manifest.json"
    if mpath.exists() and read_json(mpath) != manifests:
        raise ValueError("Refusing to silently change frozen manifests")
    if not mpath.exists():
        write_json(mpath, manifests, exclusive=True)
    audit = {"recorded_at": now(), "source": DATA_URL, "sha256": actual, "license": "CC BY 4.0",
             "annotation_provenance": "Researcher + crowd workers; hypotheses reviewed by paralegals",
             "counts": counts, "invalid_offsets": invalid, "exact_duplicates": duplicates,
             "manifest_sha256": digest(manifests), "test_tasks": len(test),
             "training_initial_documents": len({tid.split(':')[0] for tid in training[:200]}),
             "limitation": "Whitespace-normalized exact duplicates only; no guarantee against near duplicates or pretraining contamination."}
    write_json(root / "reports" / "data_audit.json", audit)
    return audit


class DatasetStore:
    def __init__(self, root=ARTIFACTS):
        self.root = Path(root)
        self.documents = read_json(self.root / "data/documents.json")
        self.tasks = read_json(self.root / "data/tasks.json")
        self.index = {t["task_id"]: t for rows in self.tasks.values() for t in rows}
        self.manifest = read_json(self.root / "data/manifest.json")

    def public(self, task_id):
        task = self.index[task_id]
        doc = self.documents[task["doc_id"]]
        # Explicit allowlist prevents labels/source URLs/metadata from leaking into the task.
        return {"task_id": task_id, "doc_id": doc["doc_id"], "hypothesis": task["hypothesis"],
                "text": doc["text"], "spans": doc["spans"]}

    def gold(self, task_id):
        return read_json(self.root / "data/annotations.json")[task_id]

    def select(self, name, limit=None):
        return self.manifest[name][:limit] if limit is not None else self.manifest[name]
