"""Transactional reservations: concurrent requests cannot silently exceed caps."""
import json
from pathlib import Path
import sqlite3
import uuid
from contextlib import contextmanager

from .settings import ARTIFACTS, API_BUDGET, GPU_BUDGET, TOTAL_BUDGET, PILOT_BUDGET
from .storage import now


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path=None):
        self.path = Path(path or ARTIFACTS / "budget.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS costs (id TEXT PRIMARY KEY, phase TEXT, category TEXT, reserved REAL, actual REAL, status TEXT, created TEXT, updated TEXT, metadata TEXT)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            yield db
        finally:
            db.close()

    def reserve(self, amount, category="api", phase="pilot", metadata=None):
        if amount < 0 or category not in ("api", "gpu", "storage", "reserve"):
            raise ValueError("invalid_budget_reservation")
        uid = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT phase,category,CASE WHEN status='reserved' THEN reserved ELSE actual END FROM costs").fetchall()
            total = sum(r[2] for r in rows)
            by_category = sum(r[2] for r in rows if r[1] == category)
            pilot = sum(r[2] for r in rows if r[0] == "pilot")
            caps = {"api": API_BUDGET, "gpu": GPU_BUDGET, "storage": 10, "reserve": 50}
            if total + amount > TOTAL_BUDGET + 1e-8 or by_category + amount > caps[category] + 1e-8 or (phase == "pilot" and pilot + amount > PILOT_BUDGET + 1e-8):
                db.rollback()
                raise BudgetExceeded(f"Budget cap reached: total={total:.4f}, category={by_category:.4f}, pilot={pilot:.4f}, requested={amount:.4f}")
            db.execute("INSERT INTO costs VALUES (?,?,?,?,?,?,?,?,?)", (uid, phase, category, amount, None, "reserved", now(), now(), json.dumps(metadata or {})))
            db.commit()
        return uid

    def settle(self, uid, actual=None, metadata=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT reserved,status,metadata FROM costs WHERE id=?", (uid,)).fetchone()
            if not row or row[1] != "reserved":
                raise ValueError("unknown_or_settled_reservation")
            actual = row[0] if actual is None else actual
            if actual < 0 or actual > row[0] + 1e-8:
                raise ValueError("Actual charge exceeds reservation; stop and reconcile")
            meta = json.loads(row[2]); meta.update(metadata or {})
            db.execute("UPDATE costs SET actual=?,status='settled',updated=?,metadata=? WHERE id=?", (actual, now(), json.dumps(meta), uid))
            db.commit()

    def summary(self):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            rows = [dict(r) for r in db.execute("SELECT * FROM costs ORDER BY created")]
        amount = lambda r: r["reserved"] if r["status"] == "reserved" else r["actual"]
        return {"committed_cny": sum(amount(r) for r in rows), "limits_cny": {"total": TOTAL_BUDGET, "pilot": PILOT_BUDGET, "api": API_BUDGET, "gpu": GPU_BUDGET},
                "categories": {k: sum(amount(r) for r in rows if r["category"] == k) for k in ("api", "gpu", "storage", "reserve")},
                "unsettled": sum(r["status"] == "reserved" for r in rows), "entries": rows}

    def reconcile_gpu_bound(self, uid, corrected_upper, reason):
        """Only reduce a GPU estimate; retain the old value and evidence in its audit trail."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT category,reserved,actual,status,metadata FROM costs WHERE id=?", (uid,)).fetchone()
            if not row or row[0] != "gpu" or row[3] != "settled" or not 0 <= corrected_upper <= row[2]:
                raise ValueError("Only settled GPU upper estimates can be reconciled downward")
            meta = json.loads(row[4])
            meta.setdefault("reconciliations", []).append({"time": now(), "previous_upper": row[2], "new_upper": corrected_upper, "reason": reason})
            db.execute("UPDATE costs SET actual=?,updated=?,metadata=? WHERE id=?", (corrected_upper, now(), json.dumps(meta), uid))
            db.commit()
