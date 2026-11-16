from __future__ import annotations
import sqlite3
import threading
import math
from pathlib import Path
from .common import dumps, now

class BudgetExceeded(RuntimeError):
    pass

class Ledger:
    """Single project-wide spending ledger, shared by all runs and threads."""
    def __init__(self, path, cap):
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError("Budget cap must be a positive finite dollar amount")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.lock = threading.RLock()
        self.cap = cap
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS attempts (
          id INTEGER PRIMARY KEY, study TEXT, task TEXT, started TEXT,
          status TEXT, reserved REAL, charged REAL, prompt_tokens INTEGER,
          completion_tokens INTEGER, detail TEXT);
        ''')
        self.db.commit()

    def reserve(self, study, task, amount):
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Invalid cost reservation")
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            spent = self.db.execute("SELECT COALESCE(SUM(charged),0) FROM attempts").fetchone()[0]
            if spent + amount > self.cap:
                self.db.rollback()
                raise BudgetExceeded(f"Cumulative estimated cost plus reservation exceeds US${self.cap:.2f}; no request sent.")
            cur = self.db.execute("INSERT INTO attempts(study,task,started,status,reserved,charged) VALUES(?,?,?,?,?,?)", (study,task,now(),"reserved",amount,amount))
            self.db.commit()
            return cur.lastrowid

    def settle(self, attempt_id, charged=None, usage=None, status="complete", detail=""):
        with self.lock:
            if charged is None:
                charged = self.db.execute("SELECT reserved FROM attempts WHERE id=?", (attempt_id,)).fetchone()[0]
            usage = usage or {}
            self.db.execute("UPDATE attempts SET status=?,charged=?,prompt_tokens=?,completion_tokens=?,detail=? WHERE id=?", (status,charged,usage.get("prompt_tokens"),usage.get("completion_tokens"),detail,attempt_id))
            self.db.commit()

    def summary(self):
        with self.lock:
            row = self.db.execute("SELECT COUNT(*),COALESCE(SUM(charged),0),COALESCE(SUM(prompt_tokens),0),COALESCE(SUM(completion_tokens),0) FROM attempts").fetchone()
            return dict(zip(["attempts", "conservative_cost_usd", "prompt_tokens", "completion_tokens"], row))

class TaskStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.lock = threading.RLock()
        self.db.execute("CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, kind TEXT, status TEXT, prompt_hash TEXT, result TEXT, detail TEXT, updated TEXT)")
        self.db.commit()

    def get(self, task_id):
        import json
        with self.lock:
            row = self.db.execute("SELECT status,prompt_hash,result,detail FROM tasks WHERE id=?", (task_id,)).fetchone()
            return None if row is None else {"status":row[0],"prompt_hash":row[1],"result":json.loads(row[2]) if row[2] else None,"detail":row[3]}

    def put(self, task_id, kind, status, prompt_hash, result=None, detail=""):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO tasks VALUES(?,?,?,?,?,?,?)", (task_id,kind,status,prompt_hash,dumps(result) if result is not None else None,detail,now()))
            self.db.commit()

    def results(self):
        import json
        with self.lock:
            return {k:json.loads(v) for k,v in self.db.execute("SELECT id,result FROM tasks WHERE status='done'")}

    def statuses(self):
        with self.lock:
            return {status:n for status,n in self.db.execute("SELECT status,COUNT(*) FROM tasks GROUP BY status")}
