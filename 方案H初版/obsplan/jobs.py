"""SQLite 持久队列：短事务领取任务，计算在独立 worker 中执行。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class Conflict(ValueError):
    pass


class QueueFull(ValueError):
    pass


class Store:
    def __init__(self, root: str | Path, capacity: int = 8):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.db = self.root / "jobs.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, idempotency_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL, request TEXT NOT NULL,
                    state TEXT NOT NULL, stage TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0,
                    created REAL NOT NULL, updated REAL NOT NULL, error TEXT,
                    revision INTEGER NOT NULL DEFAULT 0,
                    validation_state TEXT NOT NULL DEFAULT 'not_started', validation_error TEXT,
                    UNIQUE(owner, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    kind TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'queued'
                );
                CREATE TABLE IF NOT EXISTS worker (id INTEGER PRIMARY KEY CHECK(id=1), heartbeat REAL);
            """)

    @contextmanager
    def connect(self, transaction=False):
        db = sqlite3.connect(self.db, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            if transaction:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def get(self, rid, owner=None):
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
        if row is None or (owner is not None and row["owner"] != owner):
            raise KeyError(rid)
        out = dict(row)
        out["request"] = json.loads(out["request"])
        return out

    def list(self, owner):
        with self.connect() as db:
            return [self.get(r["id"], owner) for r in db.execute(
                "SELECT id FROM runs WHERE owner=? ORDER BY created DESC LIMIT 100", (owner,))]

    def _check_capacity(self, db):
        count = db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]
        if count >= self.capacity:
            raise QueueFull("计算队列已满，请稍后重试")

    def submit(self, owner, key, request):
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self.connect(True) as db:
            prev = db.execute("SELECT id,request_hash FROM runs WHERE owner=? AND idempotency_key=?",
                              (owner, key)).fetchone()
            if prev:
                if prev["request_hash"] != digest:
                    raise Conflict("同一个提交标识已用于不同参数，请创建新任务")
                rid, created = prev["id"], False
            else:
                self._check_capacity(db)
                rid, created, now = uuid.uuid4().hex, True, time.time()
                db.execute("""INSERT INTO runs
                    (id,owner,idempotency_key,request_hash,request,state,stage,created,updated)
                    VALUES (?,?,?,?,?,'queued','等待计算',?,?)""", (rid, owner, key, digest, encoded, now, now))
                db.execute("INSERT INTO jobs(run_id,kind) VALUES (?,'plan')", (rid,))
        return self.get(rid, owner), created

    def validate(self, rid, owner):
        with self.connect(True) as db:
            row = db.execute("SELECT * FROM runs WHERE id=? AND owner=?", (rid, owner)).fetchone()
            if not row:
                raise KeyError(rid)
            if row["state"] != "ready":
                raise Conflict("计划尚未完成，不能启动独立校验")
            if row["validation_state"] not in ("queued", "running", "passed", "issues"):
                self._check_capacity(db)
                db.execute("UPDATE runs SET validation_state='queued',validation_error=NULL,updated=? WHERE id=?",
                           (time.time(), rid))
                db.execute("INSERT INTO jobs(run_id,kind) VALUES (?,'validate')", (rid,))
        return self.get(rid, owner)

    def claim(self):
        with self.connect(True) as db:
            row = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY id LIMIT 1").fetchone()
            if not row:
                return None
            db.execute("UPDATE jobs SET state='running' WHERE id=?", (row["id"],))
            if row["kind"] == "plan":
                db.execute("UPDATE runs SET state='running',stage='准备数据与窗口',updated=? WHERE id=?",
                           (time.time(), row["run_id"]))
            else:
                db.execute("UPDATE runs SET validation_state='running',updated=? WHERE id=?",
                           (time.time(), row["run_id"]))
            return dict(row)

    def progress(self, rid, fraction, stage):
        with self.connect() as db:
            db.execute("UPDATE runs SET progress=?,stage=?,updated=? WHERE id=?",
                       (min(.99, max(0., float(fraction))), stage, time.time(), rid))

    def finish(self, job, revision=0, verdict=None, error=None):
        with self.connect(True) as db:
            db.execute("UPDATE jobs SET state=? WHERE id=?", ("failed" if error else "done", job["id"]))
            rid, now = job["run_id"], time.time()
            if job["kind"] == "plan":
                db.execute("""UPDATE runs SET state=?,stage=?,error=?,revision=?,progress=?,updated=? WHERE id=?""",
                           ("failed" if error else "ready", "规划失败" if error else "结果就绪", error,
                            revision, 0 if error else 1, now, rid))
            else:
                db.execute("""UPDATE runs SET validation_state=?,validation_error=?,
                    revision=CASE WHEN ? > 0 THEN ? ELSE revision END,updated=? WHERE id=?""",
                           ("failed" if error else verdict, error, revision, revision, now, rid))

    def recover(self):
        """仅持有单 worker 锁的进程调用；中断不伪装为成功，也不无限自动重试。"""
        with self.connect() as db:
            jobs = [dict(r) for r in db.execute("SELECT * FROM jobs WHERE state='running'")]
        for job in jobs:
            self.finish(job, error="计算进程中断；已保留完整历史产物，请重新运行或重试校验")

    def heartbeat(self):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO worker VALUES(1,?)", (time.time(),))

    def health(self):
        with self.connect() as db:
            beat = db.execute("SELECT heartbeat FROM worker WHERE id=1").fetchone()
            active = db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]
        return {"worker_online": bool(beat and time.time() - beat[0] < 30), "active_jobs": active}

    def artifact_dir(self, rid, revision):
        # rid comes from the database, never a caller-supplied filesystem path.
        return self.root / "runs" / rid / str(revision)


def public_run(row):
    return {key: row[key] for key in ("id", "request", "state", "stage", "progress", "created", "updated",
                                      "error", "revision", "validation_state", "validation_error")}
