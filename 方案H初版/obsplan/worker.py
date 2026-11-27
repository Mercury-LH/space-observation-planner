"""单计算进程。与 HTTP 服务分离；崩溃后的任务与完整产物可追踪。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from .app_backend import ALGOS, Run, load_problem, run_planning
from .config import Config, SCHEME_DIR
from .data import load
from .jobs import Store
from .schedule import export_csv
from .submission import json_bytes, parse_json, records, score, submission_problem, to_schedule

LOG = logging.getLogger("obsplan.worker")


def dataset_fingerprint(ds):
    digest = hashlib.sha256()
    for line in ds.line1 + ds.line2 + ds.names + ds.priority_label:
        digest.update(line.encode("utf-8") + b"\n")
    for values in (ds.satnum, ds.priority, ds.st_lon, ds.st_lat, ds.st_alt_m):
        digest.update(values.tobytes())
    return digest.hexdigest()


def write_result(folder, run, ds, request, report=None, check=None):
    cfg = Config.from_saved(run.cfg)
    sch = run.schedule()
    rows = records(cfg, ds, sch)
    payload = json_bytes(rows)
    # The file to be downloaded is parsed again, with no skipped records.
    back = to_schedule(parse_json(payload), cfg, ds)
    expected = sorted(zip(sch.station, sch.target, sch.start_s, sch.dur_s))
    actual = sorted(zip(back.station, back.target, back.start_s, back.dur_s))
    if expected != actual:
        raise ValueError("提交文件重新解析与本次计划不一致")
    (folder / "submission.json").write_bytes(payload)
    export_csv(cfg, ds, sch, folder / "plan.csv", check)
    public_cfg = {k: v for k, v in run.cfg.items() if k not in ("data_dir", "cache_dir")}
    summary = {"run_id": run.run_id, "algo": ALGOS[run.algo], "config": public_cfg,
               "request": request, "metrics": run.metrics, "runtime_s": run.runtime_s,
               "notes": run.notes, "validation": run.validation,
               "submission_sha256": hashlib.sha256(payload).hexdigest(),
               "v3": None if report is None else report["summary"],
               "forecast_source": None if report is None else report["forecast_source"]}
    (folder / "summary.json").write_bytes(json_bytes(summary))
    with (folder / "plan.csv").open(encoding="utf-8-sig", newline="") as file:
        tasks = list(csv.DictReader(file))
    if report is not None:
        for task, verdict in zip(tasks, report["tasks"]):
            task["validation_status"] = verdict["status"]
            task["validation_reasons"] = verdict["reasons"]
            task["v3_score"] = verdict["score"]
        (folder / "validation.json").write_bytes(json_bytes(report))
    result = {"summary": summary, "tasks": tasks, "stationIds": [str(s) for s in request["stations"]],
              "outages": [], "startUtc": request["date"] + "T00:00:00Z"}
    (folder / "result.json").write_bytes(json_bytes(result))


def publish(store, rid, revision, build):
    parent = store.artifact_dir(rid, revision).parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=parent))
    destination = store.artifact_dir(rid, revision)
    try:
        build(temporary)
        # Ensure bytes are durable before the database exposes the revision.
        for path in temporary.iterdir():
            with path.open("rb") as file:
                os.fsync(file.fileno())
        if destination.exists():
            if store.get(rid)["revision"] >= revision:
                raise FileExistsError("已发布的结果版本不能覆盖")
            # A crash can occur after rename but before the DB transaction. Preserve
            # that orphan and allow a failed validation to be retried safely.
            destination.rename(parent / f".orphan-{revision}-{uuid.uuid4().hex}")
        os.replace(temporary, destination)
        fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def plan(store, row):
    request, rid = row["request"], row["id"]
    started = time.monotonic()
    cfg = Config(start_utc=request["date"] + "T00:00:00", days=request["days"])
    ds = load(cfg)
    fingerprint = dataset_fingerprint(ds)
    # Cache namespace includes all input data and rule parameters, not just date.
    rule_key = hashlib.sha256(cfg.to_json().encode()).hexdigest()[:16]
    cfg = cfg.with_(cache_dir=str(Path(cfg.cache_dir) / "api-v3" / fingerprint[:16] / rule_key))
    store.progress(rid, .05, "准备数据与可见窗口")
    ds, pb, timing = load_problem(cfg)
    pb = submission_problem(pb, [s - 1 for s in request["stations"]])
    store.progress(rid, .15, "计算观测计划")
    def progress(fraction, message):
        stage = "按秒调整观测时间" if fraction >= .97 else "优化观测计划" if fraction >= .9 else "分配观测时段"
        store.progress(rid, .15 + .75 * fraction, stage)

    run = run_planning(pb, request["algo"], iters=150, fast=True, refine_seconds=True,
                       full_problem=False, chain_minutes=0,
                       progress=progress)
    run.run_id = rid
    run.notes.update({"input_sha256": fingerprint, "preparation": timing,
                      "eligible_targets": int(((ds.satnum >= 10000) & (ds.satnum <= 99999)).sum()),
                      "excluded_id_below_10000": int((ds.satnum < 10000).sum()),
                      "excluded_id_above_99999": int((ds.satnum > 99999).sum()),
                      "submission_mode": "v3-five-digit-integers",
                      "full_problem_supplement": False,
                      "total_seconds": round(time.monotonic() - started, 3)})
    store.progress(rid, .95, "导出并重新解析提交文件")

    def build(folder):
        run.save(folder)
        write_result(folder, run, ds, request)

    publish(store, rid, 1, build)
    return 1


def validate_job(store, row):
    from .forecast import local_forecasts
    from .validator import validate
    rid, rev = row["id"], row["revision"]
    folder = store.artifact_dir(rid, rev)
    run = Run.load(folder / f"{rid}.npz")
    cfg = Config.from_saved(run.cfg)
    ds = load(cfg)
    if dataset_fingerprint(ds) != run.notes["input_sha256"]:
        raise ValueError("数据目录已改变，不能用另一份输入校验原运行；请恢复数据或新建规划")
    started = time.monotonic()
    check = validate(cfg, ds, run.schedule())
    run.validation = {"legal": check["legal"], "violations": check["violations"],
                      "task_violations": check["task_violations"], "seconds": round(time.monotonic() - started, 3)}
    run.per_task = check["per_task"].astype(np.float32)
    rows = parse_json((folder / "submission.json").read_bytes())
    forecasts, source = local_forecasts(cfg, ds, rows)
    report = score(rows, ds, forecasts, source=source)
    report.update(run_id=rid, geometry=run.validation, seconds=round(time.monotonic() - started, 3))

    def build(out):
        run.save(out)
        write_result(out, run, ds, row["request"], report, check)
        (out / "forecast.json").write_bytes(json_bytes(forecasts))

    publish(store, rid, rev + 1, build)
    return rev + 1, "passed" if check["legal"] and report["summary"]["clean"] else "issues"


def work_one(store):
    job = store.claim()
    if not job:
        return False
    try:
        row = store.get(job["run_id"])
        if job["kind"] == "plan":
            store.finish(job, revision=plan(store, row))
        else:
            revision, verdict = validate_job(store, row)
            store.finish(job, revision=revision, verdict=verdict)
    except Exception as exc:
        LOG.exception("job %s (%s) failed", job["id"], job["run_id"])
        store.finish(job, error=f"{type(exc).__name__}: {exc}")
    return True


def main():
    import fcntl
    parser = argparse.ArgumentParser(description="单实例观测规划队列 worker（Linux/macOS）")
    parser.add_argument("--store", default=os.environ.get("OBSPLAN_STORE", str(SCHEME_DIR / "outputs" / "service")))
    parser.add_argument("--once", action="store_true", help="处理一项任务后退出")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    store = Store(args.store)
    with (store.root / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("此存储目录已有计算 worker，拒绝重复启动") from None
        store.recover()
        stopped = threading.Event()

        def heartbeat():
            while not stopped.is_set():
                store.heartbeat()
                stopped.wait(5)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            while True:
                active = work_one(store)
                if args.once:
                    break
                if not active:
                    time.sleep(.5)
        finally:
            stopped.set()
            thread.join(timeout=6)


if __name__ == "__main__":
    main()
