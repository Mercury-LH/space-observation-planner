"""Recoverable v0.2 production: one independently sampled scene per catalog.

Workers publish immutable catalog transactions. The parent alone maintains the
atomic manifest and append-only event log. Interrupted attempts are archived,
never replaced by another random seed. All checks retain one-second Skyfield
sampling; a cache only reuses exact independent geometry work.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

from obsplan.config import Config
from obsplan.data import load
from obsplan.famou_h_eval import FEATURES
from obsplan.problem import Problem
from obsplan.schedule import Schedule, evaluate
from obsplan.visibility import compute_windows
from .build import (WINDOW_BUDGET, jsonable, pool_windows,
                    realign_grid_labels, save_plan, sha256, _public_error)
from .catalog import write_native_input
from .teacher import build_teacher

VERSION = "synthetic-v0.2"
MARKER = "COMPLETED.json"


def file_manifest(root: Path) -> list[dict]:
    """Filter relative paths, including when a transaction lives under .cache."""
    return [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=sha256(p))
            for p in sorted(root.rglob("*")) if p.is_file()
            and p.name not in ("manifest.json", "events.jsonl")
            and ".cache" not in p.relative_to(root).parts]


def event(root: Path, kind: str, **fields) -> None:
    row = dict(fields)
    row.setdefault("time_utc", datetime.now(timezone.utc).isoformat())
    row["kind"] = kind
    with (root / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
    print(json.dumps(row, ensure_ascii=False), flush=True)


def _hash_object(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@contextmanager
def exclusive(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another producer is still using {path.name}") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _job_path(root: Path, cid: str) -> Path:
    return root / ".cache" / "jobs" / cid


def _lock_path(root: Path, cid: str) -> Path:
    return root / ".cache" / "locks" / (cid + ".lock")


def _check_files(root: Path, rows: list[dict]) -> None:
    for row in rows:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe transaction inventory path")
        p = root / relative
        if (not p.is_file() or p.is_symlink() or p.stat().st_size != row["bytes"]
                or sha256(p) != row["sha256"]):
            raise ValueError(f"immutable payload mismatch: {row['path']}")


def _read_marker(path: Path, spec: dict, identity_hash: str) -> dict:
    marker = json.loads(path.read_text())
    if (marker.get("spec_sha256") != _hash_object(spec)
            or marker.get("identity_sha256") != identity_hash):
        raise ValueError(f"transaction identity mismatch: {spec['catalog_id']}")
    if marker["catalog"]["catalog_id"] != spec["catalog_id"]:
        raise ValueError("transaction catalog mismatch")
    return marker


def publish_transaction(root: Path, spec: dict, identity_hash: str) -> dict:
    """Finish either a new transaction or a crash between its two directory moves."""
    cid = spec["catalog_id"]
    job = _job_path(root, cid)
    marker = _read_marker(job / MARKER, spec, identity_hash)
    for relative in (f"catalogs/{cid}", f"scenes/{cid}-v0"):
        src, dst = job / relative, root / relative
        rows = [r for r in marker["files"] if r["path"].startswith(relative + "/")]
        if not rows:
            raise ValueError("empty transaction subtree")
        if dst.exists():
            _check_files(root, rows)
            if src.exists():
                raise FileExistsError(f"both staged and published subtree exist: {relative}")
        else:
            _check_files(job, rows)
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dst)
    _check_files(root, marker["files"])
    target = root / "catalogs" / cid / MARKER
    if target.exists():
        if json.loads(target.read_text()) != marker:
            raise ValueError("published completion marker changed")
    else:
        atomic_json(target, marker)
    # Only disposable transaction metadata remains under .cache after publication.
    shutil.rmtree(job)
    return marker


def _archive_attempt(root: Path, spec: dict, identity_hash: str, failure: dict,
                     *, keep_transaction=False) -> dict:
    cid = spec["catalog_id"]
    job = _job_path(root, cid)
    parent = root / "failures" / cid
    parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while (parent / f"attempt-{index:03d}").exists():
        index += 1
    failure = {**failure, "catalog_id": cid, "scene_id": f"{cid}-v0",
               "seed": spec["seed"], "spec_sha256": _hash_object(spec),
               "identity_sha256": identity_hash, "attempt": index,
               "time_utc": datetime.now(timezone.utc).isoformat()}
    destination = parent / f"attempt-{index:03d}"
    if keep_transaction:
        # A complete transaction remains recoverable even if a directory move or
        # completion-marker write was interrupted. Archive the error, not the job.
        destination.mkdir()
        atomic_json(destination / "failure.json", failure)
        if (job / MARKER).exists():
            shutil.copyfile(job / MARKER, destination / "transaction_marker.json")
    else:
        job.mkdir(parents=True, exist_ok=True)
        atomic_json(job / "failure.json", failure)
        os.rename(job, destination)
    return {**failure, "path": str(destination.relative_to(root))}


def recover_catalog(root: Path, spec: dict, identity_hash: str) -> tuple[dict | None, dict | None]:
    """Import completed work; preserve unfinished work before retrying the same seed."""
    cid = spec["catalog_id"]
    with exclusive(_lock_path(root, cid)):
        final_marker = root / "catalogs" / cid / MARKER
        if final_marker.exists():
            marker = _read_marker(final_marker, spec, identity_hash)
            _check_files(root, marker["files"])
            if (_job_path(root,cid)/MARKER).exists():
                marker=publish_transaction(root,spec,identity_hash)
            return marker, None
        job = _job_path(root, cid)
        if (job / MARKER).exists():
            return publish_transaction(root, spec, identity_hash), None
        if (root / "catalogs" / cid).exists() or (root / "scenes" / f"{cid}-v0").exists():
            raise ValueError(f"published subtree without recoverable completion marker: {cid}")
        if job.exists():
            failed = _archive_attempt(root, spec, identity_hash, dict(
                type="InterruptedAttempt", retry_allowed=True,
                error="Incomplete transaction recovered without an active worker; same frozen seed will be retried"))
            return None, failed
        return None, None


def _checked_plan(cfg, ds, sch, directory, name, cache):
    from obsplan.validator import validate
    started = time.perf_counter()
    original_n = len(sch)
    report = validate(cfg, ds, sch, step_s=1.0, geometry_cache=cache)
    keep, attempts = np.ones(original_n, bool), []
    if not report["legal"]:
        save_plan(directory / f"{name}_before_repair.npz", sch)
        safe = {k:v for k,v in report.items() if k not in ("per_task", "invalid_mask")}
        atomic_json(directory / f"{name}_before_repair_validation.json", jsonable(safe))
        np.savez_compressed(directory / f"{name}_before_repair_geometry.npz",
                            per_task=report["per_task"], invalid_mask=report["invalid_mask"])
        attempts.append(safe)
        if not report["geometry_checked"]:
            raise ValueError(f"{name}: structural failure; no task-deletion repair allowed")
        keep = ~report["invalid_mask"]
        if not keep.any():
            raise ValueError(f"{name}: refusing empty repaired plan")
        sch = Schedule(sch.station[keep], sch.target[keep], sch.start_s[keep], sch.dur_s[keep], algo=sch.algo)
        report = validate(cfg, ds, sch, step_s=1.0, geometry_cache=cache)
    if not (len(sch) and report["legal"] and report["geometry_checked"]):
        raise ValueError(f"{name}: independent validation did not pass")
    np.savez_compressed(directory / f"{name}_geometry.npz", per_task=report["per_task"],
                        invalid_mask=report["invalid_mask"])
    safe = {k:v for k,v in report.items() if k not in ("per_task", "invalid_mask")}
    safe.update(step_s=1.0, tasks=len(sch), original_tasks=original_n,
                removed_tasks=original_n-len(sch), failed_attempts=jsonable(attempts),
                wall_s=time.perf_counter()-started)
    return sch, jsonable(safe), keep


def catalog_job(root_string: str, spec: dict, identity_hash: str) -> dict:
    """One process owns one whole catalog; no large arrays return to the parent."""
    import numba
    from obsplan.validator import SkyfieldGeometryCache
    from .v2_sampling import generate_catalog
    root = Path(root_string)
    cid, scene_id = spec["catalog_id"], f"{spec['catalog_id']}-v0"
    job = _job_path(root, cid)
    old_threads = numba.get_num_threads()
    numba.set_num_threads(1)
    started = time.perf_counter()
    teacher = {}
    with exclusive(_lock_path(root, cid)):
        try:
            job.mkdir(parents=True, exist_ok=False)
            cat_dir, scene_dir = job / "catalogs" / cid, job / "scenes" / scene_id
            cat_dir.mkdir(parents=True)
            scene_dir.mkdir(parents=True)
            ds, table = generate_catalog(spec)
            cfg = Config(start_utc=spec["start_utc"], cache_dir=str(root / ".cache"), seed=spec["seed"])
            write_native_input(scene_dir / "input", ds)
            scfg = cfg.with_(data_dir=str(scene_dir / "input"))
            ds = load(scfg)
            if ds.issues:
                raise ValueError(f"native input issues: {ds.issues}")
            generation_wall = time.perf_counter() - started
            geometry_start = time.perf_counter()
            np.savez_compressed(cat_dir / "targets.npz", **table)
            win = compute_windows(cfg, ds, workers=1, verbose=False)
            win.save(cat_dir / "windows.npz")
            cat = dict(**spec, n_windows=len(win.k0), sgp4_failed=int(win.sgp4_failed.sum()),
                       generation_wall_s=generation_wall,
                       geometry_wall_s=time.perf_counter()-geometry_start,
                       family_counts={name:int((table["family"]==i).sum()) for i,name in enumerate(["LEO","MEO","GEO_like"])})
            atomic_json(cat_dir / "catalog.json", cat)
            if win.sgp4_failed.any():
                raise ValueError("propagation failures retained; publication stopped")
            del table
            scene_started = time.perf_counter()
            pool, indices = pool_windows(win, spec["seed"], WINDOW_BUDGET)
            full_windows = len(win.k0)
            pool.save(scene_dir / "pool_windows.npz")
            np.savez_compressed(scene_dir / "pool_source_indices.npz", catalog_window_id=indices)
            del win, indices
            pb = Problem.build(scfg, ds, pool)
            teacher = build_teacher(pb)
            original_candidates = teacher["grid_task_candidate"].copy()
            original_rows = np.arange(len(original_candidates), dtype=np.int32)
            cache = SkyfieldGeometryCache(scfg, ds)
            teacher["grid_plan"], grid_report, grid_keep = _checked_plan(
                scfg, ds, teacher["grid_plan"], scene_dir, "grid", cache)
            realign_grid_labels(teacher, grid_keep)
            teacher["final_plan"], final_report, final_keep = _checked_plan(
                scfg, ds, teacher["final_plan"], scene_dir, "final", cache)
            arrays = {k:np.asarray(v) for k,v in teacher.items() if isinstance(v,np.ndarray)}
            for key in ("candidate_window", "grid_selection_rank", "grid_start_s", "grid_duration_s", "grid_task_candidate", "grid_selected_count"):
                arrays[key] = arrays[key].astype(np.int32)
            arrays["candidate_duration_cells"] = arrays["candidate_duration_cells"].astype(np.int16)
            for key in ("features", "rule_score_uncovered", "rule_score_covered"):
                arrays[key] = arrays[key].astype(np.float32)
            np.savez_compressed(scene_dir / "actions.npz", **arrays)
            save_plan(scene_dir / "grid_plan.npz", teacher["grid_plan"],
                      candidate_id=original_candidates[grid_keep], source_grid_row=original_rows[grid_keep])
            save_plan(scene_dir / "plan.npz", teacher["final_plan"],
                      candidate_id=original_candidates[final_keep], source_grid_row=original_rows[final_keep])
            atomic_json(scene_dir / "validation.json", dict(grid=grid_report, final=final_report))
            configuration = asdict(scfg)
            configuration.pop("cache_dir")
            configuration["data_dir"] = "input"
            meta = dict(scene_id=scene_id, catalog_id=cid, split=spec["split"],
                        path=f"scenes/{scene_id}", distribution=spec["distribution"],
                        stress_type=spec.get("stress_type"), n_targets=ds.n_targets,
                        n_stations=ds.n_stations, full_variant_windows=full_windows,
                        pool_windows=len(pool.k0), candidate_rows=len(arrays["features"]),
                        grid_positive_candidates=int(arrays["grid_selected"].sum()),
                        grid_tasks=len(teacher["grid_plan"]), final_tasks=len(teacher["final_plan"]),
                        config=configuration, variant=0, variant_name="independent_scene",
                        base_station_indices=list(range(ds.n_stations)),
                        teacher=jsonable(teacher["metadata"]), teacher_timings=jsonable(teacher["timings"]),
                        refinement=jsonable(teacher.get("finish_info",{})),
                        metrics=evaluate(scfg,ds,teacher["final_plan"]),
                        grid_metrics=evaluate(scfg,ds,teacher["grid_plan"]),
                        validation_wall_s=grid_report["wall_s"]+final_report["wall_s"],
                        scene_wall_s=time.perf_counter()-scene_started,
                        catalog_wall_s=time.perf_counter()-started,
                        independent_geometry_cache=jsonable(cache.stats()),
                        scene_numba_threads=1, legal=True, geometry_checked=True,
                        repairs=grid_report["removed_tasks"]+final_report["removed_tasks"],
                        task_provenance="source_grid_row indexes pre-repair teacher insertion order; candidate_id indexes actions.npz")
            meta["teacher"]["validation_status"] = "grid_and_final_independent_1s_passed"
            atomic_json(scene_dir / "scene.json", meta)
            marker = dict(identity_sha256=identity_hash, spec_sha256=_hash_object(spec),
                          catalog=cat, scene=meta, files=file_manifest(job))
            atomic_json(job / MARKER, marker)
            published = publish_transaction(root, spec, identity_hash)
            return dict(status="complete", marker=published)
        except BaseException as exc:
            publishing = (job / MARKER).exists() or (root / "catalogs" / cid / MARKER).exists()
            interrupted = publishing or isinstance(exc, (KeyboardInterrupt, SystemExit))
            if publishing:
                failure = _archive_attempt(root,spec,identity_hash,dict(
                    type=type(exc).__name__,error=_public_error(exc,root),
                    retry_allowed=True,phase="transaction_publication"),keep_transaction=True)
                return dict(status="interrupted",failure=failure)
            partial = getattr(exc, "partial_result", teacher)
            scene_dir = job / "scenes" / scene_id
            scene_dir.mkdir(parents=True, exist_ok=True)
            for name in ("grid_plan", "final_plan"):
                if name in partial:
                    save_plan(scene_dir / f"failed_{name}.npz", partial[name])
            arrays = {k:v for k,v in partial.items() if isinstance(v,np.ndarray)}
            if arrays:
                np.savez_compressed(scene_dir / "failed_actions.npz", **arrays)
            failure = _archive_attempt(root, spec, identity_hash, dict(
                type=type(exc).__name__, error=_public_error(exc, root),
                retry_allowed=interrupted, partial_metadata=jsonable({k:v for k,v in partial.items() if isinstance(v,(dict,str,int,float))})))
            return dict(status="interrupted" if interrupted else "failed", failure=failure)
        finally:
            numba.set_num_threads(old_threads)


def _identity(root: Path, ephemeris: Path, specs: list, workers: int, pilot: bool,
              v1_root: Path) -> tuple[dict,list,dict]:
    import numba, ortools, sgp4, skyfield
    from .v2_sampling import specs_sha256
    project = Path(__file__).resolve().parents[2]
    names = {"__init__.py","catalog.py","build.py","teacher.py","v2_build.py","v2_sampling.py"}
    code = [dict(path=str(p.relative_to(project)),sha256=sha256(p))
            for p in sorted((project / "obsplan").rglob("*.py"))
            if "training_data" not in p.relative_to(project / "obsplan").parts or p.name in names]
    dependencies = dict(python=platform.python_version(),numpy=np.__version__,numba=numba.__version__,
                        ortools=ortools.__version__,sgp4=sgp4.__version__,skyfield=skyfield.__version__)
    protocols = {str(p.relative_to(project)):sha256(p)
                 for p in sorted((project / "datasets" / VERSION).glob("PROTOCOL*.md"))}
    if not protocols or not (project / "datasets" / VERSION / "PROTOCOL.md").is_file():
        raise ValueError("v0.2 production protocol has not been frozen")
    identity = dict(dataset_version=VERSION,pilot=pilot,workers=workers,catalog_workers=workers,
                    window_workers=1,scene_numba_threads=1,window_budget=WINDOW_BUDGET,
                    catalog_specs=specs,specs_sha256=specs_sha256(specs),
                    generation_code_sha256=_hash_object(code),protocols=protocols,
                    dependencies=dependencies,ephemeris_sha256=sha256(ephemeris),
                    v1_manifest_sha256=sha256(v1_root / "manifest.json"),
                    geometry_mode="independent_skyfield_exact_time_cache")
    return identity,code,dependencies


def _inventory_subtree(root: Path, manifest: dict, directory: Path) -> None:
    rows = {r["path"]:r for r in manifest.get("files",[])}
    if directory.exists():
        for row in file_manifest(directory):
            name = str((directory / row["path"]).relative_to(root))
            rows[name] = {**row,"path":name}
    manifest["files"] = [rows[k] for k in sorted(rows)]


def _register(root: Path, manifest: dict, marker: dict, *, recovered=False) -> None:
    cid = marker["catalog"]["catalog_id"]
    if cid in {c["catalog_id"] for c in manifest["catalogs"]}:
        raise ValueError(f"duplicate transaction registration: {cid}")
    manifest["catalogs"].append(marker["catalog"])
    manifest["scenes"].append(marker["scene"])
    manifest["catalogs"].sort(key=lambda x:x["catalog_id"])
    manifest["scenes"].sort(key=lambda x:x["scene_id"])
    for d in (root/"catalogs"/cid,root/"scenes"/f"{cid}-v0",root/"failures"/cid):
        _inventory_subtree(root,manifest,d)
    atomic_json(root/"manifest.json",manifest)
    s=marker["scene"]
    event(root,"catalog_scene_complete",catalog_id=cid,scene_id=s["scene_id"],
          completed=len(manifest["scenes"]),candidates=s["candidate_rows"],final_tasks=s["final_tasks"],
          q=s["metrics"]["combined"],catalog_wall_s=s["catalog_wall_s"],repairs=s["repairs"],recovered=recovered)


def _summary(manifest: dict) -> dict:
    c,s=manifest["catalogs"],manifest["scenes"]
    return dict(catalogs=len(c),scenes=len(s),unique_synthetic_targets=sum(x["n_targets"] for x in c),
                target_instances=sum(x["n_targets"] for x in s),candidate_rows=sum(x["candidate_rows"] for x in s),
                grid_tasks=sum(x["grid_tasks"] for x in s),final_tasks=sum(x["final_tasks"] for x in s),
                repaired_tasks=sum(x["repairs"] for x in s))


def recover_failure_ledger(root: Path, manifest: dict, specs: list, identity_hash: str) -> bool:
    """Import a worker's archived error if the parent died before recording it."""
    known={row["path"] for row in manifest.get("failures",[]) if "path" in row}
    hard_failure=any(not row.get("retry_allowed",False) for row in manifest.get("failures",[]))
    for spec in specs:
        cid=spec["catalog_id"]
        for path in sorted((root/"failures"/cid).glob("attempt-*/failure.json")):
            relative=str(path.parent.relative_to(root))
            if relative in known:
                continue
            record=json.loads(path.read_text())
            if (record.get("identity_sha256")!=identity_hash
                    or record.get("spec_sha256")!=_hash_object(spec)
                    or record.get("catalog_id")!=cid):
                raise ValueError("retained failure identity mismatch")
            record={**record,"path":relative}
            manifest.setdefault("failures",[]).append(record)
            known.add(relative)
            hard_failure=hard_failure or not record.get("retry_allowed",False)
            _inventory_subtree(root,manifest,path.parent)
            atomic_json(root/"manifest.json",manifest)
            event(root,"attempt_failure_recovered",**record)
    return hard_failure


def prepare_ephemeris(cache: Path, source: Path) -> Path:
    destination=cache/"de421.bsp"
    cache.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if sha256(destination)!=sha256(source):
            raise ValueError("cached ephemeris changed")
        return destination
    temporary=cache/"de421.bsp.copying"
    if temporary.exists():
        i=1
        while (cache/f"interrupted-ephemeris-copy-{i:03d}").exists():
            i+=1
        temporary.rename(cache/f"interrupted-ephemeris-copy-{i:03d}")
    with source.open("rb") as inp, temporary.open("xb") as out:
        shutil.copyfileobj(inp,out)
        out.flush()
        os.fsync(out.fileno())
    if sha256(temporary)!=sha256(source):
        raise ValueError("ephemeris copy failed checksum verification")
    os.rename(temporary,destination)
    return destination


def build(root: Path, ephemeris: Path, v1_root: Path, *, workers=4, resume=False,
          catalog_ids: list[str] | None=None):
    from .v2_sampling import specs as make_specs, sampling_report
    root,ephemeris,v1_root=root.resolve(),ephemeris.resolve(),v1_root.resolve()
    if workers not in (1,2,3,4):
        raise ValueError("catalog workers must be 1..4")
    if root.exists() and not resume:
        raise FileExistsError("choose a new output directory or explicit --resume")
    all_specs=make_specs()
    selected=[s for s in all_specs if catalog_ids is None or s["catalog_id"] in catalog_ids]
    if not selected or (catalog_ids is not None and len(selected)!=len(set(catalog_ids))):
        raise ValueError("unknown or empty pilot catalog selection")
    pilot=catalog_ids is not None
    sampling=sampling_report(selected)
    if not pilot and sampling.get("passed") is not True:
        raise ValueError("frozen sampling design failed before production")
    root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/".cache"/"build.lock"):
        cache=root/".cache"
        prepare_ephemeris(cache,ephemeris)
        identity,code,deps=_identity(root,ephemeris,selected,workers,pilot,v1_root)
        ih=_hash_object(identity)
        path=root/"manifest.json"
        if path.exists():
            m=json.loads(path.read_text())
            if m["identity"]!=identity:
                raise ValueError("resume code/spec/protocol/dependency mismatch")
            _check_files(root,m.get("files",[]))
            if m["status"]=="complete":
                if file_manifest(root)!=m.get("files",[]):
                    raise ValueError("completed dataset file inventory changed")
                print(json.dumps({"already_complete":True,**m["summary"]}),flush=True)
                return
            if m["status"]=="failed":
                raise ValueError("recorded production failure needs diagnosis; no automatic changed-seed retry")
        else:
            m=dict(dataset_version=VERSION,identity=identity,feature_names=FEATURES,
                   status="building",catalogs=[],scenes=[],files=[],failures=[],
                   sampling_report=sampling,dependencies=deps,
                   generation_code_files=code,
                   base_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                   ephemeris=dict(name="de421.bsp",sha256=sha256(ephemeris)),
                   environment=dict(system=platform.system(),machine=platform.machine(),cpu_count=os.cpu_count(),
                                    catalog_workers=workers,window_workers=1,scene_numba_threads=1,
                                    thread_env={k:os.environ.get(k) for k in
                                                ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMBA_NUM_THREADS")}),
                   sources="original synthetic SGP4 mean elements; no external orbit catalog included")
            atomic_json(path,m)
        event(root,"build_start",pilot=pilot,catalog_workers=workers,existing_scenes=len(m["scenes"]))
        if recover_failure_ledger(root,m,selected,ih):
            m["status"]="failed"
            atomic_json(path,m)
            raise ValueError("unregistered physical/algorithm failure recovered; automatic retry refused")
        done={c["catalog_id"] for c in m["catalogs"]}
        pending=[]
        for spec in selected:
            if spec["catalog_id"] in done:
                continue
            marker,interrupted=recover_catalog(root,spec,ih)
            if interrupted:
                m["failures"].append(interrupted)
                _inventory_subtree(root,m,root/"failures"/spec["catalog_id"])
                atomic_json(path,m)
                event(root,"attempt_interrupted",**interrupted)
            if marker:
                _register(root,m,marker,recovered=True)
            else:
                pending.append(spec)
        failed=False
        hard_failure=False
        iterator=iter(pending)
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            active={}
            def fill():
                while len(active)<workers and not failed:
                    spec=next(iterator,None)
                    if spec is None:
                        break
                    active[pool.submit(catalog_job,str(root),spec,ih)]=spec
            fill()
            while active:
                ready,_=wait(active,return_when=FIRST_COMPLETED)
                for future in ready:
                    spec=active.pop(future)
                    try:
                        outcome=future.result()
                    except Exception as exc:
                        # Preserve the worker's staging directory; explicit resume will archive it.
                        with exclusive(_lock_path(root,spec["catalog_id"])):
                            publishing=(_job_path(root,spec["catalog_id"])/MARKER).exists() or (root/"catalogs"/spec["catalog_id"]/MARKER).exists()
                            failure=_archive_attempt(root,spec,ih,dict(
                                type=type(exc).__name__,error=_public_error(exc,root),retry_allowed=True,
                                phase="worker_process"),keep_transaction=publishing)
                        outcome=dict(status="interrupted",failure=failure)
                    if outcome["status"]=="complete":
                        _register(root,m,outcome["marker"])
                    else:
                        failed=True
                        hard_failure=hard_failure or outcome["status"]=="failed"
                        m["failures"].append(outcome["failure"])
                        _inventory_subtree(root,m,root/"failures"/spec["catalog_id"])
                        event(root,"catalog_failed",**outcome["failure"])
                        atomic_json(path,m)
                fill()
        if failed:
            m["status"]="failed" if hard_failure else "interrupted"
            atomic_json(path,m)
            raise RuntimeError("generation failed; original attempts retained and no replacement seeds used")
        if len(m["catalogs"])!=len(selected) or len(m["scenes"])!=len(selected):
            raise RuntimeError("incomplete fixed catalog set")
        m["summary"]=_summary(m)
        m["files"]=file_manifest(root)
        # Final event is committed before status=complete, avoiding a complete-but-growing log.
        event(root,"build_complete",**m["summary"])
        m["status"]="complete"
        atomic_json(path,m)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--ephemeris",type=Path,required=True)
    p.add_argument("--v1-root",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--catalog-ids",nargs="+",help="explicit calibration subset; always marked pilot and cannot be released")
    a=p.parse_args()
    build(a.output,a.ephemeris,a.v1_root,workers=a.workers,resume=a.resume,catalog_ids=a.catalog_ids)


if __name__=="__main__":
    main()
