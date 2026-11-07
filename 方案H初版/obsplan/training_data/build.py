"""Build the frozen v0.1 corpus; all teacher plans undergo independent checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from obsplan.config import Config
from obsplan.data import Dataset, load
from obsplan.famou_h_eval import FEATURES
from obsplan.problem import Problem
from obsplan.schedule import Schedule, evaluate
from obsplan.validator import validate
from obsplan.visibility import Windows, compute_windows
from .catalog import catalog_specs, dump_json, generate_catalog, make_variant, write_native_input
from .teacher import build_teacher

WINDOW_BUDGET = 12
DATASET_VERSION = "synthetic-v0.1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def event(root: Path, kind: str, **fields) -> None:
    row = dict(time_utc=datetime.now(timezone.utc).isoformat(), kind=kind, **fields)
    with (root / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def pool_windows(win: Windows, seed: int, budget: int = WINDOW_BUDGET) -> tuple[Windows, np.ndarray]:
    """Keep <=12 windows/target, stratifying by day and station before repeats.

    This is a deterministic, priority-independent candidate budget. The complete
    catalog geometry remains available for expanding the action space later.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(win.target, kind="stable")
    ends = np.r_[0, np.flatnonzero(np.diff(win.target[order])) + 1, len(order)]
    chosen = []
    for a, b in zip(ends[:-1], ends[1:]):
        idx = order[a:b]
        if len(idx) <= budget:
            chosen.extend(idx.tolist())
            continue
        shuffled = rng.permutation(idx)
        groups: dict[tuple, list] = {}
        for k in shuffled:
            key = (int(win.k0[k]) // 2880, int(win.station[k]))
            groups.setdefault(key, []).append(int(k))
        keys = list(groups)
        # Random priority-independent group order, one member per group per round.
        keys = [keys[i] for i in rng.permutation(len(keys))]
        keep = []
        while len(keep) < budget:
            for key in keys:
                if groups[key]:
                    keep.append(groups[key].pop())
                    if len(keep) == budget:
                        break
        chosen.extend(keep)
    selected = np.sort(np.asarray(chosen, dtype=np.int32))
    return Windows(win.station[selected], win.target[selected], win.k0[selected], win.k1[selected],
                   win.dark, win.n_samples, win.sgp4_failed.copy()), selected


def save_plan(path: Path, sch: Schedule, **extra) -> None:
    np.savez_compressed(path, station=sch.station.astype(np.int16), target=sch.target.astype(np.int32),
                        start_s=sch.start_s.astype(np.int32), dur_s=sch.dur_s.astype(np.int32), **extra)


def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def checked_plan(cfg: Config, ds, sch: Schedule, directory: Path, name: str):
    """One frozen repair: remove independently flagged tasks, retain everything."""
    t0 = time.perf_counter()
    report = validate(cfg, ds, sch, step_s=1.0)
    original_n = len(sch)
    keep = np.ones(original_n, dtype=bool)
    attempts = []
    if not report["legal"]:
        save_plan(directory / f"{name}_before_repair.npz", sch)
        safe = {k: v for k,v in report.items() if k not in ("per_task", "invalid_mask")}
        dump_json(directory / f"{name}_before_repair_validation.json", jsonable(safe))
        np.savez_compressed(directory / f"{name}_before_repair_geometry.npz",
                            per_task=report["per_task"], invalid_mask=report["invalid_mask"])
        attempts.append(safe)
        if not report["geometry_checked"]:
            raise ValueError(f"{name}: structural failure; refusing task-deletion repair")
        keep = ~report["invalid_mask"]
        if not keep.any():
            raise ValueError(f"{name}: no valid tasks; refusing empty plan")
        sch = Schedule(sch.station[keep], sch.target[keep], sch.start_s[keep], sch.dur_s[keep], algo=sch.algo)
        report = validate(cfg, ds, sch, step_s=1.0)
    if not (len(sch) and report["legal"] and report["geometry_checked"]):
        raise ValueError(f"{name}: final independent validation failed {report['violations']}")
    np.savez_compressed(directory / f"{name}_geometry.npz", per_task=report["per_task"],
                        invalid_mask=report["invalid_mask"])
    result = {k: v for k,v in report.items() if k not in ("per_task", "invalid_mask")}
    result.update(step_s=1.0, tasks=len(sch), original_tasks=original_n, removed_tasks=original_n-len(sch),
                  failed_attempts=jsonable(attempts), wall_s=time.perf_counter()-t0)
    return sch, jsonable(result), keep


def realign_grid_labels(result: dict, keep: np.ndarray) -> None:
    """Align published teacher labels after a retained independent rejection."""
    if keep.all():
        return
    task_ids = result["grid_task_candidate"][keep]
    plan = result["grid_plan"]
    n = len(result["candidate_window"])
    selected = np.zeros(n, bool)
    counts = np.zeros(n, np.int32)
    rank = np.full(n, -1, np.int32)
    starts = np.full(n, -1, np.int32)
    durations = np.full(n, -1, np.int32)
    for j,c in enumerate(task_ids):
        counts[c] += 1
        if not selected[c]:
            selected[c] = True
            rank[c], starts[c], durations[c] = j, plan.start_s[j], plan.dur_s[j]
    result.update(grid_task_candidate=task_ids, grid_selected=selected, grid_selected_count=counts,
                  grid_selection_rank=rank, grid_start_s=starts, grid_duration_s=durations)


def file_manifest(root: Path) -> list[dict]:
    return [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=sha256(p))
            for p in sorted(root.rglob("*")) if p.is_file()
            and p.name not in ("manifest.json", "events.jsonl") and ".cache" not in p.parts]


def _public_error(exc: BaseException, root: Path) -> str:
    text = str(exc)
    for path, label in ((root.resolve(), "<dataset>"),
                        (Path(__file__).resolve().parents[2], "<project>"),
                        (Path.home(), "<home>")):
        text = text.replace(str(path), label)
    return text


def _catalog_from_disk(root: Path, spec: dict) -> tuple[Dataset, Windows]:
    """Workers read the already-frozen catalog instead of pickling large geometry."""
    path = root / "catalogs" / spec["catalog_id"]
    with np.load(path / "targets.npz", allow_pickle=False) as table:
        priority = table["base_priority"].astype(float)
        stations = table["stations"]
        ds = Dataset(table["names"].tolist(), table["satnum"], table["line1"].tolist(),
                     table["line2"].tolist(), priority,
                     [{2.: "低", 3.: "中", 5.: "高"}[v] for v in priority],
                     stations[:, 0], stations[:, 1], stations[:, 2], [])
    return ds, Windows.load(path / "windows.npz", ds.n_stations)


def build_scene(root: Path, spec: dict, variant: int) -> dict:
    """One isolated scene worker; only the parent writes global manifest/events.

All heavy arrays stay in this process. Return small JSON-compatible metadata and
retain a failure record rather than replacing a failed scene with another seed.
    """
    import numba
    scene_id = f"{spec['catalog_id']}-v{variant}"
    path = root / "scenes" / scene_id
    path.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    old_threads = numba.get_num_threads()
    numba.set_num_threads(1)
    teacher = {}
    try:
        ds, win = _catalog_from_disk(root, spec)
        cfg = Config(start_utc=spec["start_utc"], cache_dir=str(root / ".cache"), seed=spec["seed"])
        sds, all_win, vmeta = make_variant(ds, win, spec["seed"], variant)
        selected_base = np.flatnonzero(np.isin(win.station, vmeta["base_station_indices"]))
        pool, pidx = pool_windows(all_win, spec["seed"] + variant * 1000)
        pool.save(path / "pool_windows.npz")
        np.savez_compressed(path / "pool_source_indices.npz", catalog_window_id=selected_base[pidx])
        full_variant_windows = len(all_win.k0)
        # Free the full catalog geometry before price/features create large arrays.
        del win, all_win, ds, selected_base, pidx
        write_native_input(path / "input", sds)
        scfg = cfg.with_(data_dir=str(path / "input"))
        sds = load(scfg)  # use the exact published station/TLE precision
        if sds.issues:
            raise ValueError(f"native input issues: {sds.issues}")
        pb = Problem.build(scfg, sds, pool)
        teacher = build_teacher(pb)
        original_candidates = teacher["grid_task_candidate"].copy()
        original_rows = np.arange(len(original_candidates), dtype=np.int32)
        teacher["grid_plan"], grid_report, grid_keep = checked_plan(
            scfg, sds, teacher["grid_plan"], path, "grid")
        realign_grid_labels(teacher, grid_keep)
        teacher["final_plan"], final_report, final_keep = checked_plan(
            scfg, sds, teacher["final_plan"], path, "final")
        arrays = {k:np.asarray(v) for k,v in teacher.items() if isinstance(v, np.ndarray)}
        for k in ("candidate_window", "grid_selection_rank", "grid_start_s", "grid_duration_s", "grid_task_candidate", "grid_selected_count"):
            arrays[k] = arrays[k].astype(np.int32)
        arrays["candidate_duration_cells"] = arrays["candidate_duration_cells"].astype(np.int16)
        for k in ("features", "rule_score_uncovered", "rule_score_covered"):
            arrays[k] = arrays[k].astype(np.float32)
        np.savez_compressed(path / "actions.npz", **arrays)
        save_plan(path / "grid_plan.npz", teacher["grid_plan"],
                  candidate_id=original_candidates[grid_keep], source_grid_row=original_rows[grid_keep])
        save_plan(path / "plan.npz", teacher["final_plan"],
                  candidate_id=original_candidates[final_keep], source_grid_row=original_rows[final_keep])
        dump_json(path / "validation.json", dict(grid=grid_report, final=final_report))
        configuration = asdict(scfg)
        configuration.pop("cache_dir")
        configuration["data_dir"] = "input"
        meta = dict(scene_id=scene_id, catalog_id=spec["catalog_id"], split=spec["split"],
                    path=str(path.relative_to(root)), distribution=spec["distribution"],
                    n_targets=sds.n_targets, n_stations=sds.n_stations,
                    full_variant_windows=full_variant_windows, pool_windows=len(pool.k0),
                    candidate_rows=len(arrays["features"]), grid_positive_candidates=int(arrays["grid_selected"].sum()),
                    grid_tasks=len(teacher["grid_plan"]), final_tasks=len(teacher["final_plan"]),
                    config=configuration, **vmeta,
                    teacher=jsonable(teacher["metadata"]), teacher_timings=jsonable(teacher["timings"]),
                    refinement=jsonable(teacher.get("finish_info", {})),
                    metrics=evaluate(scfg, sds, teacher["final_plan"]),
                    grid_metrics=evaluate(scfg, sds, teacher["grid_plan"]),
                    validation_wall_s=grid_report["wall_s"]+final_report["wall_s"],
                    scene_wall_s=time.perf_counter()-started,
                    scene_numba_threads=1,
                    legal=grid_report["legal"] and final_report["legal"],
                    geometry_checked=True, repairs=grid_report["removed_tasks"]+final_report["removed_tasks"],
                    task_provenance="source_grid_row indexes pre-repair teacher insertion order; candidate_id indexes actions.npz")
        meta["teacher"]["validation_status"] = "grid_and_final_independent_1s_passed"
        dump_json(path / "scene.json", meta)
        return {"status": "complete", "scene": meta}
    except BaseException as exc:
        partial = getattr(exc, "partial_result", teacher)
        for name in ("grid_plan", "final_plan"):
            if name in partial:
                save_plan(path / f"failed_{name}.npz", partial[name])
        arrays = {k:v for k,v in partial.items() if isinstance(v, np.ndarray)}
        if arrays:
            np.savez_compressed(path / "failed_actions.npz", **arrays)
        failure = dict(scene_id=scene_id, type=type(exc).__name__, error=_public_error(exc, root),
                       partial_metadata=jsonable({k:v for k,v in partial.items() if isinstance(v, (dict,str,int,float))}))
        dump_json(path / "failure.json", failure)
        return {"status": "failed", "failure": failure}
    finally:
        numba.set_num_threads(old_threads)


def _scene_results(root: Path, spec: dict, variants: list[int], scene_workers: int):
    """Collect in stable variant order; finish every submitted scene on failure."""
    if not variants:
        return
    if scene_workers == 1:
        for variant in variants:
            outcome = build_scene(root, spec, variant)
            yield outcome
            if outcome["status"] == "failed":
                return
        return
    with ProcessPoolExecutor(max_workers=scene_workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = [(variant, executor.submit(build_scene, root, spec, variant)) for variant in variants]
        for variant, future in futures:
            try:
                outcome = future.result()
            except Exception as exc:
                # Includes a process crash before the worker can record its error.
                failure = dict(scene_id=f"{spec['catalog_id']}-v{variant}",
                               type=type(exc).__name__, error=_public_error(exc, root))
                path = root / "scenes" / failure["scene_id"] / "orchestration_failure.json"
                if not path.exists():
                    dump_json(path, failure)
                outcome = {"status": "failed", "failure": failure}
            yield outcome


def _checkpoint_files(root: Path, manifest: dict, directory: Path) -> None:
    previous = {row["path"]: row for row in manifest.get("files", [])}
    for row in file_manifest(directory):
        relative = str((directory / row["path"]).relative_to(root))
        previous[relative] = {**row, "path": relative}
    manifest["files"] = [previous[key] for key in sorted(previous)]


def build(root: Path, ephemeris: Path, workers: int = 4, pilot: bool = False, resume: bool = False,
          scene_workers: int = 1):
    import numba, ortools, sgp4, skyfield
    if scene_workers not in (1, 2, 4):
        raise ValueError("scene_workers must be 1, 2 or 4")
    if root.exists() and not resume:
        raise FileExistsError(f"output already exists: {root.name}; choose a new directory")
    root.mkdir(parents=True, exist_ok=True)
    cache = root / ".cache"
    cache.mkdir(exist_ok=True)
    eph = cache / "de421.bsp"
    if not eph.exists():
        shutil.copyfile(ephemeris, eph)
    if sha256(eph) != sha256(ephemeris):
        raise ValueError("cached ephemeris does not match the requested input")
    specs = catalog_specs()
    if pilot:
        specs = [{**specs[0], "n_targets": 200}]
    manifest_path = root / "manifest.json"
    protocol = Path(__file__).resolve().parents[2] / "datasets" / DATASET_VERSION / "PROTOCOL.md"
    code_files = [dict(path=str(p.relative_to(Path(__file__).resolve().parents[2])), sha256=sha256(p))
                  for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
                  if "training_data" not in p.relative_to(Path(__file__).resolve().parents[1]).parts
                  or p.name in ("__init__.py", "catalog.py", "teacher.py", "build.py")]
    code_hash = hashlib.sha256(json.dumps(code_files, sort_keys=True).encode()).hexdigest()
    dependencies = dict(python=platform.python_version(), numpy=np.__version__,
                        sgp4=sgp4.__version__, skyfield=skyfield.__version__,
                        numba=numba.__version__, ortools=ortools.__version__)
    identity = dict(dataset_version=DATASET_VERSION, protocol_sha256=sha256(protocol),
                    pilot=pilot, workers=workers, scene_workers=scene_workers, scene_numba_threads=1,
                    window_budget=WINDOW_BUDGET, catalog_specs=specs,
                    generation_code_sha256=code_hash, dependencies=dependencies,
                    ephemeris_sha256=sha256(eph))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("resume protocol/config/code/dependency mismatch; refusing changed experiment")
        # Fail rather than silently accept changed completed products.
        for f in manifest.get("files", []):
            if sha256(root / f["path"]) != f["sha256"]:
                raise ValueError(f"completed file changed: {f['path']}")
    else:
        manifest = dict(identity=identity, dataset_version=DATASET_VERSION, feature_names=FEATURES,
                        scenes=[], catalogs=[], status="building",
                        dependencies=dependencies,
                        environment=dict(system=platform.system(), machine=platform.machine(),
                                         workers=workers, scene_workers=scene_workers,
                                         scene_numba_threads=1, cpu_count=os.cpu_count(),
                                         thread_env={k:os.environ.get(k) for k in
                                                     ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS")}),
                        ephemeris=dict(name="de421.bsp", sha256=sha256(eph)),
                        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        generation_code_files=code_files,
                        sources="original synthetic SGP4 mean elements; no external orbit data included")
        dump_json(manifest_path, manifest)
    completed = {s["scene_id"] for s in manifest["scenes"]}
    event(root, "build_start", pilot=pilot, workers=workers, scene_workers=scene_workers,
          existing_scenes=len(completed))
    for spec in specs:
        cat = spec["catalog_id"]
        cat_dir = root / "catalogs" / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        ds, table = generate_catalog(spec)
        cfg = Config(start_utc=spec["start_utc"], cache_dir=str(cache), seed=spec["seed"])
        if (cat_dir / "catalog.json").exists():
            cat_meta = json.loads((cat_dir / "catalog.json").read_text())
            win = Windows.load(cat_dir / "windows.npz", 10)
        else:
            start = time.perf_counter()
            np.savez_compressed(cat_dir / "targets.npz", **table)
            win = compute_windows(cfg, ds, workers=workers, verbose=False)
            win.save(cat_dir / "windows.npz")
            cat_meta = dict(**spec, n_windows=len(win.k0), sgp4_failed=int(win.sgp4_failed.sum()),
                            geometry_wall_s=time.perf_counter()-start,
                            family_counts={name:int((table["family"]==i).sum()) for i,name in enumerate(["LEO","MEO","GEO_like"])})
            dump_json(cat_dir / "catalog.json", cat_meta)
            event(root, "catalog_complete", **cat_meta)
        if win.sgp4_failed.any():
            raise ValueError(f"{cat}: propagation failures retained; stop publication")
        if cat not in {c["catalog_id"] for c in manifest["catalogs"]}:
            manifest["catalogs"].append(cat_meta)
        del ds, table, win  # scene workers reload frozen files; avoid retaining another full copy
        _checkpoint_files(root, manifest, cat_dir)
        dump_json(manifest_path, manifest)
        variants = [variant for variant in range(4) if f"{cat}-v{variant}" not in completed]
        failures = []
        # Catalog geometry workers have exited before this scene pool is created.
        for outcome in _scene_results(root, spec, variants, scene_workers):
            if outcome["status"] == "complete":
                meta = outcome["scene"]
                manifest["scenes"].append(meta)
                completed.add(meta["scene_id"])
                scene_id = meta["scene_id"]
                event(root, "scene_complete", scene_id=scene_id, completed=len(manifest["scenes"]),
                      candidates=meta["candidate_rows"], final_tasks=meta["final_tasks"],
                      q=meta["metrics"]["combined"], scene_wall_s=meta["scene_wall_s"], repairs=meta["repairs"])
            else:
                failure = outcome["failure"]
                failures.append(failure)
                scene_id = failure["scene_id"]
                manifest.setdefault("failures", []).append(failure)
                event(root, "scene_failed", **failure)
            _checkpoint_files(root, manifest, root / "scenes" / scene_id)
            dump_json(manifest_path, manifest)
        if failures:
            manifest["status"] = "failed"
            dump_json(manifest_path, manifest)
            raise RuntimeError(f"{cat}: {len(failures)} scene(s) failed; all submitted outcomes retained")
    manifest["status"] = "complete"
    manifest["summary"] = dict(catalogs=len(manifest["catalogs"]), scenes=len(manifest["scenes"]),
                               unique_synthetic_targets=sum(c["n_targets"] for c in manifest["catalogs"]),
                               target_instances=sum(s["n_targets"] for s in manifest["scenes"]),
                               candidate_rows=sum(s["candidate_rows"] for s in manifest["scenes"]),
                               grid_tasks=sum(s["grid_tasks"] for s in manifest["scenes"]),
                               final_tasks=sum(s["final_tasks"] for s in manifest["scenes"]),
                               repaired_tasks=sum(s["repairs"] for s in manifest["scenes"]))
    manifest["files"] = file_manifest(root)
    dump_json(manifest_path, manifest)
    event(root, "build_complete", **manifest["summary"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ephemeris", type=Path, required=True, help="existing de421.bsp (not copied into release)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scene-workers", type=int, choices=(1, 2, 4), default=1,
                        help="concurrent variants after catalog geometry finishes; each uses one Numba thread")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    build(args.output, args.ephemeris, args.workers, args.pilot, args.resume, args.scene_workers)


if __name__ == "__main__":
    main()
