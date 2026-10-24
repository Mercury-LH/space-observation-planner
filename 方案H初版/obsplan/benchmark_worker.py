"""One-process benchmark worker. The parent owns timeouts and final wall time."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

from .benchmark import (ROOT, SCHEMA_VERSION, check_config, digest, environment_identity,
                        public_config, read_power_profile, resolve_callable, source_identity, write_json)


def dataset_hash(ds) -> str:
    """Hash effective inputs, including TLEs, priority assignments and stations."""
    return digest({"satnum": ds.satnum.tolist(), "line1": ds.line1, "line2": ds.line2,
                   "names": ds.names, "priority": ds.priority.tolist(),
                   "priority_label": ds.priority_label, "st_lon": ds.st_lon.tolist(),
                   "st_lat": ds.st_lat.tolist(), "st_alt_m": ds.st_alt_m.tolist()})


def check_schedule(schedule, ds):
    """Protect indexing/scoring and guarantee export_csv preserves exact seconds."""
    import numpy as np
    from .schedule import Schedule
    if not isinstance(schedule, Schedule):
        raise TypeError("Solver must return a Schedule")
    arrays = [np.asarray(getattr(schedule, field)) for field in ("station", "target", "start_s", "dur_s")]
    if any(a.ndim != 1 for a in arrays) or len({len(a) for a in arrays}) != 1:
        raise ValueError("Schedule arrays must be equal-length vectors")
    for a in arrays:
        if not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a) or not np.isfinite(a).all():
            raise ValueError("Schedule arrays must be finite real numbers")
        if not (a == np.floor(a)).all():
            raise ValueError("Schedule IDs and seconds must be integers")
        if ((a < -(2**53)) | (a > 2**53)).any():
            raise ValueError("Schedule values exceed exact integer range")
    station, target, start, duration = [a.astype(np.int64) for a in arrays]
    if ((station < 0) | (station >= ds.n_stations)).any() or ((target < 0) | (target >= ds.n_targets)).any():
        raise ValueError("Schedule IDs outside the dataset")
    return Schedule(station, target, start, duration, algo="benchmark")


def _stage_times(info):
    result = {}
    for key, value in info.get("stage_timings_s", {}).items():
        if isinstance(key, str) and "/" not in key and "\\" not in key:
            if isinstance(value, (float, int)) and math.isfinite(value) and value >= 0:
                result[key] = float(value)
    return result


def _public_spec(value):
    """Publish reproducibility parameters, stripping credentials and local paths."""
    if isinstance(value, dict):
        return {k: "<redacted>" if any(word in k.lower() for word in ("key", "secret", "token", "password"))
                else _public_spec(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_public_spec(v) for v in value]
    if isinstance(value, str) and (value.startswith(("/", "~", "file:", "http:", "https:"))
                                   or (len(value) > 2 and value[1] == ":")):
        return "<path-or-url>"
    return value


def work(request_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    directory = request_path.parent
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    phase = "preprocessing"
    phase_started = started
    result.update(status="running", fresh_solve=False, rankable=False,
                  timing_s={"preprocessing": 0.0, "solver": 0.0, "validation": 0.0, "total": 0.0})

    def save():
        result["timing_s"]["total"] = time.perf_counter() - started
        result["active_phase"] = phase
        result["active_phase_started_monotonic"] = time.monotonic() - (time.perf_counter() - phase_started)
        write_json(directory / "result.json", result)

    def progress(event):
        # Progress is local diagnostic evidence; never copied into public JSON.
        print(json.dumps(event, default=str, ensure_ascii=False), flush=True)

    def check_power_profile(checkpoint):
        profile = read_power_profile()
        result["power_profile_checks"][checkpoint] = profile
        if profile != result["power_profile_checks"]["start"]:
            result["environment_changed"] = True

    try:
        import numpy as np
        from .config import Config
        from .data import load
        from .problem import Problem
        from .refine import near_windows
        from .schedule import evaluate, export_csv
        from .validator import validate
        from .visibility import get_windows

        cfg = Config(**request["config"])
        check_config(cfg)
        ds = load(cfg)
        if ds.n_targets == 0 or ds.n_stations == 0 or float(ds.priority.sum()) <= 0:
            raise ValueError("Empty or zero-priority dataset")
        input_hash = dataset_hash(ds)
        entrypoint = request["entrypoint"]
        plugin = None if entrypoint == "obsplan.portfolio:solve" else entrypoint
        provenance = source_identity(plugin)
        environment = environment_identity()
        result["power_profile_checks"] = {"start": environment["power_profile"]}
        result["environment_changed"] = False
        config = public_config(cfg)
        fixed_config = {k: v for k, v in config.items() if k != "seed"}
        cache_key = digest({"input": input_hash, "config": fixed_config,
                            "dependencies": environment["dependencies"],
                            "geometry": {k: v for k, v in provenance["source_files"].items()
                                         if k.startswith("obsplan/") and Path(k).name in
                                         ("visibility.py", "geometry.py", "config.py", "data.py", "refine.py")}})
        cache = Path(cfg.cache_dir) / "benchmark" / cache_key
        cfg = cfg.with_(cache_dir=str(cache))
        cache_file = cache / f"windows_{cfg.scenario_key()}.npz"
        near_file = cache / f"near_{cfg.scenario_key()}.npz"
        # Numba's native disk cache is shared, not equivalent to an in-process
        # warmup. Record the condition; do not subtract compilation from time.
        numba_cached = any((ROOT / "obsplan").rglob("*.nbc"))
        result.update(provenance={**provenance, "input_hash": input_hash,
                                  "config_hash": digest(fixed_config), "environment": environment},
                      config=config, entrypoint=entrypoint,
                      solver_spec=_public_spec(request["spec"]),
                      solver_spec_hash=digest({k: v for k, v in request["spec"].items() if k != "seed"}),
                      cache={"windows": "warm" if cache_file.exists() else "cold",
                             "near_windows": "warm" if near_file.exists() else "cold",
                             "namespace": cache_key, "numba_disk_cache_present": numba_cached,
                             "in_process_warmup": False, "jit_cost_included_in_solver": True},
                      input_summary={"targets": ds.n_targets, "stations": ds.n_stations,
                                     "issues_count": len(ds.issues)},
                      comparison_key=digest({"input": input_hash,
                                             "config": fixed_config, "source": provenance["source_hash"],
                                             "environment": environment, "workers": request["workers"]}))
        save()
        if ds.issues:
            # Raw data diagnostics can include target names: local log only.
            print("Input issues:", ds.issues, flush=True)
            result["status"] = "input_issues"
            return 1
        windows = get_windows(cfg, ds)
        result["input_summary"]["sgp4_failed_targets"] = int(np.asarray(windows.sgp4_failed).sum())
        problem = Problem.build(cfg, ds, windows)
        near_windows(problem)  # Shared geometry preprocessing, before every solver's stopwatch.
        solver = resolve_callable(entrypoint)
        result["timing_s"][phase] = time.perf_counter() - phase_started
        phase, phase_started = "solver", time.perf_counter()
        result["fresh_solve"] = True
        save()
        response = solver(problem, request["spec"], progress=progress)
        result["timing_s"][phase] = time.perf_counter() - phase_started
        phase, phase_started = "validation", time.perf_counter()
        check_power_profile("after_solver")  # Outside the complete solver stopwatch.
        save()
        if not isinstance(response, tuple) or len(response) != 2 or not isinstance(response[1], dict):
            raise TypeError("Solver must return (Schedule, info dict)")
        schedule, info = response
        schedule = check_schedule(schedule, ds)
        result["stage_timings_s"] = _stage_times(info)
        famou = info.get("famou", {})
        result["famou"] = {"mode": famou.get("mode", "unspecified"),
                           "runtime_cloud_calls": famou.get("runtime_cloud_calls"),
                           "invoked": bool(famou.get("invoked", famou.get("runtime_cloud_calls", 0)))}
        result["actual_solver_config"] = _public_spec(info.get("actual_config", request["spec"]))
        result["source_metrics"] = {name: {key: values[key] for key in
                                            ("covered", "score", "combined", "tasks") if key in values}
                                    for name, values in info.get("sources", {}).items()}
        result["stop_reason"] = _public_spec(info.get("stop_reason"))
        # Do not copy arbitrary plugin info into shareable output: it can contain
        # target rows, secrets or local paths. Persist it only in the local log.
        print("Solver info:", json.dumps(info, default=str, ensure_ascii=False), flush=True)
        export_csv(cfg, ds, schedule, directory / "plan.csv")
        result["plan_sha256"] = hashlib.sha256((directory / "plan.csv").read_bytes()).hexdigest()
        if request["validate"]:
            checked = validate(cfg, ds, schedule, step_s=1.0)
            result["validation"] = {key: checked.get(key) for key in
                                    ("legal", "violations", "geometry_checked", "min_elev_margin",
                                     "max_elev_margin", "sun_margin")}
            result["validation"]["step_s"] = 1.0
            result["status"] = "valid" if checked["legal"] else "invalid"
            result["rankable"] = bool(checked["legal"] and checked.get("geometry_checked") is True)
        else:
            result["validation"] = {"legal": None, "step_s": None, "geometry_checked": False}
            result["status"] = "unvalidated"
        if result["status"] in ("valid", "unvalidated"):
            metrics = evaluate(cfg, ds, schedule)
            result["metrics"] = {key: metrics[key] for key in
                                 ("tasks", "covered", "coverage_rate", "score", "combined",
                                  "repeat_tasks", "mean_dur_s")}
        check_power_profile("after_validation")
        after = source_identity(plugin)
        if any(after[key] != provenance[key] for key in ("source_hash", "plugin_source_hash")):
            result.update(status="source_changed", rankable=False)
        if result["environment_changed"]:
            result["status_before_environment_change"] = result["status"]
            result.update(status="environment_changed", rankable=False)
        return 0 if result["status"] in ("valid", "unvalidated") else 1
    except Exception as exc:
        result.update(status="error", rankable=False, error_type=type(exc).__name__, error_phase=phase)
        traceback.print_exc()
        return 1
    finally:
        result["timing_s"][phase] = time.perf_counter() - phase_started
        save()
        result.pop("active_phase", None)
        result.pop("active_phase_started_monotonic", None)
        write_json(directory / "result.json", result)


if __name__ == "__main__":
    raise SystemExit(work(Path(sys.argv[1])))
