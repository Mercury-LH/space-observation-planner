"""Reproducible local commands; source data never leave the machine."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time
from typing import Any

from . import __version__
from .data import DataAuditError, Dataset, load_dataset
from .scheduling import alns_schedule, greedy_schedule, objective_metrics, observation_score
from .types import Observation, Rules
from .validation import validate_plan


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_KEYS = {
    "name", "description", "data_root", "start_utc", "days", "horizon_s",
    "target_limit", "target_ids", "selection_seed", "station_ids", "rules",
    "windows", "seeds", "greedy_strategy", "greedy_seed", "greedy_time_limit_s",
    "alns_iterations", "alns_time_limit_s", "output_dir", "probe",
    "validation", "development_assumptions",
}
WINDOW_KEYS = {
    "scan_step_s", "target_chunk", "time_chunk_s", "root_tolerance_s", "cache_dir", "workers",
}


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (datetime, Path)):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_value, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def _nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and nonnegative")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be finite and nonnegative") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return value


def parse_start_utc(value: str) -> datetime:
    """Require an explicit timezone; normalize to UTC without using host time."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("start_utc must include Z or an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def load_config(path: Path, *, require_start: bool = True) -> dict[str, Any]:
    """Resolve relative paths against the repository root, not the shell cwd."""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    unknown = config.keys() - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    if "data_root" not in config:
        raise ValueError("Config requires data_root")
    config["data_root"] = str(_resolve_path(config["data_root"]))
    if require_start and "start_utc" not in config:
        raise ValueError("Explicit start_utc is required; no date is inferred from today's date")
    if "start_utc" in config:
        config["start_utc"] = parse_start_utc(config["start_utc"]).isoformat()
    if "days" in config and "horizon_s" in config:
        raise ValueError("Specify either days or horizon_s, not both")
    if require_start and "days" not in config and "horizon_s" not in config:
        raise ValueError("Explicit days or horizon_s is required")
    if "days" in config:
        config["horizon_s"] = _positive_number(config.pop("days"), "days") * 86400
    if "horizon_s" in config:
        config["horizon_s"] = _positive_number(config["horizon_s"], "horizon_s")
    if "target_limit" in config and "target_ids" in config:
        raise ValueError("Specify either target_limit or target_ids, not both")
    if "target_limit" in config and (
        isinstance(config["target_limit"], bool)
        or not isinstance(config["target_limit"], int)
        or config["target_limit"] < 1
    ):
        raise ValueError("target_limit must be a positive integer")
    if "target_ids" in config and (
        not isinstance(config["target_ids"], list)
        or not config["target_ids"]
        or any(isinstance(t, bool) or not isinstance(t, int) for t in config["target_ids"])
        or len(set(config["target_ids"])) != len(config["target_ids"])
    ):
        raise ValueError("target_ids must be a nonempty list of unique integers")
    if "station_ids" in config:
        if not isinstance(config["station_ids"], list) or not config["station_ids"]:
            raise ValueError("station_ids must be a nonempty list")
        config["station_ids"] = [str(value) for value in config["station_ids"]]
        if len(set(config["station_ids"])) != len(config["station_ids"]):
            raise ValueError("station_ids must be unique")
    config.setdefault("selection_seed", 0)
    config.setdefault("seeds", [0, 1, 2])
    config.setdefault("greedy_strategy", "portfolio")
    config.setdefault("greedy_seed", 0)
    config.setdefault("greedy_time_limit_s", None)
    config.setdefault("alns_iterations", 100)
    config.setdefault("alns_time_limit_s", None)
    for name in ("selection_seed", "greedy_seed", "alns_iterations"):
        if name in config and (
            isinstance(config[name], bool) or not isinstance(config[name], int)
            or (name == "alns_iterations" and config[name] < 0)
        ):
            raise ValueError(f"{name} must be an integer (iterations must be nonnegative)")
    for name in ("greedy_time_limit_s", "alns_time_limit_s"):
        if config.get(name) is not None:
            value = float(config[name])
            if isinstance(config[name], bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative, or null")
            config[name] = value
    if (
        not isinstance(config["seeds"], list) or not config["seeds"]
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in config["seeds"])
        or len(set(config["seeds"])) != len(config["seeds"])
    ):
        raise ValueError("seeds must be a nonempty list of unique integers")
    config["rules"] = asdict(Rules(**config.get("rules", {})))
    windows = dict(config.get("windows", {}))
    if windows.keys() - WINDOW_KEYS:
        raise ValueError(f"Unknown windows keys: {sorted(windows.keys() - WINDOW_KEYS)}")
    if windows.get("cache_dir") is not None:
        windows["cache_dir"] = str(_resolve_path(windows["cache_dir"]))
    config["windows"] = windows
    validation = config.get("validation", {})
    validation_keys = {"time_tolerance_s", "elevation_tolerance_deg", "geometry_step_s"}
    if not isinstance(validation, dict) or validation.keys() - validation_keys:
        raise ValueError("Invalid validation keys")
    for name, value in validation.items():
        # Normalize accepted numeric strings before passing them to arithmetic
        # in the validator. Zero tolerances are legal; sampling steps are not.
        numeric = _nonnegative_number(value, f"validation.{name}")
        if name == "geometry_step_s" and numeric == 0:
            raise ValueError("validation.geometry_step_s must be positive")
        validation[name] = numeric
    config["validation"] = validation
    if "output_dir" in config:
        config["output_dir"] = str(_resolve_path(config["output_dir"]))
    return config


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def select_dataset(dataset: Dataset, config: dict[str, Any]) -> tuple[list, list]:
    """Deterministic random subsets avoid the source file's orbit-order bias."""
    targets = dataset.targets
    if "target_ids" in config:
        chosen = set(config["target_ids"])
        missing = chosen - {target.target_id for target in targets}
        if missing:
            raise ValueError(f"Unknown target_ids: {sorted(missing)}")
        targets = [target for target in targets if target.target_id in chosen]
    elif "target_limit" in config:
        limit = config["target_limit"]
        if limit > len(targets):
            raise ValueError(f"target_limit {limit} exceeds dataset size {len(targets)}")
        # Sorting the population makes selection stable if source records move.
        population = sorted(targets, key=lambda target: target.target_id)
        targets = random.Random(config.get("selection_seed", 0)).sample(population, limit)
    targets = sorted(targets, key=lambda target: target.target_id)
    stations = dataset.stations
    if "station_ids" in config:
        chosen = set(config["station_ids"])
        missing = chosen - {station.station_id for station in stations}
        if missing:
            raise ValueError(f"Unknown station_ids: {sorted(missing)}")
        stations = [station for station in stations if station.station_id in chosen]
    return targets, stations


def _git_record() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, args in [
        ("head", ["rev-parse", "HEAD"]),
        ("status_porcelain", ["status", "--porcelain"]),
    ]:
        command = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        )
        result[name] = command.stdout.strip() if command.returncode == 0 else None
    return result


def _versions() -> dict[str, Any]:
    versions = {"python": platform.python_version(), "space_planner": __version__}
    for package in ("numpy", "scipy", "sgp4", "ortools"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _source_hashes() -> dict[str, str]:
    # Git HEAD alone does not identify a run from an uncommitted working tree.
    paths = sorted((PROJECT_ROOT / "src" / "space_planner").glob("*.py"))
    if (PROJECT_ROOT / "pyproject.toml").exists():
        paths.append(PROJECT_ROOT / "pyproject.toml")
    return {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def report_metrics(plan, priorities, rules, windows) -> dict[str, Any]:
    metrics = objective_metrics(plan, priorities, rules)
    best: dict[int, float] = {}
    for observation in plan:
        score = observation_score(observation, priorities[observation.target_id], rules)
        best[observation.target_id] = max(best.get(observation.target_id, 0), score)
    longest: dict[int, float] = {}
    for window in windows:
        longest[window.target_id] = max(
            longest.get(window.target_id, 0.0), window.end - window.start,
        )
    available = {
        target for target, duration in longest.items()
        if duration + 1e-7 >= rules.min_duration_s
    }
    upper_score = sum(
        observation_score(
            # Only duration matters to this scalar score; resource coupling is
            # deliberately relaxed for an optimistic upper bound.
            Observation(target, "", 0.0, longest[target]),
            priorities[target], rules,
        )
        for target in available
    )
    metrics.update({
        "target_count": len(priorities),
        "coverage_fraction": metrics["coverage_count"] / len(priorities) if priorities else 0,
        "unique_best_observation_score": sum(best.values()),
        "unique_best_score_fraction": sum(best.values()) / sum(priorities.values()) if priorities else 0,
        "available_target_count": len(available),
        "available_coverage_fraction": metrics["coverage_count"] / len(available) if available else 0,
        "priority_weight_sum": sum(priorities.values()),
        "score_semantics": "additive observation score; unique_best counts each target at most once",
        "development_objective": rules.objective,
        "optimistic_coverage_upper_bound": len(available),
        "optimistic_unique_best_score_upper_bound": upper_score,
        "optimistic_observation_score_upper_bound": (
            upper_score if rules.max_observations_per_target == 1 else None
        ),
        "bound_scope": (
            "Given the computed numerical windows, ignore station/target competition "
            "and give every target its longest available single observation. The "
            "observation-score upper bound applies only to at most one observation "
            "per target; unique-best and coverage bounds also apply with repeats. "
            "This is not a continuous-geometry global optimality certificate."
        ),
    })
    return metrics


def _record_plan(plan, name, output, windows, priorities, rules, horizon, geometry, config, solve_s, diagnostics=None):
    started = time.perf_counter()
    validation = validate_plan(
        plan, windows, priorities, rules, horizon, geometry=geometry,
        **config.get("validation", {}),
    )
    report = {
        "algorithm": name,
        "plan": [asdict(observation) for observation in plan],
        "metrics": report_metrics(plan, priorities, rules, windows),
        "validation": validation,
        "timing": {
            "solver_wall_s": solve_s,
            "validation_wall_s": time.perf_counter() - started,
            "process_peak_rss_mib": _peak_rss_mib(),
        },
    }
    if diagnostics is not None:
        report["diagnostics"] = diagnostics
    _write_json(output / "plans" / f"{name}.json", report)
    return {key: value for key, value in report.items() if key not in ("plan", "diagnostics")}


def _probe(plan, windows, priorities, rules, horizon, geometry, config):
    from .local_repair import probe_joint_repairs

    options = dict(config.get("probe", {}))
    if not options.get("block_pairs"):
        raise ValueError(
            "probe requires config.probe.block_pairs; use scripts/probe_real.py "
            "for automatic real-instance block sampling"
        )
    return probe_joint_repairs(
        plan, windows, priorities, rules, horizon, geometry=geometry, **options,
    )


def execute(command: str, config: dict[str, Any], output: Path, *, skip_geometry: bool = False) -> dict:
    """Run one experiment, saving partial manifests even on failure."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "schema_version": 1, "command": command,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "config": config, "versions": _versions(),
        "platform": platform.platform(), "git": _git_record(),
        "source_code_sha256": _source_hashes(),
        "resolved_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "geometry_validation_requested": not skip_geometry,
        "memory_semantics": "process lifetime high-water RSS; not incremental per algorithm",
        "assumption_notice": (
            "Start date, repeat policy, interior-duration scoring and objective are "
            "explicit development choices, not confirmed official competition rules."
        ),
        "ai_status": (
            "Adaptive classical search baseline; no interaction-gain model trained "
            "and no model novelty established."
        ),
    }
    _write_json(output / "manifest.json", manifest)
    try:
        phase = time.perf_counter()
        try:
            dataset = load_dataset(Path(config["data_root"]))
        except DataAuditError as error:
            _write_json(output / "audit.json", error.audit)
            raise
        _write_json(output / "audit.json", dataset.audit)
        targets, stations = select_dataset(dataset, config)
        selection = {
            "target_ids": [target.target_id for target in targets],
            "station_ids": [station.station_id for station in stations],
        }
        selection["sha256"] = hashlib.sha256(
            json.dumps(selection, sort_keys=True).encode()
        ).hexdigest()
        selection["priority_counts"] = dict(Counter(str(target.priority) for target in targets))
        manifest["selection"] = selection
        manifest["input_files"] = dataset.audit["files"]
        manifest["data_wall_s"] = time.perf_counter() - phase
        summary: dict[str, Any] = {
            "command": command, "output_directory": str(output),
            "target_count": len(targets), "station_count": len(stations),
            "development_objective": config["rules"]["objective"],
        }
        if command == "audit":
            summary["data_valid"] = dataset.audit["valid"]
        else:
            from .geometry import Geometry, compute_windows

            geometry = Geometry(targets, stations, parse_start_utc(config["start_utc"]))
            horizon = config["horizon_s"]
            options = dict(config.get("windows", {}))
            if options.get("cache_dir") is not None:
                options["cache_dir"] = Path(options["cache_dir"])

            def progress(*args, **kwargs):
                message = " ".join(str(value) for value in args)
                if kwargs:
                    message += " " + str(kwargs)
                print(message, file=sys.stderr, flush=True)

            phase = time.perf_counter()
            windows, stats = compute_windows(geometry, horizon, progress=progress, **options)
            manifest["windows_wall_s"] = time.perf_counter() - phase
            manifest["window_stats"] = stats
            _write_json(output / "windows.json", {
                "start_utc": config["start_utc"], "horizon_s": horizon,
                "selection_sha256": selection["sha256"],
                "windows": [asdict(window) for window in windows],
                "stats": stats,
            })
            summary["window_count"] = len(windows)
            summary["window_stats"] = stats
            if command != "windows":
                rules = Rules(**config["rules"])
                priorities = {target.target_id: target.priority for target in targets}
                validation_geometry = None if skip_geometry else geometry
                phase = time.perf_counter()
                greedy_diagnostics: dict[str, Any] = {}
                greedy = greedy_schedule(
                    windows, priorities, rules,
                    strategy=config.get("greedy_strategy", "portfolio"),
                    seed=config.get("greedy_seed", 0),
                    time_limit_s=config.get("greedy_time_limit_s"),
                    diagnostics=greedy_diagnostics,
                )
                reports = [_record_plan(
                    greedy, "greedy", output, windows, priorities, rules, horizon,
                    validation_geometry, config, time.perf_counter() - phase,
                    diagnostics=greedy_diagnostics,
                )]
                if command in ("run", "benchmark"):
                    for seed in config["seeds"]:
                        phase = time.perf_counter()
                        plan, diagnostics = alns_schedule(
                            windows, priorities, rules, initial=greedy, seed=seed,
                            iterations=config.get("alns_iterations", 100),
                            time_limit_s=config.get("alns_time_limit_s"),
                        )
                        reports.append(_record_plan(
                            plan, f"alns_seed_{seed}", output, windows, priorities, rules,
                            horizon, validation_geometry, config,
                            time.perf_counter() - phase, diagnostics,
                        ))
                if command == "probe" or (command == "run" and config.get("probe")):
                    phase = time.perf_counter()
                    probes = _probe(
                        greedy, windows, priorities, rules, horizon, validation_geometry, config,
                    )
                    _write_json(output / "probe.json", probes)
                    summary["probe_wall_s"] = time.perf_counter() - phase
                    summary["probe_count"] = len(probes)
                summary["algorithms"] = reports
                summary["all_plans_valid"] = all(item["validation"]["valid"] for item in reports)
                summary["geometry_checked"] = not skip_geometry
                summary["budget_note"] = (
                    "ALNS seeds share the saved greedy initial plan. ALNS solver times "
                    "exclude shared greedy/window computation and independent validation; "
                    "both iteration and optional soft wall-clock budgets are recorded."
                )
        summary["wall_s"] = time.perf_counter() - started
        summary["process_peak_rss_mib"] = _peak_rss_mib()
        _write_json(output / "summary.json", summary)
        manifest["status"] = "complete" if summary.get("all_plans_valid", True) else "validation_failed"
        return summary
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["wall_s"] = time.perf_counter() - started
        manifest["process_peak_rss_mib"] = _peak_rss_mib()
        _write_json(output / "manifest.json", manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "windows", "run", "benchmark", "probe"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New or empty output directory")
    parser.add_argument(
        "--skip-geometry", action="store_true",
        help="Only structural validation; output explicitly records geometry_checked=false",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, require_start=args.command != "audit")
        if args.output is None:
            base = Path(config.get("output_dir", PROJECT_ROOT / "outputs" / args.config.stem))
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            output = base / f"{run_id}_{args.command}"
        else:
            output = args.output
        summary = execute(args.command, config, output, skip_geometry=args.skip_geometry)
        print(json.dumps(summary, ensure_ascii=False, default=_json_value, allow_nan=False))
        return 0 if summary.get("all_plans_valid", True) else 2
    except (ValueError, OSError, TypeError) as error:
        print(f"space-plan: {error}", file=sys.stderr)
        return 2
