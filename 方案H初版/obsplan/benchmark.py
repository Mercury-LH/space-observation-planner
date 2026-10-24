"""Isolated, reproducible evaluation of fresh scheduling runs.

The time-weighted utility is an internal experiment metric, not a contest
score. Failed, timed-out and unvalidated runs never enter the leaderboard.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
import uuid

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIME_WEIGHT = 3 / 7
WEIGHTS = (0.0, 0.1, 0.2, 0.4)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def public_config(cfg) -> dict:
    """Paths affect file discovery, but must not leak into shareable reports."""
    return {k: v for k, v in asdict(cfg).items() if k not in ("data_dir", "cache_dir")}


def source_identity(plugin: str | None = None) -> dict:
    paths = sorted((ROOT / "obsplan").rglob("*.py"))
    paths += sorted((ROOT / "integrations").rglob("*.py"))
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    plugin_hash = None
    if plugin:
        module = importlib.import_module(plugin.split(":")[0])
        path = getattr(module, "__file__", None)
        if not path or not Path(path).is_file():
            raise ValueError("Plugin source must be a readable Python file")
        plugin_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    return {"revision": result.stdout.strip() if result.returncode == 0 else None,
            "source_hash": digest(files), "source_files": files, "plugin_source_hash": plugin_hash}


def _parse_power_profile(battery: str, custom: str) -> dict:
    """Read only the active source and its setting; ignore charge and time left."""
    profile = {"source": "unknown", "low_power_mode": "unknown"}
    match = re.search(r"^Now drawing from '(AC Power|Battery Power)'\s*$", battery, re.MULTILINE)
    if not match:
        return profile
    active = match.group(1)
    profile["source"] = "AC" if active == "AC Power" else "Battery"
    selected = False
    for line in custom.splitlines():
        if line and not line[0].isspace() and line.rstrip().endswith(":"):
            selected = line.strip() == active + ":"
        elif selected:
            setting = re.fullmatch(r"\s*lowpowermode\s+(\d+)\s*", line)
            if setting:
                profile["low_power_mode"] = int(setting.group(1))
                break
    return profile


def read_power_profile() -> dict:
    """Sample macOS power configuration without treating unavailable as off.

    Two supply readings bracket the settings read, so a source change while
    sampling yields unknown. Boundary samples cannot detect a switch and back
    occurring entirely between them; they do not measure thermal throttling.
    """
    unknown = {"source": "unknown", "low_power_mode": "unknown"}
    if platform.system() != "Darwin":
        return unknown
    try:
        outputs = []
        for mode in ("batt", "custom", "batt"):
            response = subprocess.run(["pmset", "-g", mode], capture_output=True,
                                      text=True, check=False, timeout=2)
            if response.returncode:
                return unknown
            outputs.append(response.stdout)
    except (OSError, subprocess.TimeoutExpired):
        return unknown
    before = _parse_power_profile(outputs[0], outputs[1])
    after = _parse_power_profile(outputs[2], outputs[1])
    return before if before["source"] == after["source"] else unknown


def environment_identity() -> dict:
    versions = {}
    for name in ("numpy", "sgp4", "numba", "ortools", "skyfield"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    processor, memory = platform.processor(), None
    if platform.system() == "Darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        ram = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        processor = cpu.stdout.strip() if cpu.returncode == 0 else processor
        memory = int(ram.stdout.strip()) if ram.returncode == 0 else None
    elif hasattr(os, "sysconf"):
        try:
            memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            pass
    return {"python": platform.python_version(), "system": platform.system(),
            "release": platform.release(), "machine": platform.machine(),
            "processor": processor, "memory_bytes": memory,
            "logical_cpus": os.cpu_count(), "dependencies": versions,
            "power_profile": read_power_profile()}


def resolve_callable(reference: str):
    """Import an explicit Python callable; never interpret a shell command."""
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", reference):
        raise ValueError("Solver reference must be module:function")
    module, attribute = reference.split(":")
    function = getattr(importlib.import_module(module), attribute)
    if not callable(function):
        raise TypeError("Solver entry point must be callable")
    return function


def check_config(cfg) -> None:
    # Fusion currently hard-codes these assumptions. Reject unsupported claims
    # of configurability until the underlying solvers implement them.
    if cfg.cell_s != 30 or cfg.w_cov != 0.5 or cfg.w_score != 0.5 or cfg.score_ref != "sumP":
        raise ValueError("Portfolio supports cell_s=30, w_cov=w_score=0.5 and score_ref='sumP' only")
    if cfg.obs_min_s != 120 or cfg.obs_max_s != 240 or cfg.switch_s != 30 or cfg.repeat_gap_h != 48:
        raise ValueError("Portfolio supports 120–240 s observations, 30 s switch and 48 h repeat gap only")
    if not math.isfinite(cfg.days) or cfg.days <= 0:
        raise ValueError("Planning horizon must be finite and positive")


def utility(quality: float, solver_s: float, weight: float = DEFAULT_TIME_WEIGHT,
            time_scale_s: float = 120.0) -> float:
    if not all(math.isfinite(x) for x in (quality, solver_s, weight, time_scale_s)):
        raise ValueError("Utility inputs must be finite")
    if solver_s < 0 or time_scale_s <= 0 or not 0 <= weight <= 1:
        raise ValueError("Invalid time or weight")
    return 100.0 * ((1.0 - weight) * quality + weight * time_scale_s / (time_scale_s + solver_s))


def aggregate(results: list[dict], *, weight: float = DEFAULT_TIME_WEIGHT,
              time_scale_s: float = 120.0, weights=WEIGHTS) -> dict:
    """Only fully valid repeated methods with the same experiment fingerprint rank."""
    weight, time_scale_s = float(weight), float(time_scale_s)
    utility(0.0, 0.0, weight, time_scale_s)
    all_weights = sorted(set([float(w) for w in weights] + [weight]))
    for w in all_weights:
        utility(0.0, 0.0, w, time_scale_s)
    identities = {r.get("comparison_key") for r in results if r.get("comparison_key")}
    if len(identities) > 1:
        raise ValueError("Refusing to compare different inputs/configs/code/environments")
    plugins = {}
    for r in results:
        entrypoint = r.get("entrypoint")
        plugin_hash = r.get("provenance", {}).get("plugin_source_hash")
        if entrypoint and plugin_hash:
            plugins.setdefault(entrypoint, set()).add(plugin_hash)
    if any(len(versions) > 1 for versions in plugins.values()):
        raise ValueError("Cannot compare runs with changed external plugin code")
    groups = {}
    for r in results:
        groups.setdefault(r["preset"], []).append(r)
    methods = []
    for name, runs in sorted(groups.items()):
        entrypoints = {r["entrypoint"] for r in runs if r.get("entrypoint")}
        if len(entrypoints) > 1:
            raise ValueError(f"Preset {name} has different solver entry points; use distinct names")
        specs = {r["solver_spec_hash"] for r in runs if r.get("solver_spec_hash")}
        if len(specs) > 1:
            raise ValueError(f"Preset {name} has different solver parameters; use distinct names")
        eligible = all(r.get("rankable") is True and r.get("status") == "valid"
                       and not r.get("worker_exit")
                       and r.get("comparison_key") and r.get("validation", {}).get("legal") is True
                       and r.get("validation", {}).get("step_s") == 1.0
                       and r.get("validation", {}).get("geometry_checked") is True
                       and r.get("fresh_solve") is True for r in runs)
        row = {"preset": name, "runs": len(runs), "rankable": bool(eligible),
               "status_counts": {s: sum(r["status"] == s for r in runs)
                                 for s in sorted({r["status"] for r in runs})},
               "run_ids": [f"{r.get('suite_id', '')}:{r['run_id']}" for r in runs],
               "seeds": [r.get("seed") for r in runs]}
        if eligible:
            quality = [r["metrics"]["combined"] for r in runs]
            solver = [r["timing_s"]["solver"] for r in runs]
            row.update({"quality_median": statistics.median(quality),
                        "covered_median": statistics.median(r["metrics"]["covered"] for r in runs),
                        "score_median": statistics.median(r["metrics"]["score"] for r in runs),
                        "solver_s_median": statistics.median(solver),
                        "solver_s_min": min(solver), "solver_s_max": max(solver),
                        "quality_min": min(quality), "quality_max": max(quality),
                        "time_utility_median": statistics.median(time_scale_s / (time_scale_s + t) for t in solver),
                        "utility_by_weight": {str(w): statistics.median(utility(q, t, w, time_scale_s)
                                                for q, t in zip(quality, solver)) for w in all_weights},
                        "timing_s_median": {phase: statistics.median(r["timing_s"][phase] for r in runs)
                                            for phase in ("preprocessing", "solver", "validation", "total")}})
        methods.append(row)
    valid = [row for row in methods if row["rankable"]]
    for row in valid:
        row["pareto"] = not any(other["quality_median"] >= row["quality_median"]
                                and other["solver_s_median"] <= row["solver_s_median"]
                                and (other["quality_median"] > row["quality_median"]
                                     or other["solver_s_median"] < row["solver_s_median"])
                                for other in valid if other is not row)
    return {"schema_version": SCHEMA_VERSION, "metric": "internal_time_weighted_utility",
            "quality_definition": "evaluate(cfg, dataset, schedule)['combined']; not bounded by 1",
            "formula": "100*((1-w)*Q+w*time_scale_s/(time_scale_s+solver_wall_s))",
            "default_weight": weight, "time_scale_s": time_scale_s,
            "aggregation": "median of per-run utilities; all requested repeats must validate",
            "comparison_key": next(iter(identities), None), "methods": methods,
            "rankings": {str(w): [r["preset"] for r in sorted(valid,
                         key=lambda r: (-r["utility_by_weight"][str(w)], -r["quality_median"], r["preset"]))]
                         for w in all_weights}}


def render_report(report: dict) -> str:
    w = str(report["default_weight"])
    lines = ["# 自动评测结果", "", "这是团队内部质量—时间效用，不是官方分数，也不保证上限为 100。",
             f"时间权重 {report['default_weight']:g}；固定时间尺度 {report['time_scale_s']:g} 秒。",
             "求解计时包括所有候选生成、融合和精修；预处理与独立校验另列。", "",
             "| 方法 | 状态 | Q中位数 | 覆盖数 | 累计得分 | 求解秒 | 内部效用 | Pareto |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in report["methods"]:
        if r["rankable"]:
            lines.append(f"| {r['preset']} | 全部合法 ({r['runs']}) | {r['quality_median']:.6f} | "
                         f"{r['covered_median']:g} | {r['score_median']:.3f} | {r['solver_s_median']:.3f} | "
                         f"{r['utility_by_weight'][w]:.4f} | {'是' if r['pareto'] else '否'} |")
        else:
            lines.append(f"| {r['preset']} | 不排名：{r['status_counts']} | — | — | — | — | — | — |")
    lines += ["", "时间权重敏感性（效用越高越好）：", "",
              "| 时间权重 | 排名 |", "|---:|---|"]
    for weight, names in report["rankings"].items():
        lines.append(f"| {weight} | {' > '.join(names) or '无合格结果'} |")
    lines += ["", "每个 run/result.json 保存原始指标、配置、数据/代码指纹、随机种子、线程和阶段耗时。",
              "plan.csv 和本地日志含目标明细，仅保存在忽略的 outputs 中。",
              "单次试验不能证明统计稳定或全局最优；缓存冷热和平台会影响耗时。", ""]
    return "\n".join(lines)


def report_directory(directory: Path, weight=None, time_scale_s=None, include=()) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    results = []
    seen = set()
    for folder in (directory, *(Path(p) for p in include)):
        current = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if current["suite_id"] in seen:
            raise ValueError("An experiment was included more than once")
        seen.add(current["suite_id"])
        for run in current["runs"]:
            result = json.loads((folder / run["run_id"] / "result.json").read_text(encoding="utf-8"))
            if result.get("suite_id") != current["suite_id"] or result.get("run_id") != run["run_id"]:
                raise ValueError("Result does not belong to this fresh experiment")
            results.append(result)
    report = aggregate(results, weight=manifest["time_weight"] if weight is None else weight,
                       time_scale_s=manifest["time_scale_s"] if time_scale_s is None else time_scale_s,
                       weights=manifest["weights"])
    write_json(directory / "summary.json", report)
    (directory / "report.md").write_text(render_report(report), encoding="utf-8")
    return report


def _terminate(process) -> None:
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def execute_run(request: dict, directory: Path, timeout_s: float) -> dict:
    directory.mkdir()
    write_json(directory / "request.json", request)  # Local only: resolved input/cache paths.
    result = {"schema_version": SCHEMA_VERSION, "suite_id": request["suite_id"],
              "run_id": request["run_id"], "preset": request["preset"],
              "repeat": request["repeat"], "seed": request["seed"], "workers": request["workers"],
              "fresh_solve": False, "rankable": False, "status": "starting"}
    write_json(directory / "result.json", result)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[key] = str(request["workers"])
    started = time.perf_counter()
    outcome = None
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen([sys.executable, "-m", "obsplan.benchmark_worker",
                                        str(directory / "request.json")], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=timeout_s)
                if code:
                    outcome = "worker_error"
            except subprocess.TimeoutExpired:
                _terminate(process)
                outcome = "timeout"
        except OSError:
            outcome = "launch_error"
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    result.setdefault("timing_s", {})["total"] = time.perf_counter() - started
    active_phase = result.pop("active_phase", None)
    phase_started = result.pop("active_phase_started_monotonic", None)
    if outcome and active_phase and phase_started is not None:
        result["timing_s"][active_phase] = time.monotonic() - phase_started
    if outcome and (outcome != "worker_error" or result["status"] in ("starting", "running", "valid", "unvalidated")):
        result.update(status=outcome, rankable=False)
        result["timeout_s"] = timeout_s
    if outcome:
        # A previously saved valid checkpoint is not a successful process exit.
        # Keep explicit validation/input failure details, but never its rankability.
        result["rankable"] = False
        result["worker_exit"] = outcome
    elif result["status"] in ("starting", "running"):
        result.update(status="incomplete_result", rankable=False)
    write_json(directory / "result.json", result)
    return result


def run_suite(args) -> dict:
    from .config import Config
    from .portfolio import PRESETS
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    config = dict(suite.get("config", {}))
    if args.config:
        config.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    cfg = Config(**config)
    check_config(cfg)
    presets = args.presets or suite.get("presets", list(PRESETS))
    if isinstance(presets, str):
        presets = [presets]
    presets = [item for name in presets for item in name.split(",")]
    if not presets or len(set(presets)) != len(presets):
        raise ValueError("Presets must be a nonempty unique list")
    repeats = args.repeats if args.repeats is not None else suite.get("repeats", 1)
    workers = args.workers if args.workers is not None else suite.get("workers", 8)
    timeout = args.timeout_s if args.timeout_s is not None else suite.get("timeout_s", 2400)
    if not isinstance(repeats, int) or repeats < 1 or not isinstance(workers, int) or workers < 1:
        raise ValueError("repeats and workers must be positive integers")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_s must be finite and positive")
    weight, scale = suite.get("time_weight", DEFAULT_TIME_WEIGHT), suite.get("time_scale_s", 120.0)
    utility(0, 0, weight, scale)
    specs = {}
    for name in presets:
        custom = suite.get("solvers", {}).get(name)
        if custom:
            entrypoint = custom["entrypoint"]
            resolve_callable(entrypoint)
            specs[name] = {"entrypoint": entrypoint, "spec": deepcopy(custom.get("spec", {}))}
        elif name in PRESETS:
            specs[name] = {"entrypoint": "obsplan.portfolio:solve", "spec": deepcopy(PRESETS[name])}
        else:
            raise ValueError(f"Unknown preset: {name}")
    output = Path(args.output).resolve()
    # Full schedules must remain in the repository's ignored output area.
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("Benchmark output must be under this scheme's ignored outputs/ directory")
    output.mkdir(parents=True, exist_ok=False)
    suite_id = uuid.uuid4().hex
    manifest = {"schema_version": SCHEMA_VERSION, "suite_id": suite_id, "presets": presets,
                "repeats": repeats, "workers": workers, "timeout_s": timeout,
                "time_weight": weight, "time_scale_s": scale, "weights": suite.get("weights", list(WEIGHTS)),
                "validation_enabled": not args.no_validate, "config": public_config(cfg), "runs": []}
    # Rotate order across repetitions so a particular solver does not always pay cold-cache cost.
    for repeat in range(repeats):
        order = presets[repeat % len(presets):] + presets[:repeat % len(presets)]
        for name in order:
            slug = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
            run_id = f"{len(manifest['runs']) + 1:03d}-{slug}-r{repeat + 1}"
            manifest["runs"].append({"run_id": run_id, "preset": name, "repeat": repeat})
    write_json(output / "manifest.json", manifest)
    for item in manifest["runs"]:
        spec = deepcopy(specs[item["preset"]])
        seed = cfg.seed + item["repeat"]
        spec["spec"].update(seed=seed, workers=workers)
        request = {**item, **spec, "suite_id": suite_id, "seed": seed, "workers": workers,
                   "config": asdict(cfg.with_(seed=seed)), "validate": not args.no_validate}
        result = execute_run(request, output / item["run_id"], timeout)
        print(f"{item['run_id']}: {result['status']}; total={result['timing_s']['total']:.3f}s", flush=True)
    return report_directory(output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Generate, validate and compare fresh schedules")
    run.add_argument("--suite", default="benchmarks/default.json")
    run.add_argument("--output", required=True)
    run.add_argument("--presets", nargs="+")
    run.add_argument("--repeats", type=int)
    run.add_argument("--workers", type=int)
    run.add_argument("--config")
    run.add_argument("--timeout-s", type=float)
    run.add_argument("--no-validate", action="store_true", help="Debug only: NEVER eligible for ranking")
    report = commands.add_parser("report", help="Reaggregate this experiment; do not import old plans")
    report.add_argument("directory")
    report.add_argument("--time-weight", type=float)
    report.add_argument("--time-scale-s", type=float)
    report.add_argument("--include", nargs="+", default=[], help="Additional verified experiment directories")
    args = parser.parse_args(argv)
    try:
        result = run_suite(args) if args.command == "run" else report_directory(
            Path(args.directory), args.time_weight, args.time_scale_s, args.include)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(render_report(result))
    return 0 if any(r["rankable"] for r in result["methods"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
