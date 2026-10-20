"""Local, isolated Famou evaluation against saved real-data development cases.

Only the candidate source and aggregate numeric metrics belong in the hybrid
cloud workflow. Raw orbital data, windows, and the independent evaluator stay
on this machine. The normalized score is a development objective, not the
competition's official scoring formula.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
from typing import Any

from .cli import parse_start_utc, select_dataset
from .data import load_dataset
from .geometry import GEOMETRY_VERSION, Geometry
from .types import Observation, Rules, Window
from .validation import validate_plan


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = PROJECT_ROOT / "outputs" / "congested_300_1h_run01"
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class EvaluationCase:
    """Trusted case, assembled before any candidate code starts."""

    windows: list[Window]
    priorities: dict[int, int]
    rules: Rules
    horizon_s: float
    geometry: Any

    def public_instance(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "windows": [asdict(window) for window in self.windows],
            "priorities": {str(key): value for key, value in self.priorities.items()},
            "rules": asdict(self.rules),
            "horizon_s": self.horizon_s,
            "seed": 20261020,
        }


class CandidateFailure(Exception):
    """Numeric codes can be sent to the cloud without exposing local paths."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _load_cases() -> list[EvaluationCase]:
    paths = os.environ.get("FAMOU_RUN_DIRS", str(DEFAULT_RUN)).split(os.pathsep)
    if not paths or any(not path.strip() for path in paths):
        raise ValueError("FAMOU_RUN_DIRS must contain nonempty saved-run paths")
    cases = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        manifest = json.loads((path / "manifest.json").read_text())
        config = manifest["config"]
        dataset = load_dataset(Path(config["data_root"]))
        if dataset.audit["files"] != manifest["input_files"]:
            raise ValueError("Raw input fingerprints differ from the saved run")
        targets, stations = select_dataset(dataset, config)
        if [target.target_id for target in targets] != manifest["selection"]["target_ids"]:
            raise ValueError("Target selection differs from the saved run")
        if [station.station_id for station in stations] != manifest["selection"]["station_ids"]:
            raise ValueError("Station selection differs from the saved run")
        payload = json.loads((path / "windows.json").read_text())
        if payload["stats"]["geometry_version"] != GEOMETRY_VERSION:
            raise ValueError("Saved windows use a different geometry model")
        rules = Rules(**config["rules"])
        if rules.max_observations_per_target != 1 or rules.objective != "coverage_then_score":
            raise ValueError("This evaluator requires single-observation coverage-first rules")
        priorities = {target.target_id: target.priority for target in targets}
        if not priorities or any(priority <= 0 for priority in priorities.values()):
            raise ValueError("Expected nonempty positive priorities")
        cases.append(EvaluationCase(
            windows=[Window(**window) for window in payload["windows"]],
            priorities=priorities,
            rules=rules,
            horizon_s=float(config["horizon_s"]),
            geometry=Geometry(targets, stations, parse_start_utc(config["start_utc"])),
        ))
    return cases


# This worker has no repository imports and executes only inside the OS sandbox.
# Limits are installed in the child, avoiding preexec_fn in the SDK's threads.
_WORKER = r"""
import json, math, random, resource, runpy, sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_FSIZE, (8388608, 8388608))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
cpu = int(math.ceil(float(sys.argv[2]))) + 1
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
random.seed(20261020)
instance = json.load(sys.stdin)
namespace = runpy.run_path(sys.argv[1], run_name="famou_candidate")
solve = namespace.get("solve")
if not callable(solve):
    raise ValueError("Candidate must define solve(instance)")
result = solve(instance)
sys.stdout.write(json.dumps(result, allow_nan=False, separators=(",", ":")))
sys.stdout.flush()
"""


def _sandbox_command(workdir: Path, candidate: Path, timeout_s: float) -> list[str]:
    """Fail closed if the supported macOS execution boundary is unavailable.

    The profile allows the Python runtime and scratch directory only, denies
    networking and subprocess creation, and cannot read or overwrite the
    repository, source data, user credential files, or evaluator.
    """
    sandbox = Path("/usr/bin/sandbox-exec")
    if sys.platform != "darwin" or not sandbox.is_file():
        raise CandidateFailure(6, "A supported OS sandbox is required")
    executable = Path(sys.executable).resolve()
    framework_executable = (
        Path(sys.base_prefix) / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    )
    if framework_executable.is_file():
        # Framework bin/python is a launcher that posix_spawn()s this binary.
        # Execute it directly so process creation remains denied to candidates.
        executable = framework_executable.resolve()
    roots = {
        Path(sys.base_prefix).resolve(),
        Path(sysconfig.get_path("stdlib")).resolve(),
        executable.parent,
        Path("/System"),
        Path("/usr/lib"),
        Path("/usr/share"),
        Path("/Library/Apple"),
    }
    # SBPL strings use JSON-compatible escaping for these local paths.
    quoted = lambda value: json.dumps(str(value))
    profile = [
        "(version 1)",
        "(deny default)",
        "(allow sysctl-read)",
        "(allow file-read-metadata)",
        # macOS 27 dyld ignition reads the root directory while locating its
        # cache. Literal "/" permits that directory, not its descendants.
        '(allow file-read* (literal "/"))',
        "(allow process-exec (literal " + quoted(executable) + "))",
        "(allow file-read* (subpath " + quoted(workdir) + "))",
        "(allow file-write* (literal " + quoted(workdir / "stdout.json") + "))",
        "(allow file-write* (literal " + quoted(workdir / "stderr.txt") + "))",
        '(allow file-read* (literal "/dev/urandom") (literal "/dev/random"))',
        '(allow file-read* file-write* (literal "/dev/null"))',
    ]
    profile.extend("(allow file-read* (subpath " + quoted(root) + "))" for root in sorted(roots))
    return [str(sandbox), "-p", "\n".join(profile), str(executable), "-I", "-S",
            "-c", _WORKER, str(candidate), str(timeout_s)]


def _run_candidate(source: bytes, instance: dict, timeout_s: float) -> Any:
    with tempfile.TemporaryDirectory(prefix="space-famou-") as temporary:
        workdir = Path(temporary).resolve()
        candidate = workdir / "candidate.py"
        candidate.write_bytes(source)
        command = _sandbox_command(workdir, candidate, timeout_s)
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(workdir),
            "TMPDIR": str(workdir),
            "LANG": "C.UTF-8",
        }
        # File-backed output avoids retaining an unbounded candidate stdout
        # stream in the evaluator; RLIMIT_FSIZE caps both output files.
        with (workdir / "stdout.json").open("w+b") as stdout, (workdir / "stderr.txt").open("wb") as stderr:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                cwd=workdir, env=environment, start_new_session=True,
            )
            try:
                process.communicate(
                    json.dumps(instance, allow_nan=False).encode(), timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
                raise CandidateFailure(2, "Candidate exceeded its time budget")
            if process.returncode:
                raise CandidateFailure(1, "Candidate subprocess failed")
            # Read the existing descriptor, never reopen a path a candidate
            # could have replaced with a symlink.
            stdout.seek(0)
            result_bytes = stdout.read(MAX_OUTPUT_BYTES + 1)
        if len(result_bytes) > MAX_OUTPUT_BYTES:
            raise CandidateFailure(3, "Candidate output exceeded its limit")
        try:
            return json.loads(
                result_bytes.decode(),
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")),
            )
        except (ValueError, UnicodeError) as error:
            raise CandidateFailure(3, "Candidate did not return strict JSON") from error


def _observations(result: Any, case: EvaluationCase) -> list[Observation]:
    if not isinstance(result, list) or len(result) > len(case.priorities):
        raise CandidateFailure(3, "Expected a bounded list of observation objects")
    observations = []
    for item in result:
        if not isinstance(item, dict) or set(item) != {"target_id", "station_id", "start", "end"}:
            raise CandidateFailure(3, "Observation fields must match the contract")
        if type(item["target_id"]) is not int or not isinstance(item["station_id"], str):
            raise CandidateFailure(3, "Observation identifiers have incorrect types")
        for key in ("start", "end"):
            if type(item[key]) not in (int, float) or not math.isfinite(item[key]):
                raise CandidateFailure(3, "Observation times must be finite numbers")
        observations.append(Observation(**item))
    return observations


def _score(plan: list[Observation], priorities: dict[int, int], rules: Rules) -> float:
    """Independent scalar computation after feasibility validation."""
    total = 0.0
    for observation in plan:
        duration = observation.end - observation.start
        if duration + 1e-7 >= rules.full_duration_s:
            fraction = 1.0
        elif rules.score_mode == "linear" and rules.full_duration_s > rules.min_duration_s:
            fraction = 0.6 + 0.4 * (
                (duration - rules.min_duration_s)
                / (rules.full_duration_s - rules.min_duration_s)
            )
        else:
            fraction = 0.6
        total += priorities[observation.target_id] * fraction
    return total


def evaluate(candidate_path: str) -> dict[str, float]:
    """Famou entry point. Infrastructure/data errors raise; bad candidates score 0.

    The suite score orders total covered targets first, total priority-duration
    score second. A secondary improvement can never offset one missing target:
    (coverage + observation_score / (priority_sum + 1)) / (target_count + 1).
    All cases must pass structural AND independently recomputed geometry checks.
    """
    started = time.perf_counter()
    cases = _load_cases()
    timeout_s = float(os.environ.get("FAMOU_CANDIDATE_TIMEOUT_S", "20"))
    if not math.isfinite(timeout_s) or not 0 < timeout_s <= 600:
        raise ValueError("FAMOU_CANDIDATE_TIMEOUT_S must be in (0, 600]")
    metrics = {
        "validity": 0.0, "combined_score": 0.0, "coverage_count": 0.0,
        "observation_score": 0.0, "case_count": float(len(cases)),
        "error_code": 0.0, "geometry_checked": 0.0,
    }
    try:
        path = Path(candidate_path)
        if not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
            raise CandidateFailure(6, "Candidate source is missing or oversized")
        source = path.read_bytes()
        coverage, score, priority_sum, target_count = 0, 0.0, 0, 0
        for index, case in enumerate(cases):
            result = _run_candidate(source, case.public_instance(), timeout_s)
            plan = _observations(result, case)
            structural = validate_plan(
                plan, case.windows, case.priorities, case.rules, case.horizon_s,
            )
            if not structural["valid"]:
                raise CandidateFailure(4, "Candidate violates structural constraints")
            if case.geometry is None:
                raise ValueError("Geometry validation cannot be disabled")
            validation = validate_plan(
                plan, case.windows, case.priorities, case.rules, case.horizon_s,
                geometry=case.geometry,
            )
            if not validation["valid"] or not validation["geometry_checked"]:
                raise CandidateFailure(5, "Candidate fails geometry validation")
            case_coverage = len({observation.target_id for observation in plan})
            case_score = _score(plan, case.priorities, case.rules)
            metrics[f"case_{index}_coverage"] = float(case_coverage)
            metrics[f"case_{index}_score"] = case_score
            coverage += case_coverage
            score += case_score
            priority_sum += sum(case.priorities.values())
            target_count += len(case.priorities)
        metrics.update({
            "validity": 1.0, "geometry_checked": 1.0,
            "coverage_count": float(coverage), "observation_score": score,
            "combined_score": (coverage + score / (priority_sum + 1)) / (target_count + 1),
        })
    except CandidateFailure as error:
        metrics["error_code"] = float(error.code)
    metrics["elapsed_s"] = time.perf_counter() - started
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Evaluator produced non-finite metrics")
    return metrics
