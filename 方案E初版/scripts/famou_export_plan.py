#!/usr/bin/env python3
"""Run a downloaded Famou candidate locally and export its validated schedule.

No cloud request or credential loading occurs. Candidate execution uses the
same OS sandbox as the hybrid evaluator. Files appear in the requested empty
output directory only after structural and geometry validation both succeed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

from space_planner import famou_evaluator as adapter
from space_planner.cli import report_metrics


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _require_empty_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Output must be a new directory or an existing empty directory")
    if path.is_symlink():
        raise ValueError("Output must not be a symbolic link")


def export_candidate(candidate_path: Path, run_dir: Path, output: Path, timeout_s=20.0):
    """Evaluate once; score and exported observations come from that same run."""
    started = time.perf_counter()
    candidate_path = Path(candidate_path).expanduser().resolve(strict=True)
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    _require_empty_output(output)
    if not math.isfinite(timeout_s) or not 0 < timeout_s <= 600:
        raise ValueError("Candidate timeout must be in (0, 600] seconds")
    if not candidate_path.is_file() or candidate_path.stat().st_size > adapter.MAX_SOURCE_BYTES:
        raise adapter.CandidateFailure(6, "Candidate source is missing or oversized")
    source = candidate_path.read_bytes()
    input_manifest_bytes = (run_dir / "manifest.json").read_bytes()
    input_manifest = json.loads(input_manifest_bytes)
    windows_digest = _sha256((run_dir / "windows.json").read_bytes())
    # Reuse the evaluator's verified saved-run loader without changing any
    # caller setting permanently or loading an API credential file.
    previous = os.environ.get("FAMOU_RUN_DIRS")
    os.environ["FAMOU_RUN_DIRS"] = str(run_dir)
    try:
        cases = adapter._load_cases()
    finally:
        if previous is None:
            os.environ.pop("FAMOU_RUN_DIRS", None)
        else:
            os.environ["FAMOU_RUN_DIRS"] = previous
    if len(cases) != 1:
        raise ValueError("Plan export requires exactly one saved-run case")
    case = cases[0]
    solving = time.perf_counter()
    result = adapter._run_candidate(source, case.public_instance(), timeout_s)
    solve_s = time.perf_counter() - solving
    plan = adapter._observations(result, case)
    validating = time.perf_counter()
    structural = adapter.validate_plan(
        plan, case.windows, case.priorities, case.rules, case.horizon_s,
    )
    if not structural["valid"]:
        raise adapter.CandidateFailure(4, "Candidate violates structural constraints; nothing exported")
    if case.geometry is None:
        raise ValueError("Geometry validation cannot be disabled")
    validation = adapter.validate_plan(
        plan, case.windows, case.priorities, case.rules, case.horizon_s,
        geometry=case.geometry,
    )
    if not validation["valid"] or not validation["geometry_checked"]:
        raise adapter.CandidateFailure(5, "Candidate fails geometry validation; nothing exported")
    validation_s = time.perf_counter() - validating
    coverage = len({observation.target_id for observation in plan})
    score = adapter._score(plan, case.priorities, case.rules)
    metrics = {
        "validity": 1.0,
        "geometry_checked": 1.0,
        "combined_score": (
            coverage + score / (sum(case.priorities.values()) + 1)
        ) / (len(case.priorities) + 1),
        "coverage_count": float(coverage),
        "observation_score": score,
        "case_count": 1.0,
        "case_0_coverage": float(coverage),
        "case_0_score": score,
        "error_code": 0.0,
        "elapsed_s": time.perf_counter() - started,
    }
    plan_report = {
        "algorithm": "famou_candidate",
        "plan": [asdict(observation) for observation in plan],
        "metrics": report_metrics(plan, case.priorities, case.rules, case.windows),
        "validation": validation,
        "timing": {"solver_wall_s": solve_s, "validation_wall_s": validation_s},
        "candidate_sha256": _sha256(source),
    }
    manifest = {
        "format": "famou-plan-export-v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_path": str(candidate_path),
        "candidate_sha256": _sha256(source),
        "source_run_dir": str(run_dir),
        "source_manifest_sha256": _sha256(input_manifest_bytes),
        "source_windows_sha256": windows_digest,
        "input_manifest": input_manifest,
        "geometry_version": adapter.GEOMETRY_VERSION,
        "evaluator_sha256": _sha256(Path(adapter.__file__).read_bytes()),
        "validator_sha256": _sha256(Path(sys.modules[adapter.validate_plan.__module__].__file__).read_bytes()),
        "candidate_timeout_s": timeout_s,
        "metrics_file": "metrics.json",
        "plan_file": "plans/famou_candidate.json",
    }
    # Prepare the complete bundle next to its destination, then atomically
    # rename it. A bad candidate or failed serialization leaves no valid-looking
    # plan or manifest in the requested output directory.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".famou-export-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "plans").mkdir()
        (stage / "candidate.py").write_bytes(source)
        _write_json(stage / "metrics.json", metrics)
        _write_json(stage / "plans/famou_candidate.json", plan_report)
        _write_json(stage / "manifest.json", manifest)
        _require_empty_output(output)
        os.replace(stage, output)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args(argv)
    try:
        metrics = export_candidate(args.candidate, args.run_dir, args.output, args.timeout)
    except (adapter.CandidateFailure, ValueError, OSError) as error:
        print(f"Export failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output.absolute()), "metrics": metrics}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
