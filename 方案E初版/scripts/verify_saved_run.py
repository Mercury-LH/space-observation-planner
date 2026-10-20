"""Independently revalidate saved plans with the current validator and geometry."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from space_planner.cli import _write_json, parse_start_utc, select_dataset, report_metrics
from space_planner.data import load_dataset
from space_planner.geometry import Geometry, GEOMETRY_VERSION
from space_planner.types import Window, Observation, Rules
from space_planner.validation import validate_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new filename")
    started = time.perf_counter()
    manifest = json.loads((args.run / "manifest.json").read_text())
    config = manifest["config"]
    dataset = load_dataset(Path(config["data_root"]))
    if dataset.audit["files"] != manifest["input_files"]:
        raise ValueError("Raw input fingerprints differ from the saved run")
    targets, stations = select_dataset(dataset, config)
    geometry = Geometry(targets, stations, parse_start_utc(config["start_utc"]))
    priorities = {t.target_id: t.priority for t in targets}
    window_payload = json.loads((args.run / "windows.json").read_text())
    if window_payload["stats"]["geometry_version"] != GEOMETRY_VERSION:
        raise ValueError("Geometry model version differs; regenerate windows before revalidation")
    windows = [Window(**w) for w in window_payload["windows"]]
    rules = Rules(**config["rules"])
    reports = []
    validations = {}
    for path in sorted((args.run / "plans").glob("*.json")):
        payload = json.loads(path.read_text())
        plan = [Observation(**p) for p in payload["plan"]]
        key = tuple(plan)  # Exact record equality, not a hash-only comparison.
        if key not in validations:
            validations[key] = (
                validate_plan(
                    plan, windows, priorities, rules, config["horizon_s"], geometry=geometry,
                ), str(path),
            )
        validation, validated_path = validations[key]
        reports.append({
            "file": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "validation": validation,
            "identical_plan_validated_at": validated_path,
            "metrics": report_metrics(plan, priorities, rules, windows),
        })
    source = Path(__file__).resolve().parents[1] / "src" / "space_planner"
    payload = {
        "run": str(args.run), "rules": asdict(rules),
        "validator_sha256": hashlib.sha256((source / "validation.py").read_bytes()).hexdigest(),
        "geometry_sha256": hashlib.sha256((source / "geometry.py").read_bytes()).hexdigest(),
        "all_valid": bool(reports) and all(r["validation"]["valid"] for r in reports),
        "unique_plans_validated": len(validations),
        "plans": reports, "wall_s": time.perf_counter() - started,
    }
    _write_json(args.output, payload)
    print(json.dumps({"all_valid": payload["all_valid"], "plan_count": len(reports), "wall_s": payload["wall_s"]}))
    return 0 if payload["all_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
