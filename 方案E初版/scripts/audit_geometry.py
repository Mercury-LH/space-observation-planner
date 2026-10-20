"""Compare refined window events at two scan steps on a fixed real-data subset."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from space_planner.cli import load_config, select_dataset, parse_start_utc, _write_json
from space_planner.data import load_dataset
from space_planner.geometry import Geometry, compute_windows


def comparison(first, second):
    def index(windows):
        result = defaultdict(list)
        for window in windows:
            result[(window.target_id, window.station_id)].append(window)
        for windows in result.values():
            windows.sort(key=lambda w: w.start)
        return result
    a, b = index(first), index(second)
    count_changes, differences = [], []
    for key in sorted(a.keys() | b.keys()):
        if len(a[key]) != len(b[key]):
            count_changes.append({
                "target_id": key[0], "station_id": key[1],
                "coarse": len(a[key]), "fine": len(b[key]),
            })
        else:
            for x, y in zip(a[key], b[key]):
                differences.extend([abs(x.start - y.start), abs(x.end - y.end)])
    return {
        "coarse_count": len(first), "fine_count": len(second),
        "pairs_with_count_changes": count_changes,
        "maximum_matched_boundary_difference_s": max(differences, default=0),
        "p99_matched_boundary_difference_s": float(np.quantile(differences, 0.99)) if differences else 0,
        "interpretation": "step sensitivity for this subset, not a completeness proof for every possible orbit",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fine-step", type=float, default=15)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new file")
    started = time.perf_counter()
    config = load_config(args.config)
    dataset = load_dataset(Path(config["data_root"]))
    targets, stations = select_dataset(dataset, config)
    geometry = Geometry(targets, stations, parse_start_utc(config["start_utc"]))
    options = dict(config.get("windows", {}))
    options.pop("workers", None)  # Sequential audit avoids concurrent timing claims.
    coarse, coarse_stats = compute_windows(geometry, config["horizon_s"], **options)
    options["scan_step_s"] = args.fine_step
    fine, fine_stats = compute_windows(geometry, config["horizon_s"], **options)
    result = {
        "config": config, "target_ids": [t.target_id for t in targets],
        "input_files": dataset.audit["files"],
        "coarse_stats": coarse_stats, "fine_stats": fine_stats,
        "comparison": comparison(coarse, fine),
    }
    # Removing only the upper cap diagnoses intervals that must be split by 85°.
    uncapped_geometry = Geometry(
        targets, [replace(s, max_elevation_deg=90) for s in stations], geometry.start_utc,
    )
    uncapped, _ = compute_windows(uncapped_geometry, config["horizon_s"], **options)
    result["upper_cap"] = {
        "uncapped_window_count": len(uncapped),
        "capped_window_count": len(fine),
        "capped_visible_duration_s": sum(w.end - w.start for w in fine),
        "uncapped_visible_duration_s": sum(w.end - w.start for w in uncapped),
    }
    result["wall_s"] = time.perf_counter() - started
    _write_json(args.output, result)
    print(json.dumps({"comparison": result["comparison"], "upper_cap": result["upper_cap"]}))


if __name__ == "__main__":
    main()
