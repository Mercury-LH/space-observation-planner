"""Locate model propagation failures; do not interpret SGP4 errors as real reentry."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
from sgp4.api import SGP4_ERRORS

from space_planner.cli import load_config, select_dataset, parse_start_utc, _write_json
from space_planner.data import load_dataset
from space_planner.geometry import Geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new filename")
    config = load_config(args.config)
    dataset = load_dataset(Path(config["data_root"]))
    targets, stations = select_dataset(dataset, config)
    geometry = Geometry(targets, stations, parse_start_utc(config["start_utc"]))
    satellite = geometry.satellites[args.target_id]
    step = config["windows"].get("scan_step_s", 30)
    times = np.linspace(0, config["horizon_s"], int(np.ceil(config["horizon_s"] / step)) + 1)
    jd, fraction = geometry.dates(times)
    codes, _, _ = satellite.sgp4_array(jd, fraction)
    edges = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, len(codes)]
    runs = []
    for left, right in zip(edges[:-1], edges[1:]):
        runs.append({
            "code": int(codes[left]), "message": SGP4_ERRORS.get(int(codes[left]), "success"),
            "first_sample_s": float(times[left]), "last_sample_s": float(times[right - 1]),
            "first_sample_utc": (geometry.start_utc + timedelta(seconds=float(times[left]))).isoformat(),
            "sample_count": int(right - left),
        })
    result = {
        "config": config, "target_id": args.target_id,
        "target_epoch_utc": geometry.targets[args.target_id].epoch.isoformat(),
        "sample_step_s": step, "runs": runs,
        "scope": (
            "SGP4 numerical propagation status on configured samples; transitions lie "
            "between adjacent samples. This does not establish actual physical reentry."
        ),
    }
    _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
