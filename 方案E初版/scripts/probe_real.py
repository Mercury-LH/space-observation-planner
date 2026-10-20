#!/usr/bin/env python3
"""Sample reproducible cross-station repairs from saved real-data experiments."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from space_planner.data import load_dataset
from space_planner.local_repair import probe_joint_repairs, solve_local_repair
from space_planner.types import Observation, Rules, Window
from space_planner.validation import validate_plan


def _eligible_bins(window, block_s, horizon_s, minimum_s):
    """Canonical disjoint bins with at least one minimum-duration placement."""
    first = max(0, math.floor(window.start / block_s))
    last = min(math.ceil(horizon_s / block_s) - 1, math.floor(window.end / block_s))
    return [
        index for index in range(first, last + 1)
        if min(window.end, (index + 1) * block_s, horizon_s)
        - max(window.start, index * block_s) >= minimum_s - 1e-6
    ]


def sample_blocks(
    windows, plan, rules, horizon_s, *, samples=12, seed=0,
    block_s=1800.0, metric="auto",
):
    """Uniform target sampling, then station-pair/window/bin conditional sampling.

    This is an explicit seed-target population, not an estimate over all
    possible station pairs, local blocks or continuous neighborhoods.
    """
    if samples < 0 or not math.isfinite(block_s) or block_s < rules.min_duration_s:
        raise ValueError("samples must be nonnegative; block_s must fit the minimum duration")
    covered = {observation.target_id for observation in plan}
    by_target = defaultdict(lambda: defaultdict(list))
    minimum_visible = set()
    for window in windows:
        if window.end - window.start >= rules.min_duration_s - 1e-6:
            minimum_visible.add(window.target_id)
            bins = _eligible_bins(window, block_s, horizon_s, rules.min_duration_s)
            if bins:
                by_target[window.target_id][window.station_id].append((window, bins))
    uncovered = minimum_visible - covered
    actual_metric = "coverage" if uncovered else "score" if metric == "auto" else metric
    if metric != "auto":
        actual_metric = metric
    if actual_metric not in {"coverage", "score"}:
        raise ValueError("metric must be auto, coverage or score")
    eligible = sorted(
        target for target, stations in by_target.items()
        if len(stations) >= 2 and (actual_metric == "score" or target not in covered)
    )
    rng = random.Random(seed)
    selected = rng.sample(eligible, min(samples, len(eligible)))
    draws = []
    for target in selected:
        stations = rng.sample(sorted(by_target[target]), 2)
        blocks, anchors = {}, []
        for station in sorted(stations):
            window, bins = rng.choice(by_target[target][station])
            bin_index = rng.choice(bins)
            block = (bin_index * block_s, min((bin_index + 1) * block_s, horizon_s))
            blocks[station] = block
            anchors.append({
                "target_id": target, "station_id": station,
                "window_start": window.start, "window_end": window.end,
                "bin_index": bin_index, "eligible_bins_for_window": len(bins),
                "eligible_windows_for_station": len(by_target[target][station]),
            })
        draws.append({
            "seed_target_id": target, "blocks": blocks,
            "anchor_windows": anchors,
            "station_count_for_target": len(by_target[target]),
        })
    keys = {tuple((s, *interval) for s, interval in sorted(d["blocks"].items())) for d in draws}
    return {
        "metric": actual_metric, "draws": draws,
        "sampling": {
            "strategy": (
                "uniform eligible target IDs without replacement, then uniform "
                "station pair, uniform qualifying physical window per station, "
                "uniform eligible canonical disjoint time bin"
            ),
            "seed": seed, "block_s": block_s, "requested_samples": samples,
            "eligible_seed_targets": len(eligible),
            "draw_count": len(draws), "distinct_block_pairs": len(keys),
            "targets_with_minimum_window": len(minimum_visible),
            "covered_targets": len(covered),
            "uncovered_targets_with_minimum_window": len(uncovered),
            "canonical_bins_exclude_cross_boundary_only_opportunities": True,
            "limitation": (
                "Seed-target sample; station-pair/block distribution is not uniform. "
                "Canonical block boundaries can freeze useful crossing tasks. "
                "Repeated block pairs and all solver outcomes remain in the denominator."
            ),
        },
    }


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


def resource_relaxation(windows, priorities, rules):
    """Optimistic individual-target bound in the supplied physical-window model."""
    if rules.max_observations_per_target != 1:
        raise ValueError("Resource relaxation currently supports one observation per target")
    longest = {target_id: 0.0 for target_id in priorities}
    for window in windows:
        longest[window.target_id] = max(
            longest[window.target_id], window.end - window.start
        )
    scores = {}
    for target_id, duration in longest.items():
        if duration + 1e-6 < rules.min_duration_s:
            fraction = 0.0
        elif duration + 1e-6 >= rules.full_duration_s:
            fraction = 1.0
        elif rules.score_mode == "linear":
            fraction = 0.6 + 0.4 * (
                (duration - rules.min_duration_s)
                / (rules.full_duration_s - rules.min_duration_s)
            )
        else:
            fraction = 0.6
        scores[target_id] = priorities[target_id] * fraction
    return {
        "coverage_upper": sum(value + 1e-6 >= rules.min_duration_s for value in longest.values()),
        "observation_score_upper": sum(scores.values()),
        "full_priority_sum": sum(priorities.values()),
        "targets_without_full_duration_window": {
            str(target): {
                "priority": priorities[target], "maximum_window_s": duration,
                "score_upper": scores[target],
            }
            for target, duration in longest.items()
            if duration + 1e-6 < rules.full_duration_s
        },
        "scope": (
            "Independent best feasible duration for each target in the supplied "
            "window model, ignoring all station competition; one observation "
            "per target. This upper bound is not a physical-model error certificate."
        ),
    }


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_delta(repair, initial, windows, priorities, rules, horizon_s, geometry):
    initial_set = set(initial)
    added = [observation for observation in repair["plan"] if observation not in initial_set]
    check = validate_plan(added, windows, priorities, rules, horizon_s, geometry)
    repair["new_observation_validation"] = check
    repair["geometry_validation_scope"] = (
        "Full initial plan checked once; each returned plan structurally checked "
        "by the independent validator, and every changed/new observation "
        "independently propagated here."
    )
    if not check["valid"]:
        repair["plan"] = list(initial)
        repair["gain_lower"] = 0.0
        repair["final_metrics"] = repair["initial_metrics"]
        repair["optimal"] = False
        repair["candidate_rejected_by_geometry"] = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-seconds", type=float, default=1800)
    parser.add_argument("--time-limit", type=float, default=2)
    parser.add_argument("--max-options", type=int, default=400)
    parser.add_argument("--metric", choices=("auto", "coverage", "score"), default="auto")
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument(
        "--reference", action="store_true",
        help="Release every configured station for the whole horizon in one CP reference solve",
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    # Shared CLI helpers preserve exactly the instance's original target draw.
    from space_planner.cli import load_config, select_dataset

    config = load_config(args.config)
    dataset = load_dataset(Path(config["data_root"]))
    targets, stations = select_dataset(dataset, config)
    priorities = {target.target_id: target.priority for target in targets}
    rules = Rules(**config.get("rules", {}))
    saved_windows = json.loads(args.windows.read_text())
    saved_plan = json.loads(args.plan.read_text())
    windows = [Window(**value) for value in saved_windows["windows"]]
    plan = [Observation(**value) for value in saved_plan["plan"]]
    horizon_s = float(saved_windows["horizon_s"])
    start_utc = datetime.fromisoformat(saved_windows["start_utc"].replace("Z", "+00:00"))
    configured_start = datetime.fromisoformat(config["start_utc"].replace("Z", "+00:00"))
    if start_utc != configured_start:
        raise ValueError("Window epoch disagrees with the experiment config")
    expected_horizon = float(config.get("horizon_s", float(config.get("days", 6)) * 86400))
    if abs(horizon_s - expected_horizon) > 1e-6:
        raise ValueError("Window horizon disagrees with the experiment config")
    geometry = None
    if not args.skip_geometry:
        from space_planner.geometry import Geometry
        geometry = Geometry(targets, stations, start_utc)
    initial_check = validate_plan(plan, windows, priorities, rules, horizon_s, geometry)
    if not initial_check["valid"]:
        raise ValueError(f"Initial plan failed validation: {initial_check['violations'][:3]}")
    sampled = sample_blocks(
        windows, plan, rules, horizon_s, samples=args.samples, seed=args.seed,
        block_s=args.block_seconds, metric=args.metric,
    )
    output = {
        "source": "real organizer TLE-derived windows; configured development instance",
        "inputs": {
            str(path.resolve()): _sha256(path)
            for path in (args.config, args.windows, args.plan)
        },
        "source_files": {
            str(path.resolve()): _sha256(path)
            for path in (
                Path(__file__), REPO_ROOT / "src/space_planner/local_repair.py",
                REPO_ROOT / "src/space_planner/validation.py",
                REPO_ROOT / "src/space_planner/geometry.py",
            )
        },
        "parameters": vars(args) | {
            "config": str(args.config), "windows": str(args.windows),
            "plan": str(args.plan), "output": str(args.output),
        },
        "rules": asdict(rules), "horizon_s": horizon_s,
        "start_utc": start_utc.isoformat(),
        "sampling": sampled["sampling"], "metric": sampled["metric"],
        "resource_relaxation": resource_relaxation(windows, priorities, rules),
        "initial_validation": initial_check, "results": [],
        "learning_model_trained": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.reference:
        output["sampling"] = {
            "strategy": "One complete supplied-window reference solve",
            "released_station_count": len(stations),
            "seed": args.seed,
        }
        reference = solve_local_repair(
            plan, windows, priorities, rules, horizon_s,
            {station.station_id: (0.0, horizon_s) for station in stations},
            time_limit_s=args.time_limit, max_options=args.max_options,
            seed=args.seed, metric=sampled["metric"],
        )
        if geometry is not None:
            _verify_delta(reference, plan, windows, priorities, rules, horizon_s, geometry)
        output["reference"] = reference
        output["summary"] = {
            "status": reference["status"],
            "gain_lower": reference["gain_lower"], "gain_upper": reference["gain_upper"],
            "initial_metrics": reference["initial_metrics"],
            "final_metrics": reference["final_metrics"],
            "optimal_in_restricted_model": reference["optimal"],
        }
        output["wall_time_s"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, default=_json_default, indent=2) + "\n")
        print(json.dumps(output["summary"]), flush=True)
        return 0
    for draw_index, draw in enumerate(sampled["draws"]):
        result = probe_joint_repairs(
            plan, windows, priorities, rules, horizon_s, [draw["blocks"]],
            time_limit_s=args.time_limit, max_options=args.max_options,
            seed=args.seed, metric=sampled["metric"],
        )[0]
        if geometry is not None:
            for repair in [*result["singles"], result["joint"]]:
                _verify_delta(repair, plan, windows, priorities, rules, horizon_s, geometry)
            result["interaction_lower"] = result["joint"]["gain_lower"] - sum(
                single["gain_upper"] for single in result["singles"]
            )
            result["interaction_upper"] = result["joint"]["gain_upper"] - sum(
                single["gain_lower"] for single in result["singles"]
            )
            lower, upper = result["interaction_lower"], result["interaction_upper"]
            result["certified_positive_interaction_in_restricted_model"] = lower > 1e-6
            result["interaction_label"] = (
                "positive" if lower > 1e-6 else "nonpositive" if upper <= 1e-6 else "uncertain"
            )
        result["sample"] = draw
        output["results"].append(result)
        # Save every draw, including skipped/time-limited/no-gain outcomes.
        output["wall_time_s"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, default=_json_default, indent=2) + "\n")
        print(json.dumps({
            "draw": draw_index + 1, "seed_target_id": draw["seed_target_id"],
            "statuses": [x["status"] for x in [*result["singles"], result["joint"]]],
            "joint_gain": [result["joint"]["gain_lower"], result["joint"]["gain_upper"]],
            "interaction": [result["interaction_lower"], result["interaction_upper"]],
        }), flush=True)
    repairs = [
        repair for result in output["results"] for repair in [*result["singles"], result["joint"]]
    ]
    output["summary"] = {
        "sampled_pairs": len(output["results"]),
        "solver_calls": len(repairs),
        "status_counts": dict(Counter(repair["status"] for repair in repairs)),
        "interaction_label_counts": dict(Counter(
            result["interaction_label"] for result in output["results"]
        )),
        "joint_positive_gain_pairs": sum(
            result["joint"]["gain_lower"] > 1e-6 for result in output["results"]
        ),
        "single_positive_gain_calls": sum(
            repair["gain_lower"] > 1e-6
            for result in output["results"] for repair in result["singles"]
        ),
        "new_geometry_rejections": sum(
            bool(repair.get("candidate_rejected_by_geometry")) for repair in repairs
        ),
    }
    output["wall_time_s"] = time.monotonic() - started
    args.output.write_text(json.dumps(output, default=_json_default, indent=2) + "\n")
    print(json.dumps(output["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
