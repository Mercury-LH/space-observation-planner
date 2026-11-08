#!/usr/bin/env python3
"""Bounded CPU wiring smoke on a real train scene subset; never quality evidence."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from obsplan.neural_lns.counterfactual import collect_state
from obsplan.neural_lns.features import FeatureContext
from obsplan.neural_lns.plugin import initialize, solve
from obsplan.neural_lns.runtime import environment, file_hash, load_scene, source_hash, write_new
from obsplan.neural_lns.search import generate_operations, metrics, schedule_hash
from obsplan.problem import Problem
from obsplan.schedule import Schedule
from obsplan.validator import validate
from obsplan.visibility import Windows


def run(args):
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cache = out / "cache"; cache.mkdir()
    shutil.copyfile(args.ephemeris, cache / "de421.bsp")
    full, metadata = load_scene(args.dataset_root, args.catalog_id, cache, allowed_splits=("train",))
    n = min(args.targets, full.n_targets)
    ds = replace(full.ds, names=full.ds.names[:n], satnum=full.ds.satnum[:n].copy(),
                 line1=full.ds.line1[:n], line2=full.ds.line2[:n],
                 priority=full.ds.priority[:n].copy(), priority_label=full.ds.priority_label[:n])
    sel = full.w_tg < n
    win = Windows(full.w_st[sel].astype(np.int16), full.w_tg[sel].astype(np.int32),
                  full.w_k0[sel].astype(np.int32), full.w_k1[sel].astype(np.int32),
                  full.win.dark, full.n_samples, full.win.sgp4_failed[:n].copy())
    pb = Problem.build(full.cfg, ds, win)
    write_new(out / "config.json", {**vars(args), "dev_smoke": True, "quality_evidence": False,
             "dataset_metadata": metadata, "target_indices": list(range(n)),
             "window_count": len(pb.w_st), "geometry_recomputed": False,
             "ephemeris_sha256": file_hash(cache / "de421.bsp"), "code_hash": source_hash()})
    write_new(out / "environment.json", environment("local-cpu-development-smoke"))
    initial, mu, initial_info = initialize(pb, {"initial_h_iters": 1, "initial_h_fast": False, "workers": 1})
    write_new(out / "initial.json", initial_info)
    runs = []
    for mode in ("rule", "random", args.learned_mode):
        directory = out / mode; directory.mkdir()
        solve_started = time.perf_counter()
        plan, info = solve(pb, {"enabled": True, "mode": mode, "budget_s": args.budget_s,
                    "initial_h_iters": 1, "initial_h_fast": False, "workers": 1, "seed": 2026,
                    "max_rounds": 8, "finish_near": False, "finish_reserve_s": 3.,
                    "device": "cpu", "checkpoint": args.checkpoint})
        solve_s = time.perf_counter() - solve_started
        np.savez_compressed(directory / "plan.npz", **{k:getattr(plan, k) for k in ("station", "target", "start_s", "dur_s")})
        write_new(directory / "solver-info.json", info)
        validation_started = time.perf_counter(); report = validate(pb.cfg, pb.ds, plan, step_s=1)
        validation_s = time.perf_counter() - validation_started
        np.savez_compressed(directory / "validation-geometry.npz", per_task=report["per_task"], invalid_mask=report["invalid_mask"])
        public_report = {k:v for k,v in report.items() if k not in ("per_task", "invalid_mask")}
        public_report.update(engine="skyfield", step_s=1, endpoints=True)
        write_new(directory / "validation.json", public_report)
        row = {"mode": mode, "metrics": metrics(pb, plan), "solve_s": solve_s,
               "validation_s": validation_s, "legal": bool(report["legal"] and report["geometry_checked"]),
               "over_budget": bool(info["over_budget"]), "initial_plan_hash": info["initial_plan_hash"],
               "plan_sha256": file_hash(directory / "plan.npz"), "quality_evidence": False}
        runs.append(row)
        print(json.dumps(row), flush=True)
    # Explicitly weakened developer-only repair budget. These observations
    # exercise real operation labels but may NEVER enter the formal pilot.
    empty = Schedule(*(np.empty(0, np.int64) for _ in range(4)))
    context = FeatureContext(pb, mu)
    rows = []
    for rd in range(pb.n_stations * 36):
        operations, _ = generate_operations(pb, empty, round_index=rd, limit=4)
        if operations:
            rows = collect_state(pb, empty, context, catalog_id=args.catalog_id,
                    scene_id=metadata["scene_id"], split="train", initial_plan_hash=schedule_hash(initial),
                    seed=2026, round_index=rd, max_candidates=4, max_trials=0, timeout_s=2.,
                    code_fingerprint=source_hash(), dependency_fingerprint="see environment.json")
            break
    with (out / "counterfactual-dev-smoke.jsonl").open("x") as stream:
        for row in rows:
            row.update(dev_smoke=True, quality_evidence=False, target_subset=list(range(n)),
                       deviation="empty parent, first 32 train targets, H+1 iteration and zero repair trials; formal protocol remains H+300/64 trials")
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {"runs": runs, "all_plans_independently_legal": all(r["legal"] for r in runs),
               "same_initial_plan": len({r["initial_plan_hash"] for r in runs}) == 1,
               "counterfactual_rows": len(rows), "counterfactual_statuses": [r["status"] for r in rows],
               "wall_s": time.perf_counter()-started, "dev_smoke": True, "quality_evidence": False,
               "formal_100_operation_calibration_run": False, "formal_1_percent_gate_passed": False}
    write_new(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--catalog-id", default="v2-cat-342")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--learned-mode", choices=("static_only", "neural"), default="static_only")
    parser.add_argument("--ephemeris", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--targets", type=int, default=32)
    parser.add_argument("--budget-s", type=float, default=20.)
    args = parser.parse_args()
    if not 1 <= args.targets <= 32 or args.budget_s <= 3:
        raise ValueError("CPU smoke permits 1–32 targets and requires >3s refinement reserve")
    try:
        run(args)
    except BaseException as exc:
        out = Path(args.output)
        if out.exists() and not (out / "failure.json").exists():
            write_new(out / "failure.json", {"status": "failed", "error": type(exc).__name__+": "+str(exc), "quality_evidence": False})
        raise


if __name__ == "__main__":
    main()
