"""Opt-in portfolio-compatible solver; existing presets are untouched."""
from __future__ import annotations

from time import perf_counter

import numba

from .. import lagrange
from ..evolved import load_rule
from ..lns import state_from_schedule
from ..local_search import improve
from ..refine import finish
from .features import FeatureContext
from .search import check_schedule, copy_schedule, metrics, schedule_hash, search


def initialize(pb, spec, progress=None):
    """Return fixed H+ GRID schedule, its prices, and compact provenance."""
    started = perf_counter()
    iterations = int(spec.get("initial_h_iters", 300))
    fast = bool(spec.get("initial_h_fast", False))
    if iterations < 1:
        raise ValueError("initial_h_iters must be positive")
    rule = load_rule()
    if rule is None:
        raise RuntimeError("the adopted H+ rule is required")
    previous_threads = numba.get_num_threads()
    workers = int(spec.get("workers", previous_threads))
    if not 1 <= workers <= numba.config.NUMBA_NUM_THREADS:
        raise ValueError("workers outside NUMBA_NUM_THREADS")
    numba.set_num_threads(workers)
    try:
        schedule, details = lagrange.solve(pb, iters=iterations, fast=fast, rule=rule,
                                           verbose=False, name="H+/fixed-grid")
    finally:
        numba.set_num_threads(previous_threads)
    if check_schedule(pb, schedule):
        raise RuntimeError("fixed H+ failed public structural validation")
    info = {"wall_s": perf_counter() - started, "metrics": metrics(pb, schedule),
            "initial_plan_hash": schedule_hash(schedule), "iterations": iterations,
            "fast": fast, "local_adopted_rule": True,
            "details": {k: v for k, v in details.items() if k != "mu"}}
    if progress is not None:
        progress({"event": "complete", "stage": "hplus_initial", "elapsed_s": info["wall_s"]})
    return schedule, details["mu"], info


def solve(pb, spec, progress=None):
    if not spec.get("enabled", False):
        from ..portfolio import solve as original
        return original(pb, spec.get("baseline_spec", {"preset": "hplus_fast"}), progress=progress)
    started = perf_counter()
    budget = float(spec.get("budget_s", 0))
    if not 0 < budget < float("inf"):
        raise ValueError("enabled plugin requires a finite calibrated budget_s")
    deadline = started + budget
    reserve = float(spec.get("finish_reserve_s", 0.05))
    if not 0 <= reserve < budget:
        raise ValueError("finish_reserve_s must fit inside the calibrated budget")
    search_deadline = deadline - reserve
    mode = spec.get("mode", "rule")
    if mode not in ("original_hplus", "rule", "random", "neural", "static_only"):
        raise ValueError("unknown comparison arm")
    old_threads = numba.get_num_threads()
    workers = int(spec.get("workers", 1))
    if not 1 <= workers <= numba.config.NUMBA_NUM_THREADS:
        raise ValueError("workers outside NUMBA_NUM_THREADS")
    timings = {}; details = {}; grid = None
    numba.set_num_threads(workers)
    try:
        t0 = perf_counter(); grid, mu, initial_info = initialize(pb, spec, progress)
        timings["hplus_initial"] = perf_counter() - t0
        initial = copy_schedule(grid)
        ranker = context = None
        if mode in ("neural", "static_only") and perf_counter() < search_deadline:
            from .inference import Ranker
            expected = {"feature_version": "neural-lns-features-v1", "pool_version": "full-windows-v1",
                        "price_version": "hplus-fixed-mu-v1"}
            t0 = perf_counter(); ranker = Ranker(spec["checkpoint"], device=spec.get("device", "cpu"),
                                                expected_sha256=spec.get("checkpoint_sha256"),
                                                expected_versions=expected,
                                                allow_static_domain_transfer=mode == "static_only")
            timings["model_load"] = perf_counter() - t0
            t0 = perf_counter(); context = FeatureContext(pb, mu)
            timings["feature_context"] = perf_counter() - t0
            details["features"] = context.fingerprint
            details["model"] = {"sha256": ranker.sha256, "domain_transfer": ranker.domain_transfer}
        t0 = perf_counter()
        if mode == "original_hplus":
            rounds = []
            for _ in range(int(spec.get("max_rounds", 256))):
                if perf_counter() >= search_deadline:
                    break
                state = state_from_schedule(pb, grid)
                before = metrics(pb, grid)
                stats = improve(pb, state, rounds=1, repeat_min_priority=3.0 if spec.get("initial_h_fast", False) else None)
                candidate = state.schedule(pb, "H+/extended-grid", 0)
                after = metrics(pb, candidate)
                good = not check_schedule(pb, candidate) and after["C"] >= before["C"] and after["Q"] > before["Q"] + 1e-12
                accepted = bool(good and perf_counter() < search_deadline)
                rounds.append({"before": before, "after": after, "accepted": accepted, "stats": stats})
                if accepted:
                    grid = candidate
                # Continue original search to its fixed point, not expensive
                # repeated calls that deterministically reproduce no change.
                if not good:
                    break
            details["original_continuation"] = rounds
        elif perf_counter() < search_deadline:
            grid, details["search"] = search(pb, grid, deadline=search_deadline, mode=mode,
                                               seed=int(spec.get("seed", 2026)),
                                               max_rounds=int(spec.get("max_rounds", 256)),
                                               ranker=ranker, feature_context=context)
        timings["candidate_search"] = perf_counter() - t0
        # Final refinement cost is real. If it crosses the deadline, retain
        # the latest valid grid and mark the whole attempt over budget.
        final = grid
        if perf_counter() < deadline:
            t0 = perf_counter(); candidate, finish_info = finish(pb, grid, near=bool(spec.get("finish_near", False)), verbose=False)
            timings["integer_second_refinement"] = perf_counter() - t0
            details["finish"] = finish_info
            candidate_metrics, grid_metrics = metrics(pb, candidate), metrics(pb, grid)
            if (perf_counter() < deadline and candidate_metrics["C"] >= grid_metrics["C"]
                    and candidate_metrics["Q"] >= grid_metrics["Q"] - 1e-12):
                final = candidate
        runtime = perf_counter() - started
        final.algo = "neural-lns/" + mode
        final.runtime_s = runtime
        info = {"actual_config": {"enabled": True, "mode": mode, "budget_s": budget,
                                  "initial_h_iters": int(spec.get("initial_h_iters", 300)),
                                  "initial_h_fast": bool(spec.get("initial_h_fast", False)),
                                  "workers": workers, "seed": int(spec.get("seed", 2026)),
                                  "max_rounds": int(spec.get("max_rounds", 256)),
                                  "finish_near": bool(spec.get("finish_near", False)),
                                  "finish_reserve_s": reserve,
                                  "device": spec.get("device", "cpu"), "checkpoint": spec.get("checkpoint")},
                "runtime_s": runtime, "stage_timings_s": timings, "initial": metrics(pb, initial),
                "final": metrics(pb, final), "initial_plan_hash": schedule_hash(initial),
                "final_plan_hash": schedule_hash(final), "initialization": initial_info,
                "details": details, "over_budget": runtime > budget,
                "budget_semantics": "deadline_checked_between_calls; overrun_is_failed_attempt; latest_legal_grid_returned",
                "independent_geometry_checked": False, "requires_independent_skyfield_1s": True,
                "famou": {"mode": "local_adopted_rule", "runtime_cloud_calls": 0}}
        return final, info
    finally:
        numba.set_num_threads(old_threads)
