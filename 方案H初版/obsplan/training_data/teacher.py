"""Auditable, inexpensive heuristic labels for independent training scenes.

These are decisions of a fixed local H+ teacher, not optimality or feasibility
labels. Candidate features describe the whole initial scene. They do not include
the changing resource occupancy of a sequential policy. The runner must validate
both returned plans independently with Skyfield before publishing a scene.
"""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

import numpy as np

from .. import lagrange
from ..checks import structural_violations
from ..evolved import load_rule
from ..famou_h_eval import FEATURES, feature_rows, run_rule
from ..greedy import candidates
from ..problem import Problem
from ..refine import finish


TEACHER_NAME = "hplus_3_no_local_search"
PRICE_ITERATIONS = 3


class TeacherBuildError(RuntimeError):
    """A failed teacher stage, with completed artifacts for the runner to retain."""

    def __init__(self, message: str, partial_result: dict):
        super().__init__(message)
        self.partial_result = partial_result


def _grid_labels(pb, cw, cm, state, grid_plan) -> dict:
    """Map State's actual window/duration decisions to candidate IDs losslessly.

One candidate can be placed more than once in an unusually long window. The
candidate-level rank/start fields then describe its first selection; the task
mapping and count retain every action. Never derive labels from refined seconds.
    """
    count = len(cw)
    if count >= np.iinfo(np.int32).max:
        raise ValueError("Candidate IDs exceed int32 capacity")
    lookup = np.full((len(pb.w_k0), pb.m_max - pb.m_min + 1), -1, np.int32)
    lookup[cw, cm - pb.m_min] = np.arange(count, dtype=np.int32)
    rows = np.flatnonzero(state.alive[:state.n])
    windows, durations = state.w[rows], state.m[rows]
    if (len(rows) != len(grid_plan)
            or np.any((windows < 0) | (windows >= len(pb.w_k0)))
            or np.any((durations < pb.m_min) | (durations > pb.m_max))):
        raise ValueError("Teacher State does not map to the grid plan")
    task_candidate = lookup[windows, durations - pb.m_min]
    if np.any(task_candidate < 0):
        raise ValueError("Teacher selected an unenumerated window/duration")
    if not (np.array_equal(grid_plan.station, pb.w_st[windows])
            and np.array_equal(grid_plan.target, pb.w_tg[windows])
            and np.array_equal(grid_plan.start_s, state.a[rows] * pb.cfg.cell_s)
            and np.array_equal(grid_plan.dur_s, durations * pb.cfg.cell_s)):
        raise ValueError("Teacher State and grid plan disagree")
    selected_count = np.bincount(task_candidate, minlength=count).astype(np.int32)
    rank = np.full(count, -1, np.int32)
    starts = np.full(count, -1, np.int64)
    seconds = np.full(count, -1, np.int32)
    for task_rank, candidate_id in enumerate(task_candidate):
        if rank[candidate_id] < 0:
            rank[candidate_id] = task_rank
            starts[candidate_id] = grid_plan.start_s[task_rank]
            seconds[candidate_id] = grid_plan.dur_s[task_rank]
    return {
        "grid_selected": selected_count > 0,
        "grid_selected_count": selected_count,
        "grid_selection_rank": rank,
        "grid_start_s": starts,
        "grid_duration_s": seconds,
        "grid_task_candidate": task_candidate,
    }


def build_teacher(pb: Problem) -> dict:
    """Construct fixed-budget labels, then independently retain the refined plan.

The three H iterations choose a price using the existing recovery objective;
that best price can still be all zero. No price-convergence claim is made.
Existing H recovery is repeated to obtain its State and candidate IDs. Both
recoveries and all feature construction are included in the measured timings.

No local search, near-window insertion, validation, or network service runs here.
The caller owns thread settings and independent geometry validation. Exceptions
are not replaced with a different teacher or a new random scene.
    """
    cfg = pb.cfg
    if (cfg.cell_s != 30 or cfg.switch_s != 30 or cfg.obs_min_s != 120
            or cfg.obs_max_s != 240 or cfg.repeat_gap_h != 48
            or not 0 < cfg.days <= 6 or pb.n_targets <= 0):
        raise ValueError("Teacher requires the shared 30s/120-240s/48h rules and a 0-6 day horizon")
    rule = load_rule()
    if rule is None:
        raise RuntimeError("The adopted local recovery rule is missing")
    started = perf_counter()
    timings: dict[str, float] = {}
    result: dict = {"timings": timings}

    @contextmanager
    def stage(name):
        before = perf_counter()
        try:
            yield
        finally:
            timings[name] = perf_counter() - before

    try:
        with stage("price_fit_and_initial_recovery_s"):
            _, price_info = lagrange.solve(
                pb, iters=PRICE_ITERATIONS, local_search=False, rule=rule,
                verbose=False, name=TEACHER_NAME,
            )
            mu = np.asarray(price_info["mu"], dtype=np.float64)
            if mu.shape != (pb.n_stations, pb.n_samples + 2) or not np.isfinite(mu).all():
                raise ValueError("Invalid price matrix returned by H")
        with stage("candidate_features_and_rule_s"):
            cw, cm = candidates(pb)
            if not len(cw):
                raise ValueError("Scene has no teacher candidates")
            rows = feature_rows(pb, mu, cw, cm)
            if rows.shape != (len(cw), len(FEATURES)) or not np.isfinite(rows).all():
                raise ValueError("Teacher features must be finite candidate-by-20 rows")
            features = rows.astype(np.float32)
            if not np.isfinite(features).all():
                raise ValueError("Teacher features overflow float32")
            # Match the existing local rule's float64 inputs. Stored float32
            # features are a compact training representation, not bitwise replay.
            uncovered, covered = [np.asarray(v, dtype=np.float64) for v in rule(rows)]
            if any(v.shape != (len(cw),) or not np.isfinite(v).all() for v in (uncovered, covered)):
                raise ValueError("Local rule returned invalid candidate scores")
            result.update({
                "features": features,
                "candidate_window": cw.astype(np.int32),
                "candidate_duration_cells": cm.astype(np.int16),
                "rule_score_uncovered": uncovered,
                "rule_score_covered": covered,
            })
        with stage("state_recovery_and_grid_labels_s"):
            state = run_rule(pb, mu, cw, cm, uncovered, covered, local_search=False)
            grid = state.schedule(pb, TEACHER_NAME, perf_counter() - started)
            result["grid_plan"] = grid
            violations = structural_violations(pb, grid)
            if violations:
                raise ValueError(f"Teacher grid is structurally invalid: {violations}")
            result.update(_grid_labels(pb, cw, cm, state, grid))
        with stage("integer_second_refinement_s"):
            final, finish_info = finish(pb, grid, near=False, verbose=False)
            result["final_plan"] = final
            result["finish_info"] = finish_info
            if not finish_info.get("refine", {}).get("refined", False):
                raise RuntimeError(f"Integer-second refinement failed: {finish_info}")
            if not (np.array_equal(final.station, grid.station)
                    and np.array_equal(final.target, grid.target)):
                raise ValueError("Refinement unexpectedly changed teacher task identities")
            if any(not np.isfinite(values).all() or np.any(values != np.round(values))
                   for values in (final.start_s, final.dur_s)):
                raise ValueError("Refined teacher times must be finite integer seconds")
        timings["total_s"] = perf_counter() - started
        final.runtime_s = timings["total_s"]
        result["metadata"] = {
            "teacher": TEACHER_NAME,
            "price_iterations_requested": PRICE_ITERATIONS,
            "price_iterations_completed": len(price_info.get("history", [])),
            "price_nonzero_cells": int(np.count_nonzero(mu)),
            "price_max": float(mu.max()),
            "price_convergence_claimed": False,
            "price_selection": "best recovered manual-rule objective in the fixed H iteration budget",
            "duplicate_state_recovery_counted": True,
            "local_search": False,
            "finish_near": False,
            "feature_names": list(FEATURES),
            "feature_context": "static whole-scene features at the teacher-selected H price",
            "feature_storage": "float32; existing rule evaluated float64 rounded-to-4-decimal feature rows",
            "label_semantics": {
                "grid_selected": "chosen at least once by this heuristic; false does not mean infeasible or inferior",
                "grid_selection_rank": "zero-based first insertion rank; -1 means not selected",
                "grid_start_s": "first grid placement start; -1 means not selected",
                "grid_duration_s": "first grid placement duration; -1 means not selected",
                "grid_task_candidate": "candidate ID of each grid_plan row in insertion order",
                "grid_selected_count": "number of placements; preserves repeats within a long window",
                "rule_score_covered": "raw covered score; recovery clamps this to min(uncovered, covered)",
            },
            "optimality_claimed": False,
            "sequential_state_features_included": False,
            "validation_status": "not_run; runner must validate grid and final independently at 1 second",
            "famou": {"mode": "local_adopted_rule", "runtime_cloud_calls": 0},
            "timing_scope": "all H iterations, recoveries, features, labels and refinement; excludes windows and validation",
        }
        return result
    except Exception as exc:
        timings["total_s"] = perf_counter() - started
        raise TeacherBuildError(str(exc), result) from exc
