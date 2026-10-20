"""Small CP-SAT repair probes with explicit, restricted-model bounds.

Every supplied physical window intersecting a released station/time block is
considered. Starts are integer seconds and durations are minimum/full, plus
exact incumbent anchors. These bounds do NOT upper-bound the original
continuous-time problem or its intermediate-duration alternatives.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import time
from typing import Any, Iterable, Mapping, Sequence

from ortools.sat.python import cp_model

from .types import Observation, Rules, Window
from .validation import validate_plan

_EPS = 1e-6  # Match the independent validator's time feasibility tolerance.
_SCORE_EPS = 1e-7
_SCORE_SCALE = 1_000_000


def _score(observation: Observation, priorities: Mapping[int, int], rules: Rules) -> float:
    duration = observation.end - observation.start
    if duration + _SCORE_EPS >= rules.full_duration_s:
        fraction = 1.0
    elif rules.score_mode == "linear":
        fraction = 0.6 + 0.4 * (
            (duration - rules.min_duration_s)
            / (rules.full_duration_s - rules.min_duration_s)
        )
    else:
        fraction = 0.6
    return priorities[observation.target_id] * fraction


def _metrics(plan: Sequence[Observation], priorities: Mapping[int, int], rules: Rules):
    return {
        "coverage": len({observation.target_id for observation in plan}),
        "score": sum(_score(observation, priorities, rules) for observation in plan),
    }


@dataclass
class _Option:
    target_id: int
    station_id: str
    minimum: int
    maximum: int
    offset: float
    duration: float
    anchor: Observation | None = None
    start_var: Any = None
    present: Any = None

    @property
    def earliest(self) -> float:
        return self.minimum + self.offset

    @property
    def latest_end(self) -> float:
        return self.maximum + self.offset + self.duration


def _ordering_bound(first: _Option, second: _Option, gap: float) -> int:
    # x_first + offset_first + duration_first + gap <= x_second + offset_second
    # x are integers; this preserves exact incumbent fractional-time offsets.
    return math.floor(
        second.offset - first.offset - first.duration - gap + _EPS
    )


def _integral_service(option: _Option, gap: float) -> bool:
    """When true, the integer NoOverlap interval is exact, not a relaxation."""
    return float(option.offset).is_integer() and float(option.duration + gap).is_integer()


def solve_local_repair(
    plan: Iterable[Observation],
    windows: Iterable[Window],
    priorities: Mapping[int, int],
    rules: Rules,
    horizon_s: float,
    blocks: Mapping[str, tuple[float, float]],
    *,
    time_limit_s: float = 2.0,
    max_options: int = 400,
    seed: int = 0,
    metric: str = "coverage",
    geometry: Any = None,
) -> dict[str, Any]:
    """Repair within non-overlapping station blocks, retaining the initial plan.

    A task is released only when fully contained in its station block. Tasks
    crossing block boundaries stay fixed. Returned gain bounds concern this
    exact restricted candidate model; timeout is never interpreted as zero
    possible gain. ``max_options`` skips oversized probes instead of pruning.
    """
    if rules.max_observations_per_target != 1:
        raise ValueError("Local probe currently supports max_observations_per_target=1 only")
    if metric not in {"coverage", "score"}:
        raise ValueError("metric must be coverage or score")
    if time_limit_s <= 0 or not math.isfinite(time_limit_s):
        raise ValueError("time_limit_s must be finite and positive")
    initial, all_windows = list(plan), list(windows)
    initial_validation = validate_plan(
        initial, all_windows, priorities, rules, horizon_s, geometry
    )
    if not initial_validation["valid"]:
        raise ValueError(f"Initial plan is invalid: {initial_validation['violations'][:3]}")
    for station_id, (start, end) in blocks.items():
        if not all(math.isfinite(float(x)) for x in (start, end)) or not 0 <= start < end <= horizon_s:
            raise ValueError(f"Invalid block at {station_id}: {(start, end)}")
    started = time.monotonic()
    fixed, released = [], []
    for observation in initial:
        block = blocks.get(observation.station_id)
        if block and observation.start >= block[0] - _EPS and observation.end <= block[1] + _EPS:
            released.append(observation)
        else:
            fixed.append(observation)
    fixed_targets = {observation.target_id for observation in fixed}
    options: list[_Option] = []
    seen: set[tuple[Any, ...]] = set()
    for observation in released:
        option = _Option(
            observation.target_id, observation.station_id, 0, 0,
            observation.start, observation.end - observation.start, observation,
        )
        options.append(option)
    for window in all_windows:
        block = blocks.get(window.station_id)
        if block is None or window.target_id in fixed_targets:
            continue
        start, end = max(window.start, block[0]), min(window.end, block[1])
        for duration in sorted({rules.min_duration_s, rules.full_duration_s}):
            minimum, maximum = math.ceil(start - _EPS), math.floor(end - duration + _EPS)
            key = window.target_id, window.station_id, minimum, maximum, duration
            if maximum >= minimum and key not in seen:
                seen.add(key)
                options.append(_Option(
                    window.target_id, window.station_id, minimum, maximum,
                    0.0, duration,
                ))
    initial_metrics = _metrics(initial, priorities, rules)
    fixed_metrics = _metrics(fixed, priorities, rules)
    initial_value = initial_metrics[metric]
    per_target_values: dict[int, float] = {}
    for option in options:
        value = 1.0 if metric == "coverage" else _score(
            Observation(option.target_id, option.station_id, 0, option.duration),
            priorities, rules,
        )
        per_target_values[option.target_id] = max(
            per_target_values.get(option.target_id, 0.0), value
        )
    effective_switch = max(0.0, rules.switch_s - _EPS)
    effective_minimum = max(rules.min_duration_s - _EPS, rules.min_duration_s / 2)
    station_capacities = {
        station: math.floor(
            (end - start + effective_switch + 2 * _EPS)
            / (effective_minimum + effective_switch)
        )
        for station, (start, end) in blocks.items()
    }
    # Capacity ignores crossing fixed tasks and inter-target/window competition,
    # so it is a safe optimistic bound. It does not remove any candidates.
    maximum_count = sum(station_capacities.values())
    independent_upper = sum(
        sorted(per_target_values.values(), reverse=True)[:maximum_count]
    )
    trivial_upper_gain = max(
        0.0, fixed_metrics[metric] + independent_upper - initial_value
    )
    common = {
        "metric": metric,
        "blocks": {key: list(value) for key, value in sorted(blocks.items())},
        "options": len(options),
        "candidate_targets": len(per_target_values),
        "released_observations": len(released),
        "fixed_observations": len(fixed),
        "initial_metrics": initial_metrics,
        "time_limit_s": time_limit_s,
        "seed": seed,
        "model_scope": (
            "all supplied windows clipped to declared station blocks; integer-second "
            "starts, minimum/full durations, plus exact initial anchors; outside and "
            "boundary-crossing tasks fixed; one observation per target"
        ),
        "bound_scope": (
            "gain bounds apply only to this restricted local model; feasible gains "
            "are lower bounds on continuous local potential after validation, but "
            "reported upper bounds are NOT continuous-time upper bounds"
        ),
        "score_rounding_unit": 1 / _SCORE_SCALE if metric == "score" else 0,
        "station_capacity_upper": station_capacities,
    }
    if len(options) > max_options:
        return {
            **common, "status": "SKIPPED_OPTION_LIMIT", "plan": initial,
            "final_metrics": initial_metrics, "gain_lower": 0.0,
            "gain_upper": trivial_upper_gain, "optimal": False,
            "wall_time_s": time.monotonic() - started,
        }
    model = cp_model.CpModel()
    by_target: dict[int, list[_Option]] = defaultdict(list)
    by_station: dict[str, list[_Option]] = defaultdict(list)
    for index, option in enumerate(options):
        option.start_var = model.NewIntVar(option.minimum, option.maximum, f"start_{index}")
        option.present = model.NewBoolVar(f"present_{index}")
        by_target[option.target_id].append(option)
        by_station[option.station_id].append(option)
        model.AddHint(option.present, int(option.anchor is not None))
        if option.anchor is not None:
            model.AddHint(option.start_var, 0)
    for target_options in by_target.values():
        model.AddAtMostOne(option.present for option in target_options)
    for station_id, station_options in by_station.items():
        model.Add(
            sum(option.present for option in station_options)
            <= station_capacities[station_id]
        )
        intervals = []
        for index, option in enumerate(station_options):
            # This inner integer interval is contained in the true occupied
            # [start, end + switch] interval. Thus NoOverlap is a safe redundant
            # relaxation for fractional anchors, and exact for integer service.
            offset = math.ceil(option.offset)
            size = math.floor(option.offset + option.duration + rules.switch_s) - offset
            if size > 0:
                intervals.append(model.NewOptionalFixedSizeIntervalVar(
                    option.start_var + offset, size, option.present,
                    f"resource_{station_id}_{index}",
                ))
        for index, observation in enumerate(x for x in fixed if x.station_id == station_id):
            start = math.ceil(observation.start)
            size = math.floor(observation.end + rules.switch_s) - start
            if size > 0:
                intervals.append(model.NewFixedSizeIntervalVar(
                    start, size, f"fixed_resource_{station_id}_{index}"
                ))
        model.AddNoOverlap(intervals)
    ordering_constraints = 0
    for station_id, station_options in by_station.items():
        for i, first in enumerate(station_options):
            for second in station_options[i + 1:]:
                if first.target_id == second.target_id:
                    continue  # At-most-one already excludes this pair.
                gap = rules.switch_s
                if first.latest_end + gap <= second.earliest + _EPS or second.latest_end + gap <= first.earliest + _EPS:
                    continue
                if _integral_service(first, gap) and _integral_service(second, gap):
                    continue  # Exactly represented by native NoOverlap above.
                first_bound = _ordering_bound(first, second, gap)
                second_bound = _ordering_bound(second, first, gap)
                first_possible = first.minimum - second.maximum <= first_bound
                second_possible = second.minimum - first.maximum <= second_bound
                together = [first.present, second.present]
                if not first_possible and not second_possible:
                    model.AddAtMostOne(together)
                elif not second_possible:
                    model.Add(first.start_var - second.start_var <= first_bound).OnlyEnforceIf(together)
                elif not first_possible:
                    model.Add(second.start_var - first.start_var <= second_bound).OnlyEnforceIf(together)
                else:
                    order = model.NewBoolVar(f"order_{ordering_constraints}")
                    model.Add(first.start_var - second.start_var <= first_bound).OnlyEnforceIf([*together, order])
                    model.Add(second.start_var - first.start_var <= second_bound).OnlyEnforceIf([*together, order.Not()])
                ordering_constraints += 1
        for fixed_observation in (x for x in fixed if x.station_id == station_id):
            anchor = _Option(
                fixed_observation.target_id, station_id, 0, 0,
                fixed_observation.start,
                fixed_observation.end - fixed_observation.start,
            )
            for option in station_options:
                gap = rules.switch_s
                if option.latest_end + gap <= anchor.earliest + _EPS or anchor.latest_end + gap <= option.earliest + _EPS:
                    continue
                if _integral_service(option, gap) and _integral_service(anchor, gap):
                    continue
                before_bound = _ordering_bound(option, anchor, gap)
                after_bound = _ordering_bound(anchor, option, gap)
                before_possible = option.minimum <= before_bound
                after_possible = -option.maximum <= after_bound
                if not before_possible and not after_possible:
                    model.Add(option.present == 0)
                elif not after_possible:
                    model.Add(option.start_var <= before_bound).OnlyEnforceIf(option.present)
                elif not before_possible:
                    model.Add(-option.start_var <= after_bound).OnlyEnforceIf(option.present)
                else:
                    order = model.NewBoolVar(f"fixed_order_{ordering_constraints}")
                    model.Add(option.start_var <= before_bound).OnlyEnforceIf([option.present, order])
                    model.Add(-option.start_var <= after_bound).OnlyEnforceIf([option.present, order.Not()])
                ordering_constraints += 1
    weights = [
        1 if metric == "coverage" else round(_SCORE_SCALE * _score(
            Observation(option.target_id, option.station_id, 0, option.duration),
            priorities, rules,
        )) for option in options
    ]
    rounding_by_target: dict[int, float] = {}
    if metric == "score":
        for option, weight in zip(options, weights):
            actual = _score(
                Observation(option.target_id, option.station_id, 0, option.duration),
                priorities, rules,
            )
            rounding_by_target[option.target_id] = max(
                rounding_by_target.get(option.target_id, 0.0),
                actual - weight / _SCORE_SCALE,
            )
    model.Maximize(sum(weight * option.present for weight, option in zip(weights, options)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    candidate = list(initial)
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        candidate = list(fixed)
        for option in options:
            if solver.BooleanValue(option.present):
                start = solver.Value(option.start_var) + option.offset
                candidate.append(option.anchor or Observation(
                    option.target_id, option.station_id, start, start + option.duration
                ))
    validation = validate_plan(candidate, all_windows, priorities, rules, horizon_s, geometry)
    candidate_metrics = _metrics(candidate, priorities, rules)
    if not validation["valid"] or candidate_metrics[metric] < initial_value - _EPS:
        candidate, candidate_metrics = initial, initial_metrics
    # Keep the incumbent when the primary diagnostic metric ties but its other
    # metric is worse. This does not alter the solver's primary-metric bound.
    secondary = "score" if metric == "coverage" else "coverage"
    if abs(candidate_metrics[metric] - initial_value) <= _EPS and candidate_metrics[secondary] < initial_metrics[secondary]:
        candidate, candidate_metrics = initial, initial_metrics
    gain_lower = max(0.0, candidate_metrics[metric] - initial_value)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE, cp_model.UNKNOWN):
        # UNKNOWN may report a vacuous zero bound before presolve has initialized.
        raw_bound = solver.BestObjectiveBound()
        if status == cp_model.UNKNOWN and raw_bound == 0 and options:
            gain_upper = trivial_upper_gain
        else:
            scale = 1 if metric == "coverage" else _SCORE_SCALE
            rounding_allowance = sum(rounding_by_target.values())
            bound = raw_bound / scale + fixed_metrics[metric] + rounding_allowance
            gain_upper = min(trivial_upper_gain, max(gain_lower, bound - initial_value))
    else:
        # With exact anchors, INFEASIBLE implies a bug/model mismatch. Never
        # manufacture a zero-gain label from a failed model.
        gain_upper = trivial_upper_gain
    return {
        **common, "status": status_name, "plan": sorted(candidate, key=lambda x: (x.start, x.station_id, x.target_id)),
        "final_metrics": candidate_metrics, "gain_lower": gain_lower,
        "gain_upper": gain_upper,
        "optimal": status == cp_model.OPTIMAL and validation["valid"],
        "solver_candidate_validation": validation,
        "solver_wall_time_s": solver.WallTime(),
        "wall_time_s": time.monotonic() - started,
        "ordering_constraints": ordering_constraints,
        "branches": solver.NumBranches(), "conflicts": solver.NumConflicts(),
    }


def probe_joint_repairs(
    plan: Iterable[Observation],
    windows: Iterable[Window],
    priorities: Mapping[int, int],
    rules: Rules,
    horizon_s: float,
    block_pairs: Iterable[Mapping[str, tuple[float, float]]],
    **solve_options: Any,
) -> list[dict[str, Any]]:
    """Compare single-station and joint repairs using the same candidate universe."""
    initial, all_windows = list(plan), list(windows)
    results = []
    for pair in block_pairs:
        if len(pair) != 2:
            raise ValueError("Each probe must contain two distinct station blocks")
        singles = [
            solve_local_repair(initial, all_windows, priorities, rules, horizon_s,
                               {station: interval}, **solve_options)
            for station, interval in sorted(pair.items())
        ]
        joint = solve_local_repair(
            initial, all_windows, priorities, rules, horizon_s, pair, **solve_options
        )
        low = joint["gain_lower"] - sum(single["gain_upper"] for single in singles)
        high = joint["gain_upper"] - sum(single["gain_lower"] for single in singles)
        results.append({
            "blocks": {key: list(value) for key, value in sorted(pair.items())},
            "singles": singles, "joint": joint,
            "interaction_lower": low, "interaction_upper": high,
            "certified_positive_interaction_in_restricted_model": low > 1e-6,
            "interaction_label": "positive" if low > 1e-6 else (
                "nonpositive" if high <= 1e-6 else "uncertain"
            ),
            "limitation": (
                "This is a local restricted-model interaction diagnostic, not "
                "evidence of learned-model improvement or methodological novelty."
            ),
        })
    return results
