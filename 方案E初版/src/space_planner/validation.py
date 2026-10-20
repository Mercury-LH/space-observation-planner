"""Independent schedule validation; all times are seconds after the epoch.

This module intentionally does not import scheduling/repair feasibility code.
Geometry is recomputed from the supplied propagator, never inferred from a
learned feasibility prediction or from the window file alone.
"""

from __future__ import annotations

from collections import defaultdict
import math
from numbers import Integral
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import minimize_scalar


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite_number(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        return False


def _geometry_limits(geometry: Any, station_id: str) -> tuple[float, float]:
    stations = getattr(geometry, "stations", [])
    if isinstance(stations, Mapping):
        station = stations.get(station_id)
    else:
        station = next(
            (s for s in stations if _field(s, "station_id") == station_id), None
        )
    return (
        float(_field(station, "min_elevation_deg", 10.0)),
        float(_field(station, "max_elevation_deg", 85.0)),
    )


def _elevation_extrema(
    geometry: Any,
    target_id: int,
    station_id: str,
    start: float,
    end: float,
    sample_step_s: float,
) -> tuple[float, float, int]:
    """Sample and refine smooth elevation extrema, including near endpoints.

    This is a numerical check, not an interval-arithmetic proof. The grid is
    independent of the visibility extractor's grid. Local extrema are refined
    with bounded scalar optimization; both signs are searched near endpoints.
    """
    count = max(2, int(math.ceil((end - start) / sample_step_s)) + 1)
    times = np.linspace(start, end, count)
    values = np.asarray(
        geometry.elevation(target_id, station_id, times), dtype=float
    ).reshape(-1)
    if len(values) != count or not np.isfinite(values).all():
        raise ValueError("propagator returned missing/non-finite elevations")
    low, high = float(np.min(values)), float(np.max(values))
    brackets: set[tuple[float, float, float]] = set()
    # Endpoint subintervals matter when a peak falls before the first interior
    # sample. Flat triples need no minimization unless bounded by a change.
    for left, right in ((times[0], times[1]), (times[-2], times[-1])):
        brackets.add((float(left), float(right), 1.0))
        brackets.add((float(left), float(right), -1.0))
    for i in range(1, count - 1):
        previous, current, following = values[i - 1 : i + 2]
        if current <= previous and current <= following and (
            current < previous or current < following
        ):
            brackets.add((float(times[i - 1]), float(times[i + 1]), 1.0))
        if current >= previous and current >= following and (
            current > previous or current > following
        ):
            brackets.add((float(times[i - 1]), float(times[i + 1]), -1.0))

    evaluations = count
    for left, right, sign in sorted(brackets):
        def objective(t: float) -> float:
            result = np.asarray(
                geometry.elevation(target_id, station_id, t), dtype=float
            )
            if result.size != 1 or not np.isfinite(result).all():
                raise ValueError("propagator returned a non-finite elevation")
            return sign * float(result.item())

        result = minimize_scalar(
            objective,
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1e-5, "maxiter": 80},
        )
        if not result.success or not math.isfinite(float(result.fun)):
            raise ValueError("elevation extremum refinement failed")
        value = float(result.fun) * sign
        low, high = min(low, value), max(high, value)
        evaluations += int(result.nfev)
    return low, high, evaluations


def validate_plan(
    plan: Iterable[Any],
    windows: Iterable[Any],
    priorities: Mapping[int, Any],
    rules: Any,
    horizon_s: float,
    geometry: Any = None,
    *,
    time_tolerance_s: float = 1e-6,
    elevation_tolerance_deg: float = 1e-5,
    geometry_step_s: float = 5.0,
) -> dict[str, Any]:
    """Check a plan independently and return structured violations.

    Window containment is required for the *whole* observation in one window.
    Repeat spacing, when enabled, is defined start-to-start. With no geometry,
    ``valid`` means structural feasibility only; ``geometry_checked`` is false.
    """
    observations, all_windows = list(plan), list(windows)
    violations: list[dict[str, Any]] = []
    for label, value in (
        ("time_tolerance_s", time_tolerance_s),
        ("elevation_tolerance_deg", elevation_tolerance_deg),
    ):
        if not _finite_number(value) or float(value) < 0:
            raise ValueError(f"{label} must be finite and nonnegative")
    time_tolerance_s, elevation_tolerance_deg = float(time_tolerance_s), float(elevation_tolerance_deg)
    for window in all_windows:
        target, station = _field(window, "target_id"), _field(window, "station_id")
        if (
            isinstance(target, bool) or not isinstance(target, Integral)
            or not isinstance(station, str) or not station
        ):
            raise ValueError("Window identifiers must be integer target and nonempty string station")
    indexed_windows: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
    station_ids = {_field(w, "station_id") for w in all_windows}
    geometry_stations = getattr(geometry, "stations", None)
    if geometry_stations is not None:
        if isinstance(geometry_stations, Mapping):
            station_ids.update(geometry_stations)
        else:
            station_ids.update(_field(s, "station_id") for s in geometry_stations)
    for window in all_windows:
        ws, we = _field(window, "start"), _field(window, "end")
        if _finite_number(ws) and _finite_number(we) and float(ws) < float(we):
            indexed_windows[
                (_field(window, "target_id"), _field(window, "station_id"))
            ].append((float(ws), float(we)))

    def record(code: str, indices: list[int], **details: Any) -> None:
        violations.append({"code": code, "observation_indices": indices, **details})

    if not _finite_number(horizon_s) or float(horizon_s) <= 0:
        raise ValueError("horizon_s must be finite and positive")
    if not _finite_number(geometry_step_s) or float(geometry_step_s) <= 0:
        raise ValueError("geometry_step_s must be finite and positive")
    geometry_step_s = float(geometry_step_s)
    min_duration = _field(rules, "min_duration_s", 120.0)
    switch = _field(rules, "switch_s", 30.0)
    repeat_gap = _field(rules, "repeat_gap_s")
    max_per_target = _field(rules, "max_observations_per_target", 1)
    if not _finite_number(min_duration) or float(min_duration) <= 0:
        raise ValueError("Minimum observation duration must be finite and positive")
    if not _finite_number(switch) or float(switch) < 0:
        raise ValueError("Switch duration must be finite and nonnegative")
    if repeat_gap is not None and (not _finite_number(repeat_gap) or float(repeat_gap) < 0):
        raise ValueError("Repeat gap must be finite and nonnegative or None")
    if max_per_target is not None and (
        isinstance(max_per_target, bool) or not isinstance(max_per_target, Integral)
        or max_per_target < 1
    ):
        raise ValueError("Observation limit must be a positive integer or None")
    min_duration, switch = float(min_duration), float(switch)
    station_plan: dict[str, list[tuple[float, float, int, int]]] = defaultdict(list)
    target_plan: dict[int, list[tuple[float, int]]] = defaultdict(list)
    geometry_evaluations = 0
    geometry_observations = 0
    for index, observation in enumerate(observations):
        target_id, station_id = (
            _field(observation, "target_id"), _field(observation, "station_id")
        )
        start, end = _field(observation, "start"), _field(observation, "end")
        invalid_identifier = False
        if isinstance(target_id, bool) or not isinstance(target_id, Integral):
            record("invalid_target_id", [index])
            invalid_identifier = True
        if not isinstance(station_id, str) or not station_id:
            record("invalid_station_id", [index])
            invalid_identifier = True
        if invalid_identifier:
            continue
        if target_id not in priorities:
            record("unknown_target", [index], target_id=target_id)
        if station_id not in station_ids:
            record("unknown_station", [index], station_id=station_id)
        if not _finite_number(start) or not _finite_number(end):
            record("non_finite_time", [index])
            continue
        start, end = float(start), float(end)
        duration = end - start
        if duration <= 0:
            record("non_positive_duration", [index], duration_s=duration)
            continue
        if duration + time_tolerance_s < min_duration:
            record("duration_too_short", [index], duration_s=duration)
        if start < -time_tolerance_s or end > float(horizon_s) + time_tolerance_s:
            record("outside_horizon", [index], start=start, end=end)
        eligible = indexed_windows.get((target_id, station_id), [])
        if not any(
            start >= ws - time_tolerance_s and end <= we + time_tolerance_s
            for ws, we in eligible
        ):
            record("outside_visibility_window", [index])
        station_plan[station_id].append((start, end, index, target_id))
        target_plan[target_id].append((start, index))
        if geometry is not None and target_id in priorities and station_id in station_ids:
            try:
                low, high, evaluations = _elevation_extrema(
                    geometry, target_id, station_id, start, end, geometry_step_s
                )
                lower_limit, upper_limit = _geometry_limits(geometry, station_id)
                geometry_observations += 1
                geometry_evaluations += evaluations
                if low < lower_limit - elevation_tolerance_deg:
                    record(
                        "elevation_below_minimum", [index],
                        minimum_elevation_deg=low, limit_deg=lower_limit,
                    )
                if high > upper_limit + elevation_tolerance_deg:
                    record(
                        "elevation_above_maximum", [index],
                        maximum_elevation_deg=high, limit_deg=upper_limit,
                    )
            except Exception as error:
                record("geometry_error", [index], message=str(error))

    for station_id, tasks in station_plan.items():
        ordered = sorted(tasks)
        # Do not only inspect adjacent pairs: a long overlapping observation can
        # extend past several short ones. Sweep with the furthest occupied end.
        active: list[tuple[float, float, int, int]] = []
        for current in ordered:
            start, end, index, target_id = current
            active = [
                previous for previous in active
                if previous[1] + switch > start + time_tolerance_s
            ]
            for previous in active:
                _, previous_end, previous_index, previous_target = previous
                required_gap = switch if previous_target != target_id else 0.0
                if start < previous_end - time_tolerance_s:
                    record("station_overlap", [previous_index, index],
                           station_id=station_id)
                elif start < previous_end + required_gap - time_tolerance_s:
                    record("insufficient_switch_time", [previous_index, index],
                           station_id=station_id, gap_s=start - previous_end,
                           required_gap_s=required_gap)
            active.append(current)
    for target_id, tasks in target_plan.items():
        if max_per_target is not None and len(tasks) > int(max_per_target):
            record("target_observation_limit", [i for _, i in tasks],
                   target_id=target_id, maximum=int(max_per_target))
        if repeat_gap is not None:
            ordered = sorted(tasks)
            for (first_start, first_index), (second_start, second_index) in zip(
                ordered, ordered[1:]
            ):
                if second_start - first_start < float(repeat_gap) - time_tolerance_s:
                    record("repeat_gap_too_short", [first_index, second_index],
                           target_id=target_id, start_gap_s=second_start - first_start)
    geometry_codes = {
        "geometry_error", "elevation_below_minimum", "elevation_above_maximum"
    }
    return {
        "valid": not violations,
        "structural_valid": not any(v["code"] not in geometry_codes for v in violations),
        "geometry_checked": geometry is not None,
        "geometry_observations_checked": geometry_observations,
        "geometry_evaluations": geometry_evaluations,
        "observations_checked": len(observations),
        "violations": violations,
        "repeat_gap_convention": "start-to-start",
        "time_tolerance_s": time_tolerance_s,
        "elevation_tolerance_deg": elevation_tolerance_deg,
        "geometry_method": (
            f"independent propagation, <= {geometry_step_s:g} s samples, "
            "bounded extrema refinement; numerical check, not analytic proof"
            if geometry is not None else "not performed"
        ),
    }
