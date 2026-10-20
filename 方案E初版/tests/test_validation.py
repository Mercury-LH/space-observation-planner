"""Adversarial validator checks independent of the scheduler."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from space_planner.validation import validate_plan


@dataclass(frozen=True)
class Interval:
    target_id: int
    station_id: str
    start: float
    end: float


def rules(**overrides):
    values = dict(
        min_duration_s=120, switch_s=30,
        max_observations_per_target=1, repeat_gap_s=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def codes(result):
    return {v["code"] for v in result["violations"]}


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -1, True])
@pytest.mark.parametrize("parameter", ["time_tolerance_s", "elevation_tolerance_deg"])
def test_invalid_tolerance_cannot_bypass_constraints(parameter, value):
    plan = [Interval(1, "S", 0, 120), Interval(2, "S", 0, 120)]
    with pytest.raises(ValueError, match="finite and nonnegative"):
        validate_plan(plan, plan, {1: 5, 2: 3}, rules(), 300, **{parameter: value})


@pytest.mark.parametrize("target", [True, [], {}, "1", 1.0])
def test_malformed_target_identifier_is_structured_violation(target):
    windows = [Interval(1, "S", 0, 300)]
    plan = [{"target_id": target, "station_id": "S", "start": 0, "end": 120}]
    result = validate_plan(plan, windows, {1: 5}, rules(), 300)
    assert not result["valid"]
    assert "invalid_target_id" in codes(result)


def test_malformed_station_identifier_is_structured_violation():
    windows = [Interval(1, "S", 0, 300)]
    result = validate_plan(
        [{"target_id": 1, "station_id": [], "start": 0, "end": 120}],
        windows, {1: 5}, rules(), 300,
    )
    assert not result["valid"] and "invalid_station_id" in codes(result)


def test_exact_boundaries_and_switch_are_accepted():
    plan = [Interval(1, "S", 0, 120), Interval(2, "S", 150, 270)]
    result = validate_plan(plan, plan, {1: 5, 2: 3}, rules(), 270)
    assert result["valid"]
    assert not result["geometry_checked"]


def test_two_windows_do_not_allow_an_observation_across_a_gap():
    windows = [Interval(1, "S", 0, 100), Interval(1, "S", 101, 300)]
    result = validate_plan([Interval(1, "S", 50, 200)], windows, {1: 5}, rules(), 300)
    assert "outside_visibility_window" in codes(result)


def test_nan_unknown_and_short_outside_tasks_are_rejected():
    windows = [Interval(1, "S", 0, 300)]
    plan = [
        Interval(1, "S", float("nan"), 200),
        Interval(9, "X", -1, 4),
    ]
    result = validate_plan(plan, windows, {1: 5}, rules(), 300)
    assert codes(result) >= {
        "non_finite_time", "unknown_target", "unknown_station",
        "duration_too_short", "outside_horizon", "outside_visibility_window",
    }


def test_long_overlap_is_detected_beyond_adjacent_pair():
    plan = [
        Interval(1, "S", 0, 500),
        Interval(2, "S", 50, 170),
        Interval(3, "S", 200, 320),
    ]
    result = validate_plan(plan, plan, {1: 5, 2: 3, 3: 2}, rules(), 500)
    overlaps = [
        v["observation_indices"] for v in result["violations"]
        if v["code"] == "station_overlap"
    ]
    assert [0, 2] in overlaps


def test_same_target_needs_no_switch_but_cannot_overlap_on_station():
    plan = [Interval(1, "S", 0, 120), Interval(1, "S", 120, 240)]
    result = validate_plan(
        plan, [Interval(1, "S", 0, 240)], {1: 5},
        rules(max_observations_per_target=2), 240,
    )
    assert result["valid"]
    plan[1] = Interval(1, "S", 119, 240)
    assert "station_overlap" in codes(validate_plan(
        plan, [Interval(1, "S", 0, 240)], {1: 5},
        rules(max_observations_per_target=2), 240,
    ))


def test_cross_station_repeat_limit_and_start_gap():
    plan = [Interval(1, "A", 0, 120), Interval(1, "B", 500, 620)]
    result = validate_plan(plan, plan, {1: 5}, rules(repeat_gap_s=600), 1000)
    assert codes(result) >= {"target_observation_limit", "repeat_gap_too_short"}


class SyntheticGeometry:
    stations = [SimpleNamespace(
        station_id="S", min_elevation_deg=10, max_elevation_deg=85
    )]

    def elevation(self, target_id, station_id, seconds):
        seconds = np.asarray(seconds)
        # Peak 85.02 degrees at a non-grid time; 5-second samples miss 85.
        return 85.02 - 0.01 * (seconds - 61.234) ** 2


def test_upper_elevation_peak_between_grid_samples_is_rejected():
    plan = [Interval(1, "S", 0, 120)]
    result = validate_plan(
        plan, plan, {1: 5}, rules(), 120, geometry=SyntheticGeometry()
    )
    assert "elevation_above_maximum" in codes(result)
    assert result["structural_valid"]
    assert result["geometry_observations_checked"] == 1


def test_non_finite_geometry_fails_closed():
    geometry = SyntheticGeometry()
    geometry.elevation = lambda target, station, times: np.full_like(
        np.asarray(times, dtype=float), np.nan
    )
    plan = [Interval(1, "S", 0, 120)]
    result = validate_plan(plan, plan, {1: 5}, rules(), 120, geometry=geometry)
    assert "geometry_error" in codes(result)
