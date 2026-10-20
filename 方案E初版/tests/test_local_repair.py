from itertools import product
import math
import random

import pytest

from space_planner.local_repair import probe_joint_repairs, solve_local_repair
from space_planner.types import Observation, Rules, Window
from space_planner.validation import validate_plan


def test_synthetic_joint_unlock_has_certified_unit_interaction():
    windows = [
        Window(1, "S1", 0, 120), Window(1, "S2", 0, 120),
        Window(2, "S2", 0, 270), Window(3, "S1", 0, 120),
    ]
    initial = [Observation(1, "S1", 0, 120), Observation(2, "S2", 0, 120)]
    result = probe_joint_repairs(
        initial, windows, {1: 5, 2: 5, 3: 5}, Rules(), 300,
        [{"S1": (0, 300), "S2": (0, 300)}], time_limit_s=2,
    )[0]
    assert [x["gain_lower"] for x in result["singles"]] == [0, 0]
    assert result["joint"]["gain_lower"] == 1
    assert result["interaction_lower"] == result["interaction_upper"] == 1
    assert result["certified_positive_interaction_in_restricted_model"]
    assert validate_plan(
        result["joint"]["plan"], windows, {1: 5, 2: 5, 3: 5}, Rules(), 300
    )["valid"]


def test_fractional_initial_tasks_are_retained_without_grid_rounding():
    windows = [
        Window(1, "S", 0.123456789, 120.123456789),
        Window(2, "S", 150.123456789, 270.123456789),
    ]
    initial = [Observation(w.target_id, w.station_id, w.start, w.end) for w in windows]
    result = solve_local_repair(
        initial, windows, {1: 5, 2: 3}, Rules(), 300, {"S": (0, 300)}
    )
    assert result["status"] == "OPTIMAL"
    assert result["plan"] == initial
    assert result["gain_lower"] == result["gain_upper"] == 0


def test_crossing_boundary_task_remains_fixed_and_blocks_switch():
    initial = [Observation(1, "S", 0, 120)]
    windows = [Window(1, "S", 0, 120), Window(2, "S", 140, 260)]
    result = solve_local_repair(
        initial, windows, {1: 5, 2: 5}, Rules(), 300, {"S": (50, 300)}
    )
    assert result["fixed_observations"] == 1
    assert result["plan"] == initial
    assert result["gain_upper"] == 0


def test_option_cap_skips_without_pruning_or_false_zero_label():
    windows = [Window(i, "S", 0, 600) for i in range(1, 5)]
    result = solve_local_repair(
        [], windows, {i: 5 for i in range(1, 5)}, Rules(), 600,
        {"S": (0, 600)}, max_options=2,
    )
    assert result["status"] == "SKIPPED_OPTION_LIMIT"
    assert result["options"] == 8
    assert result["gain_lower"] == 0
    assert result["gain_upper"] == 4
    assert not result["optimal"]


def test_fractional_fixed_anchor_blocks_an_illegal_integer_start():
    initial = [Observation(1, "S", 0.1, 120.1)]
    windows = [Window(1, "S", 0.1, 120.1), Window(2, "S", 150, 270)]
    result = solve_local_repair(
        initial, windows, {1: 5, 2: 5}, Rules(), 300, {"S": (130, 300)}
    )
    # Candidate at t=150 has only 29.9 seconds switching; it must be rejected.
    assert result["gain_lower"] == result["gain_upper"] == 0
    assert result["plan"] == initial


def test_local_cp_matches_exhaustive_small_integer_schedules():
    """Compare against independent exhaustive placement, including boundaries."""
    rng = random.Random(31)
    rules = Rules(min_duration_s=2, full_duration_s=4, switch_s=1)
    for _ in range(12):
        windows = []
        choices = []
        for target in range(1, 4):
            target_choices = [None]
            for station in ("A", "B"):
                start = rng.randint(0, 6) + rng.choice([0.0, 0.2])
                end = min(12.0, start + rng.randint(2, 6))
                windows.append(Window(target, station, start, end))
                for t in range(math.ceil(start), math.floor(end - 2) + 1):
                    target_choices.append(Observation(target, station, t, t + 2))
            choices.append(target_choices)
        exact_coverage = 0
        for combination in product(*choices):
            plan = [x for x in combination if x is not None]
            feasible = True
            for station in ("A", "B"):
                ordered = sorted(
                    [x for x in plan if x.station_id == station], key=lambda x: x.start
                )
                if any(b.start < a.end + 1 for a, b in zip(ordered, ordered[1:])):
                    feasible = False
            if feasible:
                exact_coverage = max(exact_coverage, len(plan))
        result = solve_local_repair(
            [], windows, {1: 5, 2: 3, 3: 2}, rules, 12,
            {"A": (0, 12), "B": (0, 12)}, time_limit_s=2,
        )
        assert result["optimal"]
        assert result["gain_lower"] == result["gain_upper"] == exact_coverage


def test_score_probe_has_exact_endpoint_score_bound():
    initial = [Observation(1, "S", 0, 120), Observation(2, "S", 150, 270)]
    windows = [Window(1, "S", 0, 300), Window(2, "S", 0, 300)]
    result = solve_local_repair(
        initial, windows, {1: 5, 2: 3}, Rules(), 300,
        {"S": (0, 300)}, metric="score",
    )
    assert result["optimal"]
    assert result["gain_lower"] == pytest.approx(0.2)
    assert result["gain_upper"] == pytest.approx(0.2)
