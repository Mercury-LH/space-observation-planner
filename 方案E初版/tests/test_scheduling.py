"""Synthetic scheduling tests; these are not contest performance evidence."""
from itertools import combinations

import pytest

from space_planner.scheduling import (
    alns_schedule,
    greedy_schedule,
    objective_metrics,
)
from space_planner.types import Observation, Rules, Window


def assert_feasible(plan, windows, rules):
    """Test oracle deliberately does not call scheduler feasibility helpers."""
    for item in plan:
        assert item.end - item.start >= rules.min_duration_s - 1e-6
        assert any(
            window.station_id == item.station_id
            and window.target_id == item.target_id
            and window.start <= item.start + 1e-6
            and item.end <= window.end + 1e-6
            for window in windows
        )
    for first, second in combinations(plan, 2):
        if first.station_id == second.station_id:
            first, second = sorted((first, second), key=lambda item: item.start)
            switch = rules.switch_s if first.target_id != second.target_id else 0
            assert second.start + 1e-6 >= first.end + switch
        if first.target_id == second.target_id and rules.repeat_gap_s is not None:
            assert abs(first.start - second.start) + 1e-6 >= rules.repeat_gap_s
    for target in {item.target_id for item in plan}:
        assert sum(item.target_id == target for item in plan) <= rules.max_observations_per_target


@pytest.mark.parametrize("strategy", ["scarcity", "priority", "earliest", "regret", "portfolio"])
def test_continuous_non_grid_boundaries_and_switch(strategy):
    windows = [
        Window(1, "A", 0.37, 120.37),
        Window(2, "A", 150.37, 270.37),
        Window(3, "A", 299.0, 419.0),
        Window(4, "B", 1.13, 241.13),
    ]
    rules = Rules()
    plan = greedy_schedule(windows, {1: 5, 2: 3, 3: 2, 4: 5}, rules, strategy=strategy)
    assert_feasible(plan, windows, rules)
    assert {item.target_id for item in plan} == {1, 2, 4}
    assert objective_metrics(plan, {1: 5, 2: 3, 3: 2, 4: 5}, rules)["observation_score"] == pytest.approx(9.8)


def test_coverage_before_extension():
    windows = [Window(1, "A", 0, 270), Window(2, "A", 0, 270)]
    rules = Rules()
    plan = greedy_schedule(windows, {1: 5, 2: 3}, rules)
    assert_feasible(plan, windows, rules)
    assert len(plan) == 2
    assert all(item.end - item.start == 120 for item in plan)


def test_score_first_portfolio_explores_full_duration_tradeoff():
    windows = [Window(1, "A", 0, 270), Window(2, "A", 0, 270)]
    priorities = {1: 5, 2: 2}
    coverage_plan = greedy_schedule(windows, priorities, Rules())
    rules = Rules(objective="score_then_coverage")
    score_plan = greedy_schedule(windows, priorities, rules)
    assert len(coverage_plan) == 2
    assert len(score_plan) == 1
    assert objective_metrics(score_plan, priorities, rules)["observation_score"] == 5
    assert_feasible(score_plan, windows, rules)


@pytest.mark.parametrize("score_mode,expected_duration", [("step", 120), ("linear", 180)])
def test_configurable_interior_duration_credit(score_mode, expected_duration):
    rules = Rules(score_mode=score_mode)
    windows = [Window(1, "A", 0, 180)]
    plan = greedy_schedule(windows, {1: 5}, rules)
    assert plan[0].end - plan[0].start == expected_duration
    assert_feasible(plan, windows, rules)


def test_all_alternative_windows_are_available():
    # The only useful final opportunity is deliberately the last of many. A
    # candidate-count cap or earliest-window-only filter would lose target 99.
    windows = [Window(i, "A", i * 150, i * 150 + 120) for i in range(20)]
    windows += [Window(99, "A", i * 150, i * 150 + 120) for i in range(21)]
    priorities = {i: 5 for i in range(20)} | {99: 2}
    plan = greedy_schedule(windows, priorities, Rules(), strategy="scarcity")
    assert len(plan) == 21
    assert next(item for item in plan if item.target_id == 99).start == 3000


def test_cross_station_joint_repair_synthetic_fixture():
    windows = [
        Window(1, "S1", 0, 120), Window(1, "S2", 0, 120),
        Window(2, "S2", 0, 270), Window(3, "S1", 0, 120),
    ]
    initial = [Observation(1, "S1", 0, 120), Observation(2, "S2", 0, 120)]
    priorities = {1: 5, 2: 5, 3: 5}
    rules = Rules()
    result, diagnostics = alns_schedule(
        windows, priorities, rules, initial=initial, iterations=30, seed=7,
        destroy_fraction=(1, 1),
    )
    assert_feasible(result, windows, rules)
    assert len(result) == 3
    assert diagnostics["best_metrics"]["coverage_count"] == 3
    assert diagnostics["completed_iterations"] == 30
    assert all(value > 0 for value in diagnostics["destroy_uses"].values())
    assert all(value > 0 for value in diagnostics["repair_uses"].values())


def test_repeats_use_start_to_start_across_stations():
    windows = [Window(1, "A", 0, 700), Window(1, "B", 0, 700)]
    rules = Rules(max_observations_per_target=3, repeat_gap_s=300)
    plan = greedy_schedule(windows, {1: 5}, rules)
    assert_feasible(plan, windows, rules)
    assert len(plan) == 2
    assert objective_metrics(plan, {1: 5}, rules)["coverage_count"] == 1
    assert objective_metrics(plan, {1: 5}, rules)["observation_score"] == 10


def test_repeats_can_overlap_across_stations_if_not_forbidden():
    windows = [Window(1, "A", 0, 120), Window(1, "B", 0, 120)]
    rules = Rules(max_observations_per_target=2, repeat_gap_s=None)
    plan = greedy_schedule(windows, {1: 5}, rules)
    assert len(plan) == 2
    assert_feasible(plan, windows, rules)


def test_alns_preserves_best_and_seed_reproducibility():
    windows = [
        Window(target, station, 0.25, 900.25)
        for target in range(9)
        for station in ("A", "B")
    ]
    priorities = {target: (5 if target % 2 else 2) for target in range(9)}
    rules = Rules()
    initial = greedy_schedule(windows, priorities, rules)
    first, diagnostics = alns_schedule(windows, priorities, rules, initial=initial, iterations=20, seed=9)
    second, other = alns_schedule(windows, priorities, rules, initial=initial, iterations=20, seed=9)
    assert first == second
    assert [entry.get("destroy") for entry in diagnostics["trace"]] == [
        entry.get("destroy") for entry in other["trace"]
    ]
    assert diagnostics["best_metrics"]["coverage_count"] >= len(initial)
    assert diagnostics["neutral_accepted_moves"] > 0
    assert_feasible(first, windows, rules)


def test_invalid_initial_and_zero_budget():
    windows = [Window(1, "A", 0, 120), Window(2, "A", 0, 120)]
    with pytest.raises(ValueError, match="station gap"):
        alns_schedule(
            windows, {1: 5, 2: 3}, Rules(),
            initial=[Observation(1, "A", 0, 120), Observation(2, "A", 0, 120)],
        )
    initial = [Observation(1, "A", 0, 120)]
    result, diagnostics = alns_schedule(
        windows, {1: 5, 2: 3}, Rules(), initial=initial, time_limit_s=0
    )
    assert result == initial
    assert diagnostics["completed_iterations"] == 0


def test_empty_problem_and_bad_windows():
    assert greedy_schedule([], {}, Rules()) == []
    result, diagnostics = alns_schedule([], {}, Rules(), iterations=2)
    assert result == []
    assert diagnostics["best_metrics"]["coverage_count"] == 0
    with pytest.raises(ValueError, match="finite"):
        greedy_schedule([Window(1, "A", float("nan"), 120)], {1: 5}, Rules())
    with pytest.raises(ValueError, match="finite"):
        greedy_schedule([], {}, Rules(), time_limit_s=float("nan"))


def test_duplicate_window_representation_does_not_change_result():
    windows = [Window(1, "A", 0, 270), Window(2, "A", 0, 270)]
    assert greedy_schedule(windows, {1: 5, 2: 3}, Rules()) == greedy_schedule(
        windows * 4, {1: 5, 2: 3}, Rules()
    )


def test_randomized_plans_pass_independent_pairwise_oracle():
    import random

    rng = random.Random(713)
    for case in range(20):
        windows = []
        priorities = {target: rng.choice((2, 3, 5)) for target in range(12)}
        for target in priorities:
            for _ in range(rng.randint(1, 6)):
                start = rng.uniform(0, 800)
                windows.append(
                    Window(target, rng.choice(("A", "B", "C")), start, start + rng.uniform(100, 500))
                )
        rules = Rules(
            score_mode="linear" if case % 2 else "step",
            objective="score_then_coverage" if case % 3 else "coverage_then_score",
            max_observations_per_target=2 if case % 4 else 1,
            repeat_gap_s=300 if case % 4 else None,
        )
        initial = greedy_schedule(windows, priorities, rules, seed=case)
        plan, _ = alns_schedule(windows, priorities, rules, initial=initial, iterations=5, seed=case)
        assert_feasible(plan, windows, rules)


def test_regret_cache_matches_unconditional_recomputation(monkeypatch):
    """The acceleration must preserve the original greedy decisions exactly."""
    import random
    import space_planner.scheduling as scheduling

    rng = random.Random(149)
    windows = []
    priorities = {target: rng.choice((2, 3, 5)) for target in range(120)}
    for target in priorities:
        for _ in range(12):
            start = rng.uniform(0, 2500)
            windows.append(
                Window(target, rng.choice(("A", "B", "C")), start, start + rng.uniform(120, 600))
            )
    for rules in (
        Rules(),
        Rules(max_observations_per_target=2, repeat_gap_s=600, score_mode="linear"),
    ):
        accelerated = scheduling.greedy_schedule(
            windows, priorities, rules, strategy="regret", seed=7
        )
        with monkeypatch.context() as patch:
            patch.setattr(scheduling, "_insertion_affects_target", lambda *args: True)
            reference = scheduling.greedy_schedule(
                windows, priorities, rules, strategy="regret", seed=7
            )
        assert accelerated == reference
        assert_feasible(accelerated, windows, rules)


def test_portfolio_reports_all_actual_constructions_and_shared_budget():
    diagnostics = {}
    windows = [Window(1, "A", 0, 270), Window(2, "A", 0, 270)]
    plan = greedy_schedule(windows, {1: 5, 2: 2}, Rules(), diagnostics=diagnostics)
    assert len(plan) == 2
    assert diagnostics["started_strategy_count"] == 5
    assert diagnostics["completed_strategy_count"] == 5
    assert diagnostics["elapsed_s"] >= sum(
        run["elapsed_s"] for run in diagnostics["strategy_runs"]
    )
    zero_budget = {}
    assert greedy_schedule(
        windows, {1: 5, 2: 2}, Rules(), time_limit_s=0, diagnostics=zero_budget
    ) == []
    assert zero_budget["started_strategy_count"] == 0
    assert zero_budget["completed_strategy_count"] == 0


def test_long_windows_skip_only_provably_too_short_station_gaps():
    # A day-long opportunity spans 50 existing observations. Only one interior
    # gap can fit the new task; both horizon-end gaps are deliberately too short.
    initial = [Observation(target, "A", target * 150, target * 150 + 120) for target in range(50)]
    initial = [item for item in initial if item.target_id != 27]
    windows = [Window(item.target_id, "A", item.start, item.end) for item in initial]
    windows.append(Window(100, "A", 0, 7470))
    priorities = {item.target_id: 5 for item in initial} | {100: 3}
    plan, _ = alns_schedule(
        windows, priorities, Rules(), initial=initial, iterations=1,
        destroy_fraction=(0.01, 0.01), seed=4,
    )
    assert_feasible(plan, windows, Rules())
    inserted = next(item for item in plan if item.target_id == 100)
    assert inserted.start == 4050
    assert len(plan) == 50
