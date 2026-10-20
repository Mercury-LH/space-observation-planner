from pathlib import Path
import runpy

from space_planner.types import Observation, Rules, Window

probe_module = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "probe_real.py")
)
sample_blocks = probe_module["sample_blocks"]
resource_relaxation = probe_module["resource_relaxation"]


def test_sampling_is_deterministic_and_accounts_for_population():
    windows = [
        Window(target, station, 0, 1800)
        for target in range(1, 9) for station in ("A", "B", "C")
    ]
    plan = [Observation(1, "A", 0, 120)]
    first = sample_blocks(windows, plan, Rules(), 1800, samples=3, seed=7, block_s=600)
    second = sample_blocks(windows, plan, Rules(), 1800, samples=3, seed=7, block_s=600)
    assert first == second
    assert first["metric"] == "coverage"
    assert first["sampling"]["eligible_seed_targets"] == 7
    assert first["sampling"]["draw_count"] == 3
    assert len({draw["seed_target_id"] for draw in first["draws"]}) == 3
    for draw in first["draws"]:
        assert len(draw["blocks"]) == 2
        for start, end in draw["blocks"].values():
            assert start % 600 == 0
            assert end - start == 600


def test_full_coverage_selects_score_and_empty_population_is_reported():
    windows = [Window(1, station, 0, 600) for station in ("A", "B")]
    plan = [Observation(1, "A", 0, 120)]
    result = sample_blocks(windows, plan, Rules(), 600, samples=12, block_s=600)
    assert result["metric"] == "score"
    assert result["sampling"]["requested_samples"] == 12
    assert result["sampling"]["draw_count"] == 1
    result = sample_blocks(windows, plan, Rules(), 600, metric="coverage")
    assert result["draws"] == []
    assert result["sampling"]["eligible_seed_targets"] == 0


def test_boundary_only_opportunities_are_explicitly_excluded():
    windows = [Window(1, station, 550, 670) for station in ("A", "B")]
    result = sample_blocks(windows, [], Rules(), 1200, samples=12, block_s=600)
    assert result["sampling"]["uncovered_targets_with_minimum_window"] == 1
    assert result["sampling"]["eligible_seed_targets"] == 0
    assert result["sampling"]["canonical_bins_exclude_cross_boundary_only_opportunities"]


def test_resource_relaxation_uses_window_lengths_and_rule_choice():
    windows = [
        Window(1, "A", 0, 240), Window(2, "A", 0, 180),
        Window(3, "A", 0, 119),
    ]
    step = resource_relaxation(windows, {1: 5, 2: 5, 3: 5}, Rules())
    linear = resource_relaxation(
        windows, {1: 5, 2: 5, 3: 5}, Rules(score_mode="linear")
    )
    assert step["coverage_upper"] == 2
    assert step["observation_score_upper"] == 8
    assert linear["observation_score_upper"] == 9
    assert step["full_priority_sum"] == 15
