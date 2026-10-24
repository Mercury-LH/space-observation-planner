"""档位必须保持已发布求解配置及证据边界，不运行全量实验。"""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from obsplan.config import Config
from obsplan.effort import EFFORT_ORDER, EFFORT_PROFILES, get_effort, spec_for_effort
from obsplan import portfolio


RESULTS = Path(__file__).resolve().parents[1] / "results/time-aware-2026-10-24"
RUN_PATHS = {
    "instant": "confirmation/confirmation-v4/002-hplus_1_fast-r1/result.json",
    "medium": "confirmation/confirmation-v4/003-hplus_10_fast-r1/result.json",
    "high": "exploration/iteration-v2/003-hplus_30_fast-r1/result.json",
    "xhigh": "exploration/iteration-v2/004-hplus_75_fast-r1/result.json",
    "max": "confirmation/confirmation-v4/004-hplus_300_full-r1/result.json",
    "ultra": "confirmation/confirmation-v4/001-fusion_baseline-r1/result.json",
}


def test_controlled_metadata_matches_published_cohort_without_mixing_power_modes():
    report = json.loads((RESULTS / "summary.json").read_text())
    published = {row["preset"]: row for row in report["methods"]}
    controlled = [get_effort(name) for name in EFFORT_ORDER
                  if get_effort(name).evidence_status == "controlled"]
    assert {p.key for p in controlled} == {"instant", "medium", "max", "ultra"}
    for profile in controlled:
        row = published[profile.benchmark_name]
        assert row["rankable"] and row["pareto"]
        assert profile.reference_quality == row["quality_median"]
        assert profile.reference_covered == row["covered_median"]
        assert profile.reference_solver_s == row["solver_s_median"]
        assert profile.reference_total_s == row["timing_s_median"]["total"]
        assert profile.reference_total_s > profile.reference_solver_s


@pytest.mark.parametrize("name", ["high", "xhigh"])
def test_exploratory_quality_does_not_invent_comparable_runtime(name):
    profile = get_effort(name)
    run = json.loads((RESULTS / RUN_PATHS[name]).read_text())
    assert run["validation"]["legal"] and run["validation"]["geometry_checked"]
    assert profile.evidence_status == "exploratory"
    assert profile.reference_quality == run["metrics"]["combined"]
    assert profile.reference_covered == run["metrics"]["covered"]
    assert profile.reference_solver_s is None
    assert profile.reference_total_s is None


@pytest.mark.parametrize("name", EFFORT_ORDER)
def test_resolved_solver_config_matches_actual_published_run(name):
    # Compare against recorded complete configurations, not just our own presets.
    # One worker avoids requiring the 8-thread historical hardware in CI.
    expected = json.loads((RESULTS / RUN_PATHS[name]).read_text())["actual_solver_config"]
    expected["workers"] = 1
    pb = SimpleNamespace(cfg=Config(), m_min=4, m_max=8)
    actual = portfolio._config(spec_for_effort(name, workers=1), pb)
    assert actual == expected


@pytest.mark.parametrize("name", EFFORT_ORDER)
def test_demo_subset_disables_full_dataset_near_window_insertion(name):
    full = spec_for_effort(name)
    subset = spec_for_effort(name, full_problem=False)
    assert full["finish_near"] is True
    assert subset.pop("finish_near") is False
    assert subset == {k: v for k, v in full.items() if k != "finish_near"}


def test_profile_and_nested_solver_specs_do_not_leak_mutation_between_ui_reruns():
    original = deepcopy(portfolio.PRESETS)
    spec = spec_for_effort("ultra", seed=7, workers=1)
    spec["fusion"]["durations"].append(999)
    spec["sources"].clear()
    spec["ga"]["pop"] = 100
    clean = spec_for_effort("ultra", seed=7, workers=1)
    assert clean["sources"] == ["G+", "E", "H+"]
    assert 999 not in clean["fusion"]["durations"]
    assert clean["ga"]["pop"] == 16
    assert portfolio.PRESETS == original
    with pytest.raises(FrozenInstanceError):
        get_effort("instant").h_iters = 300
    with pytest.raises(TypeError):
        EFFORT_PROFILES["instant"] = get_effort("max")


@pytest.mark.parametrize("name", ["", "Instant", "unknown", None, []])
def test_unknown_effort_cannot_silently_fall_back_to_another_solver(name):
    with pytest.raises(ValueError, match="未知计算档位"):
        get_effort(name)
    with pytest.raises(ValueError, match="未知计算档位"):
        spec_for_effort(name)


@pytest.mark.parametrize("kwargs", [
    {"seed": -1}, {"seed": 2**31}, {"seed": True}, {"seed": 1.2},
    {"workers": 0}, {"workers": False}, {"workers": 1.5},
    {"full_problem": 0}, {"full_problem": "False"},
])
def test_invalid_execution_controls_fail_before_solver_initialization(kwargs):
    with pytest.raises(ValueError):
        spec_for_effort("instant", **kwargs)
