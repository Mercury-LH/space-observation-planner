"""UI effort routing uses real catalog specs, with no optimization or astronomy."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from obsplan import app_backend as app, portfolio, refine, validator
from obsplan.config import Config
from obsplan.data import Dataset
from obsplan.effort import EFFORT_ORDER, get_effort, spec_for_effort
from obsplan.schedule import Schedule


@pytest.fixture
def execution(monkeypatch):
    ds = Dataset(names=["synthetic"], satnum=np.array([1]), line1=["fake1"], line2=["fake2"],
                 priority=np.array([5.]), priority_label=["高"], st_lon=np.array([0.]),
                 st_lat=np.array([0.]), st_alt_m=np.array([0.]), issues=[])
    pb = SimpleNamespace(cfg=Config(seed=17), ds=ds)
    state = SimpleNamespace(pb=pb, clock=0., calls=[], legal=True, geometry_checked=True,
                            failure=None, spec=None)
    original_time = app.time
    monkeypatch.setattr(app, "time", SimpleNamespace(
        perf_counter=lambda: state.clock, strftime=original_time.strftime))

    def prepare(problem):
        assert problem is pb
        state.calls.append("near")
        state.clock += 2

    def solve(problem, spec, progress=None):
        assert problem is pb
        state.calls.append("solve")
        state.spec = deepcopy(spec)
        state.clock += 7
        for event in [
            {"event": "start", "stage": "hplus"},
            {"event": "progress", "stage": "hplus", "fraction": .8},
            {"event": "progress", "stage": "hplus", "fraction": .1},
            {"event": "complete", "stage": "hplus"},
            {"event": "start", "stage": "finish"},
            {"event": "complete", "stage": "finish"},
        ]:
            progress(event)
        schedule = Schedule(np.array([0]), np.array([0]), np.array([0]), np.array([120]))
        return schedule, {"actual_config": deepcopy(spec), "stage_timings_s": {"hplus": 6., "finish": 1.},
                          "famou": {"mode": "local_adopted_rule", "runtime_cloud_calls": 0},
                          "finish": {"refine": {"refined": True}, "insert": {"added": 0}}}

    def check(cfg, dataset, schedule, *, step_s):
        assert cfg is pb.cfg and dataset is pb.ds
        assert step_s == 1.0
        np.testing.assert_array_equal(schedule.start_s, [0])
        state.calls.append("validate")
        state.clock += 3
        if state.failure:
            raise state.failure
        return {"legal": state.legal, "geometry_checked": state.geometry_checked,
                "violations": {} if state.legal else {"仰角低于下限": 1},
                "min_elev_margin": 2. if state.geometry_checked else None,
                "max_elev_margin": 3. if state.geometry_checked else None,
                "sun_margin": 4. if state.geometry_checked else None,
                "per_task": np.array([[12., 82., -10.]])}

    def forbidden_finish(*args, **kwargs):
        pytest.fail("run_effort must not refine a portfolio result a second time")

    monkeypatch.setattr(refine, "near_windows", prepare)
    monkeypatch.setattr(refine, "finish", forbidden_finish)
    monkeypatch.setattr(portfolio, "solve", solve)
    monkeypatch.setattr(validator, "validate", check)
    return state


@pytest.mark.parametrize("effort", EFFORT_ORDER)
def test_every_effort_routes_to_portfolio_once_and_validates(execution, effort):
    updates = []
    run = app.run_effort(execution.pb, effort, progress=lambda *values: updates.append(values), workers=1)
    expected = spec_for_effort(effort, seed=17, workers=1, full_problem=True)
    assert execution.spec == expected
    assert execution.calls == ["near", "solve", "validate"]
    assert run.algo == "portfolio" and "portfolio" in app.ALGOS
    assert run.mu is None
    assert run.notes["effort_level"] == effort
    assert run.notes["label"] == get_effort(effort).title
    assert run.notes["actual_solver_config"] == expected
    assert run.notes["stage_timings_s"] == {"hplus": 6., "finish": 1.}
    assert run.notes["famou"] == {"mode": "local_adopted_rule", "runtime_cloud_calls": 0}
    assert run.notes["refine"]["refined"] is True
    assert run.validation["legal"] is True
    assert run.validation["geometry_checked"] is True
    assert run.validation["step_s"] == 1.0
    assert run.metrics["score"] == 3.
    assert run.runtime_s == run.notes["solver_time_s"] == 7.
    assert run.notes["preprocessing_time_s"] == 2.
    assert run.notes["validation_time_s"] == 3.
    assert run.notes["total_time_s"] == 12.
    fractions = [fraction for fraction, _ in updates]
    assert fractions == sorted(fractions)
    assert fractions[0] == 0 and fractions[-1] == 1
    assert all(0 <= fraction <= 1 for fraction in fractions)
    assert any("逐秒校验" in message for _, message in updates)


def test_subset_disables_near_and_still_validates(execution):
    run = app.run_effort(execution.pb, "ultra", full_problem=False, workers=1)
    assert execution.calls == ["solve", "validate"]
    assert execution.spec["finish_near"] is False
    assert run.notes["full_problem"] is False
    assert run.notes["preprocessing_time_s"] == 0.
    assert run.notes["total_time_s"] == 10.
    assert run.validation["legal"] is True


@pytest.mark.parametrize("legal,geometry_checked", [(False, True), (False, False), (True, False)])
def test_invalid_or_unchecked_plan_is_retained_without_legal_claim(execution, legal, geometry_checked):
    execution.legal, execution.geometry_checked = legal, geometry_checked
    run = app.run_effort(execution.pb, "instant", workers=1)
    assert run.validation["legal"] is False
    assert run.validation["geometry_checked"] is geometry_checked
    assert len(run.schedule()) == 1
    # UI formats all margins numerically, including unavailable geometry.
    assert all(isinstance(value, float) for value in run.validation["margins"])
    [format(value, ".3f") for value in run.validation["margins"]]


def test_validation_failure_preserves_diagnostic_run(execution, tmp_path):
    execution.failure = OSError("synthetic ephemeris unavailable")
    run = app.run_effort(execution.pb, "instant", workers=1)
    assert run.validation["legal"] is False
    assert run.validation["geometry_checked"] is False
    assert run.validation["error_type"] == "OSError"
    assert run.validation["violations"]
    assert all(np.isnan(run.validation["margins"]))
    assert run.per_task is None
    assert run.notes["validation_time_s"] == 3.
    assert "ephemeris" in run.notes["validation_error"]
    restored = app.Run.load(run.save(tmp_path))
    assert restored.validation["legal"] is False
    assert restored.per_task is None


def test_failed_validation_summary_is_strict_json_without_mutating_internal_nan(execution):
    execution.failure = OSError("synthetic ephemeris unavailable")
    run = app.run_effort(execution.pb, "instant", workers=1)

    def reject_nonstandard_constant(value):
        pytest.fail(f"Public summary contains nonstandard JSON constant {value}")

    exported = json.loads(app.summary_json(run), parse_constant=reject_nonstandard_constant)
    assert exported["validation"]["legal"] is False
    assert exported["validation"]["margins"] == [None, None, None]
    assert all(np.isnan(run.validation["margins"]))


def test_legacy_summary_normalizes_nested_numpy_and_nonfinite_notes(execution):
    run = app.run_effort(execution.pb, "instant", workers=1)
    run.algo = "H"
    run.notes = {"weights": np.array([1.25, np.nan, np.inf]),
                 "nested": [{"finite": np.float32(2.5), "iterations": np.int64(7),
                             "enabled": np.bool_(True), "missing": -np.inf}],
                 "bounds": (np.float64(3.5), None)}
    exported = json.loads(app.summary_json(run), parse_constant=lambda value: pytest.fail(value))
    assert exported["notes"] == {
        "weights": [1.25, None, None],
        "nested": [{"finite": 2.5, "iterations": 7, "enabled": True, "missing": None}],
        "bounds": [3.5, None]}
    assert isinstance(exported["notes"]["nested"][0]["iterations"], int)
    assert np.isnan(run.notes["weights"][1])


def test_save_load_and_summary_preserve_effort_provenance(execution, tmp_path):
    run = app.run_effort(execution.pb, "max", workers=1)
    restored = app.Run.load(run.save(tmp_path))
    assert restored.algo == "portfolio"
    assert restored.notes == run.notes
    assert restored.validation == run.validation
    assert restored.runtime_s == run.runtime_s
    np.testing.assert_array_equal(restored.start_s, run.start_s)
    np.testing.assert_array_equal(restored.per_task, run.per_task)
    summary = json.loads(app.summary_json(restored))
    assert summary["notes"]["effort_level"] == "max"
    assert summary["notes"]["famou"]["runtime_cloud_calls"] == 0
    assert summary["validation"]["geometry_checked"] is True


@pytest.mark.parametrize("change", [
    {"w_cov": .4, "w_score": .6}, {"w_score": .4}, {"score_ref": 12345},
    {"cell_s": 60}, {"switch_s": 60}, {"repeat_gap_h": 24},
])
def test_unsupported_configuration_fails_before_geometry_or_solve(execution, change):
    execution.pb.cfg = execution.pb.cfg.with_(**change)
    with pytest.raises(ValueError):
        app.run_effort(execution.pb, "instant", workers=1)
    assert execution.calls == []


def test_unknown_effort_fails_without_work(execution):
    with pytest.raises(ValueError, match="未知计算档位"):
        app.run_effort(execution.pb, "not-a-level", workers=1)
    assert execution.calls == []
