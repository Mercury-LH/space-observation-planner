"""Teacher label contract on actual greedy/H logic; geometry is a marked stub."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from obsplan.config import Config
from obsplan.problem import Problem
from obsplan.schedule import Schedule
from obsplan.training_data import teacher as t
from obsplan.visibility import Windows


@pytest.fixture
def tiny():
    cfg = Config(days=1 / 24, use_sun=False, use_sunlit=False)
    ds = SimpleNamespace(n_targets=3, n_stations=1,
                         priority=np.array([5., 3., 2.]), priority_label=["高", "中", "低"])
    win = Windows(np.zeros(3, np.int16), np.arange(3, dtype=np.int32),
                  np.array([0, 0, 12], np.int32), np.array([8, 4, 20], np.int32),
                  [[(0, 120)]], 121, np.zeros(3, bool))
    return Problem.build(cfg, ds, win)


def altered_finish(pb, grid, **kw):
    assert kw == {"near": False, "verbose": False}
    final = Schedule(grid.station.copy(), grid.target.copy(), grid.start_s + 1,
                     grid.dur_s - 1, algo="explicit geometry stub")
    return final, {"refine": {"refined": True}, "synthetic_geometry_stub": True}


def test_real_three_iteration_teacher_has_finite_static_features_and_separate_plans(tiny, monkeypatch):
    monkeypatch.setattr(t, "finish", altered_finish)

    def no_local_search(*args, **kwargs):
        raise AssertionError("teacher must not run local search")

    monkeypatch.setattr(t.lagrange, "improve", no_local_search)
    import obsplan.famou_h_eval as h_eval
    monkeypatch.setattr(h_eval, "improve", no_local_search)
    result = t.build_teacher(tiny)
    grid, final = result["grid_plan"], result["final_plan"]
    assert result["features"].shape == (11, 20)
    assert result["features"].dtype == np.float32
    assert np.isfinite(result["features"]).all()
    assert result["metadata"]["price_iterations_requested"] == 3
    assert result["metadata"]["price_iterations_completed"] == 3
    assert not result["metadata"]["price_convergence_claimed"]
    assert not result["metadata"]["sequential_state_features_included"]
    assert result["metadata"]["famou"]["runtime_cloud_calls"] == 0
    assert "not_run" in result["metadata"]["validation_status"]
    assert np.array_equal(final.start_s, grid.start_s + 1)
    assert np.array_equal(final.dur_s, grid.dur_s - 1)
    ids = result["grid_task_candidate"]
    windows = result["candidate_window"][ids]
    assert np.array_equal(grid.target, tiny.w_tg[windows])
    assert np.array_equal(grid.station, tiny.w_st[windows])
    assert np.array_equal(grid.dur_s, result["candidate_duration_cells"][ids] * 30)
    assert result["grid_selected_count"].sum() == len(grid)
    for i, candidate_id in enumerate(ids):
        assert result["grid_selection_rank"][candidate_id] == i
        assert result["grid_start_s"][candidate_id] == grid.start_s[i]
        assert result["grid_duration_s"][candidate_id] == grid.dur_s[i]
    unselected = ~result["grid_selected"]
    assert unselected.any()
    assert np.all(result["grid_selection_rank"][unselected] == -1)
    assert np.all(result["grid_start_s"][unselected] == -1)
    assert np.all(result["grid_duration_s"][unselected] == -1)
    assert "false does not mean infeasible" in result["metadata"]["label_semantics"]["grid_selected"]
    assert result["timings"]["total_s"] >= sum(v for k, v in result["timings"].items() if k != "total_s")
    json.dumps(result["metadata"], allow_nan=False)


def test_mapping_preserves_multiple_actions_for_one_candidate(tiny):
    cw, cm = np.array([0, 0, 2]), np.array([4, 8, 4])
    state = SimpleNamespace(n=3, alive=np.ones(3, bool), w=np.array([0, 2, 0]),
                            m=np.array([4, 4, 4]), a=np.array([0, 12, 6000]))
    grid = Schedule.from_cells(tiny.cfg, [0, 0, 0], [0, 2, 0], state.a, state.m)
    labels = t._grid_labels(tiny, cw, cm, state, grid)
    assert labels["grid_task_candidate"].tolist() == [0, 2, 0]
    assert labels["grid_selected_count"].tolist() == [2, 0, 1]
    assert labels["grid_selection_rank"].tolist() == [0, -1, 1]
    assert labels["grid_start_s"].tolist() == [0, -1, 360]


def test_nonfinite_features_fail_before_recovery(tiny, monkeypatch):
    monkeypatch.setattr(t.lagrange, "solve", lambda *a, **kw: (None, {
        "mu": np.zeros((tiny.n_stations, tiny.n_samples + 2)), "history": []}))
    monkeypatch.setattr(t, "feature_rows", lambda pb, mu, cw, cm: np.full((len(cw), 20), np.nan))
    monkeypatch.setattr(t, "run_rule", lambda *a, **kw: pytest.fail("must reject features before recovery"))
    with pytest.raises(t.TeacherBuildError, match="finite candidate-by-20") as exc:
        t.build_teacher(tiny)
    assert "candidate_features_and_rule_s" in exc.value.partial_result["timings"]


def test_failed_refinement_retains_grid_and_failure_without_substituting_teacher(tiny, monkeypatch):
    monkeypatch.setattr(t, "finish", lambda pb, grid, **kw: (grid, {
        "refine": {"refined": False, "reason": "test LP failure"}}))
    with pytest.raises(t.TeacherBuildError, match="test LP failure") as exc:
        t.build_teacher(tiny)
    partial = exc.value.partial_result
    assert len(partial["grid_plan"]) > 0
    assert partial["finish_info"]["refine"]["refined"] is False
    assert "grid_task_candidate" in partial


def test_missing_rule_does_not_fall_back_to_manual_teacher(tiny, monkeypatch):
    monkeypatch.setattr(t, "load_rule", lambda: None)
    with pytest.raises(RuntimeError, match="adopted local recovery rule is missing"):
        t.build_teacher(tiny)
