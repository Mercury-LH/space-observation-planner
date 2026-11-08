"""Real constraint and transaction regressions; geometric truth is not mocked as passed."""
from dataclasses import replace
from types import SimpleNamespace
from time import perf_counter

import numpy as np
import pytest

from obsplan.config import Config
from obsplan.problem import Problem
from obsplan.schedule import Schedule
from obsplan.visibility import Windows
from obsplan.neural_lns.search import (Operation, Task, apply_operation, check_schedule,
                                       generate_operations, metrics, schedule_hash, search)
from obsplan.neural_lns.features import DYNAMIC_FEATURE_NAMES, FeatureContext, dynamic_row
from obsplan.neural_lns.counterfactual import bounded_repair, collect_state


def problem(rows, n=4, stations=2, days=6):
    cfg = Config(days=days, use_sun=False, use_sunlit=False)
    ds = SimpleNamespace(n_targets=n, n_stations=stations, priority=np.array([5., 3., 2., 2.])[:n],
                         priority_label=["高", "中", "低", "低"][:n])
    a = np.asarray(rows, np.int32)
    win = Windows(a[:, 0].astype(np.int16), a[:, 1], a[:, 2], a[:, 3],
                  [[(0, cfg.n_cells)] for _ in range(stations)], cfg.n_cells + 1, np.zeros(n, bool))
    return Problem.build(cfg, ds, win)


def plan(rows):
    a = np.asarray(rows, np.int64).reshape(-1, 4)
    return Schedule(*(a[:, i].copy() for i in range(4)))


def op(parent, insertion, remove=(), replacements=(), kind="insert", region=(0, 0, 14400)):
    return Operation(kind, schedule_hash(parent), tuple(remove), (insertion, *replacements),
                     *region, tuple(range(4)))


def test_cross_station_repeat_gap_is_end_to_start_and_checks_outside_region():
    pb = problem([(0, 0, 0, 8), (1, 0, 5760, 5790)])
    parent = plan([(0, 0, 0, 240)])
    # Exactly 48h since START is 240s too early even at a different station.
    bad = op(parent, Task(1, 0, 172800, 120, 1), region=(1, 172800, 187200))
    candidate, status = apply_operation(pb, parent, bad)
    assert candidate is None and "48" in status
    good = replace(bad, additions=(Task(1, 0, 173040, 120, 1),))
    candidate, status = apply_operation(pb, parent, good)
    assert status == "completed" and check_schedule(pb, candidate) == {}


def test_transaction_rollback_after_invalid_compound_keeps_all_arrays():
    pb = problem([(0, 0, 0, 20), (0, 1, 0, 20), (1, 0, 0, 20)])
    parent = plan([(0, 0, 0, 240)])
    before = schedule_hash(parent)
    operation = op(parent, Task(0, 1, 0, 120, 1), (0,), (Task(1, 0, 0, 120, 2),), "migrate")
    result, status = apply_operation(pb, parent, operation)
    assert status == "completed" and result is not parent
    assert schedule_hash(parent) == before
    # Index mismatch must fail before any schedule mutation.
    broken = replace(operation, additions=(Task(0, 1, 0, 120, 0), *operation.additions[1:]))
    assert apply_operation(pb, parent, broken)[1] == "invalid_task_window_mapping"
    assert schedule_hash(parent) == before


def test_timeout_rolls_back_even_after_candidate_construction():
    pb = problem([(0, 0, 0, 20)])
    parent = plan([]); operation = op(parent, Task(0, 0, 0, 120, 0))
    ticks = iter([0., 2.])
    result, status = apply_operation(pb, parent, operation, deadline=1, clock=lambda: next(ticks))
    assert result is None and status == "timeout" and len(parent) == 0


def test_stale_parent_and_more_than_three_modifications_rejected():
    pb = problem([(0, 0, 0, 20)])
    parent = plan([]); operation = op(parent, Task(0, 0, 0, 120, 0))
    assert apply_operation(pb, parent, replace(operation, parent_state_hash="bad"))[1] == "stale_parent"
    assert apply_operation(pb, parent, replace(operation, remove_indices=(0, 1, 2, 3)))[1] == "invalid_removed_indices"


def test_outside_local_tasks_fixed_even_if_move_would_be_legal():
    pb = problem([(0, 0, 0, 800), (0, 1, 0, 800)])
    parent = plan([(0, 0, 15000, 120)])
    operation = op(parent, Task(0, 1, 0, 120, 1), (0,), kind="replace")
    assert apply_operation(pb, parent, operation)[1] == "outside_region"


def test_generation_shared_deterministic_bounded_and_finite():
    pb = problem([(0, 0, 0, 8), (0, 1, 0, 20), (1, 0, 0, 20)], days=1/24)
    parent = plan([(0, 0, 0, 240)])
    first, log1 = generate_operations(pb, parent, seed=2026)
    second, log2 = generate_operations(pb, parent, seed=2026)
    assert first == second and log1 == log2 and 0 < len(first) <= 32
    assert all(len(o.remove_indices) <= 3 and len(o.related_targets) <= 64 for o in first)
    assert all(o.parent_state_hash == schedule_hash(parent) for o in first)


def test_rule_and_random_search_monotone_and_no_parent_mutation():
    pb = problem([(0, 0, 0, 8), (0, 1, 10, 20)], days=1/24)
    parent = plan([]); before = schedule_hash(parent)
    for mode in ("rule", "random"):
        final, info = search(pb, parent, deadline=perf_counter()+2, mode=mode, max_rounds=3)
        assert check_schedule(pb, final) == {}
        assert metrics(pb, final)["C"] >= metrics(pb, parent)["C"]
        assert metrics(pb, final)["Q"] >= metrics(pb, parent)["Q"]
        assert info["independent_geometry_checked"] is False
    assert schedule_hash(parent) == before


def test_expired_search_returns_last_legal_plan():
    pb = problem([(0, 0, 0, 8)], days=1/24)
    parent = plan([(0, 0, 0, 120)])
    final, info = search(pb, parent, deadline=perf_counter()-1)
    assert schedule_hash(final) == schedule_hash(parent) and info["deadline_reached"]


def test_feature_collection_and_inference_identical_missing_history_masked():
    pb = problem([(0, 0, 0, 8), (1, 0, 10, 20)], days=1/24)
    parent = plan([]); operation = op(parent, Task(0, 0, 0, 120, 0))
    ctx = FeatureContext(pb, np.zeros((pb.n_stations, pb.n_samples)))
    static, dynamic = ctx.rows(parent, [operation])
    static2, dynamic2 = FeatureContext(pb, ctx.mu).rows(parent, [operation])
    assert static.shape == (1, 20) and dynamic.shape == (1, 24)
    assert np.array_equal(static, static2) and np.array_equal(dynamic, dynamic2)
    assert len(DYNAMIC_FEATURE_NAMES) == 24
    assert list(dynamic[0, 2:6]) == [0, 0, 0, 0]
    assert ctx.fingerprint["pool_version"] == "full-windows-v1"


def test_counterfactual_refuses_validation_and_test_before_reading_features():
    for split in ("validation", "test"):
        with pytest.raises(ValueError, match="train"):
            collect_state(None, None, None, catalog_id="bad", scene_id="bad", split=split, initial_plan_hash="bad")


def test_counterfactual_hard_timeout_is_censored_not_zero():
    pb = problem([(0, 0, 0, 8)], days=1/24)
    result = bounded_repair(pb, plan([]), timeout_s=0.000001)
    assert result["status"] == "timeout" and result["censored"] is True and result["metrics"] is None


def test_counterfactual_complete_reference_has_observed_metrics():
    pb = problem([(0, 0, 0, 8)], days=1/24)
    result = bounded_repair(pb, plan([]), max_trials=1, timeout_s=2)
    assert result["status"] == "completed" and result["censored"] is False
    assert result["metrics"]["C"] == 1 and result["repair_trials"] == 1


def test_noninteger_schedule_rejected_before_public_indexing():
    pb = problem([(0, 0, 0, 8)], days=1/24)
    malformed = Schedule(np.array([0.]), np.array([0]), np.array([0]), np.array([120]))
    assert "integer_grid_required" in check_schedule(pb, malformed)


def test_search_rejects_score_gain_if_coverage_falls(monkeypatch):
    import obsplan.neural_lns.search as engine
    pb = problem([(0, 0, 0, 20), (0, 1, 0, 4), (0, 2, 5, 9)], days=1/24)
    parent = plan([(0, 1, 0, 120), (0, 2, 150, 120)])
    operation = op(parent, Task(0, 0, 0, 240, 0), (0, 1), kind="replace")
    candidate, status = apply_operation(pb, parent, operation)
    assert status == "completed" and metrics(pb, candidate)["S"] > metrics(pb, parent)["S"]
    monkeypatch.setattr(engine, "generate_operations", lambda *args, **kw: ([operation], {}))
    final, _ = engine.search(pb, parent, deadline=perf_counter()+2, max_rounds=1)
    assert schedule_hash(final) == schedule_hash(parent)


def test_plugin_output_contract_and_refinement_cost(monkeypatch):
    from obsplan.neural_lns import plugin
    pb = problem([(0, 0, 0, 8)], days=1/24)
    parent = plan([(0, 0, 0, 120)])
    monkeypatch.setattr(plugin, "initialize", lambda pb, spec, progress: (parent,
                        np.zeros((pb.n_stations, pb.n_samples)), {"wall_s": 0}))
    monkeypatch.setattr(plugin, "finish", lambda pb, grid, **kw: (grid, {"test_stub": True}))
    final, info = plugin.solve(pb, {"enabled": True, "mode": "rule", "budget_s": 2,
                                   "finish_reserve_s": .1, "max_rounds": 1, "workers": 1})
    assert check_schedule(pb, final) == {}
    assert info["initial_plan_hash"] == schedule_hash(parent)
    assert info["initial"] == metrics(pb, parent)
    assert "integer_second_refinement" in info["stage_timings_s"]
    assert info["requires_independent_skyfield_1s"] and not info["independent_geometry_checked"]


def test_empty_feature_batch_and_no_cf_worker_for_empty_pool(monkeypatch):
    import obsplan.neural_lns.counterfactual as cf
    pb = problem([(1, 0, 0, 8)], days=1/24)
    ctx = FeatureContext(pb, np.zeros((pb.n_stations, pb.n_samples)))
    x, d = ctx.rows(plan([]), [])
    assert x.shape == (0, 20) and d.shape == (0, 24)
    monkeypatch.setattr(cf, "bounded_repair", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no candidate")))
    assert collect_state(pb, plan([]), ctx, catalog_id="train", scene_id="train-v0",
                         initial_plan_hash=schedule_hash(plan([]))) == []


def test_mixed_sources_get_candidate_slots_and_unvisited_are_explicit():
    pb = problem([(0, t, 0, 20) for t in range(4)], days=1/24)
    ops, info = generate_operations(pb, plan([]))
    assert {o.source for o in ops} == {"fixed", "scarcity", "random"}
    assert info["enumeration"]["complete"] is False
    assert "no completeness guarantee" in info["enumeration"]["deferred_policy"]


def test_real_hplus_initialization_prices_feed_shared_feature_context():
    from obsplan.neural_lns.plugin import initialize
    pb = problem([(0, 0, 0, 8), (0, 1, 10, 20)], days=1/24)
    initial, mu, _ = initialize(pb, {"initial_h_iters": 1, "initial_h_fast": True})
    assert mu.shape == (pb.n_stations, pb.n_samples + 2)
    context = FeatureContext(pb, mu)
    ops, _ = generate_operations(pb, initial)
    static, dynamic = context.rows(initial, ops)
    assert static.shape == (len(ops), 20) and dynamic.shape == (len(ops), 24)
