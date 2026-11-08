"""Compare the real baseline/candidate operators on mutable scheduling states.

No contest data or full-instance timing is needed. Target observation records
are compared as sets because harmless detach/attach probes can reorder them.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import h_acceleration_bounds as acceleration
from obsplan import local_search as ls
from obsplan.config import Config
from obsplan.greedy import State, _find_start, candidates, run_greedy
from obsplan.problem import Problem
from obsplan.schedule import Schedule
from obsplan.visibility import Windows


def _problem(windows, priorities=(5.0, 3.0, 2.0), stations=2, samples=96, gap=10):
    rows = np.asarray(sorted(windows, key=lambda item: (item[0], item[2])), dtype=np.int64)
    cfg = Config(days=(samples - 1) * 30 / 86400, repeat_gap_h=gap * 30 / 3600,
                 use_sun=False, use_sunlit=False)
    ds = SimpleNamespace(n_targets=len(priorities), n_stations=stations,
                         priority=np.asarray(priorities, dtype=float),
                         priority_label=np.full(len(priorities), "高"))
    win = Windows(rows[:, 0].astype(np.int16), rows[:, 1].astype(np.int32),
                  rows[:, 2].astype(np.int32), rows[:, 3].astype(np.int32),
                  [[(0, samples - 1)] for _ in range(stations)], samples,
                  np.zeros(len(priorities), bool))
    return Problem.build(cfg, ds, win)


def _add(pb, state, target, station, start, duration):
    choices = np.flatnonzero((pb.w_tg == target) & (pb.w_st == station)
                             & (pb.w_k0 <= start) & (pb.w_k1 >= start + duration))
    assert len(choices)
    assert np.all(state.owner[station, start:start + duration + 1] == -1)
    k, count = state.n, state.t_cnt[target]
    assert count < 3
    state.st[k], state.tg[k], state.a[k], state.m[k], state.w[k] = station, target, start, duration, choices[0]
    state.alive[k] = True
    state.owner[station, start:start + duration + 1] = k
    state.t_task[target, count] = k
    state.t_obs_a[target, count], state.t_obs_m[target, count] = start, duration
    state.t_cnt[target] += 1
    state.n += 1
    return k


def _apply(operator, pb, state, targets, max_pos=40):
    state.n, inserted = operator(np.asarray(targets, dtype=np.int64),
        state.st, state.tg, state.a, state.m, state.w, state.alive, state.n,
        state.owner, state.t_cnt, state.t_obs_a, state.t_obs_m, state.t_task,
        pb.w_st, pb.w_k0, pb.w_k1, pb.t_ptr, pb.t_win, pb.P, pb.frac,
        pb.vc, pb.vs, pb.gap, pb.m_min, max_pos)
    return inserted


def _assert_equal(left, right):
    assert left.n == right.n
    for name in ("st", "tg", "a", "m", "w", "alive", "owner", "t_cnt"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name), err_msg=name)
    for target, count in enumerate(left.t_cnt):
        def records(state):
            return sorted(zip(state.t_task[target, :count].tolist(),
                              state.t_obs_a[target, :count].tolist(),
                              state.t_obs_m[target, :count].tolist()))
        assert records(left) == records(right)


def _paired(pb, state, targets, max_pos=40):
    baseline, candidate = deepcopy(state), deepcopy(state)
    expected = _apply(ls.repair_coverage, pb, baseline, targets, max_pos)
    actual = _apply(acceleration.repair_coverage_multi_bound, pb, candidate, targets, max_pos)
    assert actual == expected
    _assert_equal(baseline, candidate)
    return candidate, actual


def test_empty_position_keeps_first_maximum_across_windows():
    pb = _problem([(0, 0, 0, 30), (1, 0, 0, 30)], priorities=(5.0,))
    result, inserted = _paired(pb, State(pb, cap=64), [0])
    assert inserted == 1
    assert (result.st[0], result.a[0], result.m[0]) == (0, 0, 4)


def test_zero_loss_relocation_can_change_station():
    pb = _problem([(0, 0, 0, 4), (0, 1, 0, 4), (1, 1, 0, 8)], priorities=(5.0, 2.0))
    state = State(pb, cap=64)
    blocker = _add(pb, state, 1, 0, 0, 4)
    result, inserted = _paired(pb, state, [0])
    assert inserted == 1 and result.st[blocker] == 1 and result.m[blocker] == 4
    assert result.t_cnt.tolist() == [1, 1]


def test_positive_loss_shortening_still_searches_and_applies():
    pb = _problem([(0, 1, 0, 8), (0, 0, 5, 9)], priorities=(5.0, 2.0))
    state = State(pb, cap=64)
    blocker = _add(pb, state, 1, 0, 0, 8)
    result, inserted = _paired(pb, state, [0])
    assert inserted == 1
    assert (result.a[blocker], result.m[blocker]) == (0, 4)
    assert result.a[result.n - 1] == 5


def test_repeat_boundary_is_measured_from_previous_end():
    pb = _problem([(0, 0, 13, 17), (0, 0, 14, 18), (1, 0, 0, 4)], priorities=(5.0,))
    state = State(pb, cap=64)
    _add(pb, state, 0, 1, 0, 4)
    result, inserted = _paired(pb, state, [0])
    assert inserted == 1 and result.a[result.n - 1] == 14
    assert result.a[result.n - 1] - (result.a[0] + result.m[0]) == pb.gap


def test_two_blockers_keep_original_repeat_deletion_behavior():
    pb = _problem([(0, 1, 0, 4), (0, 0, 3, 7), (0, 2, 5, 9),
                   (1, 1, 30, 34), (1, 2, 45, 49)], priorities=(5.0, 2.0, 2.0))
    state = State(pb, cap=64)
    first = _add(pb, state, 1, 0, 0, 4)
    second = _add(pb, state, 2, 0, 5, 4)
    _add(pb, state, 1, 1, 30, 4)
    _add(pb, state, 2, 1, 45, 4)
    result, inserted = _paired(pb, state, [0])
    assert inserted == 1 and not result.alive[first] and not result.alive[second]
    assert result.t_cnt.tolist() == [1, 1, 1]


def test_impossible_insertion_keeps_valuable_first_coverage():
    pb = _problem([(0, 0, 0, 4), (0, 1, 0, 4)], priorities=(2.0, 5.0))
    state = State(pb, cap=64)
    blocker = _add(pb, state, 1, 0, 0, 4)
    result, inserted = _paired(pb, state, [0])
    assert inserted == 0 and result.alive[blocker]
    _assert_equal(state, result)


@pytest.mark.parametrize("seed", range(24))
def test_randomized_real_greedy_states_match_over_successive_repairs(seed):
    rng = np.random.default_rng(seed)
    priorities = rng.choice([2.0, 3.0, 5.0], size=8)
    windows = []
    for target in range(len(priorities)):
        for _ in range(4):
            start = int(rng.integers(0, 85))
            end = min(95, start + int(rng.integers(4, 25)))
            windows.append((int(rng.integers(0, 2)), target, start, end))
    pb = _problem(windows, priorities=priorities, gap=12)
    cw, cm = candidates(pb)
    value = pb.vs * pb.P[pb.w_tg[cw]] * pb.frac[cm]
    initial = run_greedy(pb, cw, cm, value + pb.vc, value, best_fit=True, state=State(pb, cap=128))
    baseline, candidate = deepcopy(initial), deepcopy(initial)
    for _ in range(3):
        targets = rng.permutation(pb.n_targets)
        expected = _apply(ls.repair_coverage, pb, baseline, targets, max_pos=3)
        actual = _apply(acceleration.repair_coverage_multi_bound, pb, candidate, targets, max_pos=3)
        assert actual == expected
        _assert_equal(baseline, candidate)


def test_adapter_requires_explicit_opt_in_and_does_not_mutate_spec(monkeypatch):
    pb = _problem([(0, 0, 0, 8)], priorities=(5.0,))
    original = ls.repair_coverage
    spec = {"preset": "hplus_fast", "acceleration": "multi_bound"}
    events = []

    def solve(problem, passed, progress=None):
        assert problem is pb and passed == {"preset": "hplus_fast"}
        assert ls.repair_coverage is acceleration.repair_coverage_multi_bound
        assert progress is events.append or progress == events.append
        return Schedule.from_cells(pb.cfg, [], [], [], []), {"runtime_s": 0.0}

    monkeypatch.setattr(acceleration.portfolio, "solve", solve)
    schedule, info = acceleration.solve(pb, spec, events.append)
    assert ls.repair_coverage is original
    assert spec == {"preset": "hplus_fast", "acceleration": "multi_bound"}
    assert info["acceleration"]["changed_factors"] == 3
    assert info["runtime_s"] == schedule.runtime_s
    with pytest.raises(ValueError, match="explicit acceleration"):
        acceleration.solve(pb, {"preset": "hplus_fast"})
    assert ls.repair_coverage is original


def test_adapter_restores_operator_after_solver_exception(monkeypatch):
    pb = _problem([(0, 0, 0, 8)], priorities=(5.0,))
    original = ls.repair_coverage

    def fail(*args, **kwargs):
        assert ls.repair_coverage is acceleration.repair_coverage_multi_bound
        raise RuntimeError("synthetic failure during solve")

    monkeypatch.setattr(acceleration.portfolio, "solve", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        acceleration.solve(pb, {"acceleration": "multi_bound"})
    assert ls.repair_coverage is original


@pytest.mark.parametrize("field,value", [("P", [-1.0]), ("P", [np.nan]),
    ("vc", -1.0), ("vs", -1.0), ("vs", np.inf),
    ("frac", [0.0, 0.0, 0.0, 0.0, 0.6, 0.7, 0.65, 0.9, 1.0]),
    ("frac", [0.0, 0.0, 0.0, 0.0, 0.6, 0.7, 0.8, 0.9, np.nan])])
def test_unsupported_objectives_are_rejected_before_patching(monkeypatch, field, value):
    pb = _problem([(0, 0, 0, 8)], priorities=(5.0,))
    setattr(pb, field, np.asarray(value) if isinstance(value, list) else value)
    original = ls.repair_coverage

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported objective must not reach the portfolio")

    monkeypatch.setattr(acceleration.portfolio, "solve", unexpected)
    with pytest.raises(ValueError, match="finite nonnegative"):
        acceleration.solve(pb, {"acceleration": "multi_bound"})
    assert ls.repair_coverage is original


def _find_pair(owner, k0, k1, m, obs_a=(), obs_m=(), gap=10, best_fit=True):
    starts = np.zeros(3, np.int64)
    durations = np.zeros(3, np.int64)
    starts[:len(obs_a)], durations[:len(obs_m)] = obs_a, obs_m
    args = (owner, k0, k1, m, len(obs_a), starts, durations, gap, best_fit, len(owner))
    expected = _find_start(*args)
    actual = acceleration._find_start_exact(*args)
    assert actual == expected
    return actual


@pytest.mark.parametrize("best_fit", [False, True])
def test_find_start_multiple_free_runs_keep_earliest_zero_waste(best_fit):
    owner = np.full(64, 7, np.int32)
    owner[0:12] = -1
    owner[16:40] = -1
    assert _find_pair(owner, 0, 39, 4, best_fit=best_fit) == 0


def test_find_start_does_not_stop_at_positive_waste():
    owner = np.full(64, 7, np.int32)
    owner[0:7] = -1   # The first run leaves two unusable cells.
    owner[10:20] = -1  # This later run can leave zero waste.
    assert _find_pair(owner, 0, 19, 4, best_fit=True) == 10
    assert _find_pair(owner, 0, 19, 4, best_fit=False) == 0


@pytest.mark.parametrize("best_fit", [False, True])
def test_find_start_record_permutation_and_end_gap_boundary(best_fit):
    owner = np.full(80, -1, np.int32)
    forward = _find_pair(owner, 13, 35, 4, (0, 40), (4, 4), gap=10, best_fit=best_fit)
    reverse = _find_pair(owner, 13, 35, 4, (40, 0), (4, 4), gap=10, best_fit=best_fit)
    assert forward == reverse == 14
    assert _find_pair(owner, 13, 17, 4, (0,), (4,), gap=10, best_fit=best_fit) == -1


@pytest.mark.parametrize("seed", range(64))
@pytest.mark.parametrize("best_fit", [False, True])
def test_randomized_find_start_exact_output(seed, best_fit):
    rng = np.random.default_rng(seed)
    owner = np.where(rng.random(80) < 0.16, rng.integers(0, 8, size=80), -1).astype(np.int32)
    # Three already observed tasks are valid with respect to one another; their
    # stored order is deliberately randomized because it has no semantic role.
    count = int(rng.integers(0, 4))
    indices = rng.permutation(3)[:count]
    observations = np.array([0, 28, 56], np.int64)[indices]
    durations = rng.integers(4, 9, size=count)
    k0 = int(rng.integers(0, 65))
    k1 = int(rng.integers(k0 + 1, 80))
    m = int(rng.integers(4, 9))
    _find_pair(owner, k0, k1, m, observations, durations, best_fit=best_fit)


def test_first_blocker_loss_prunes_second_without_changing_state(monkeypatch):
    # Target 0 is already covered, so its repeat value cannot pay for deleting
    # even the first blocker, which is another target's unique observation.
    pb = _problem([(0, 1, 0, 4), (0, 0, 3, 7), (0, 2, 5, 9),
                   (1, 0, 30, 34)], priorities=(2.0, 5.0, 5.0))
    state = State(pb, cap=64)
    _add(pb, state, 1, 0, 0, 4)
    _add(pb, state, 2, 0, 5, 4)
    _add(pb, state, 0, 1, 30, 4)
    expected, inserted = _paired(pb, state, [0])
    assert inserted == 0
    calls = []
    original_relocate = acceleration._relocate_option_exact

    def counted(*args):
        calls.append(int(args[2][args[0]]))
        return original_relocate(*args)

    # Run the identical Python body only for observing this branch. The paired
    # test above exercises the actual compiled candidate against the baseline.
    monkeypatch.setattr(acceleration, "_relocate_option_exact", counted)
    observed = deepcopy(state)
    assert _apply(acceleration.repair_coverage_multi_bound.py_func, pb, observed, [0]) == 0
    assert calls == [1]
    _assert_equal(expected, observed)
