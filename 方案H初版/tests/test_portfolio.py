"""独立的小型合成窗口场景；不用赛题数据、不请求伐谋、不跑全量优化。"""
from copy import deepcopy
import json
from types import SimpleNamespace

import numba
import numpy as np
import pytest

from obsplan import portfolio as p
from obsplan.checks import structural_violations
from obsplan.config import Config
from obsplan.problem import Problem
from obsplan.schedule import Schedule, evaluate
from obsplan.visibility import Windows


@pytest.fixture
def tiny():
    cfg = Config(days=1 / 24, use_sun=False, use_sunlit=False)
    ds = SimpleNamespace(n_targets=3, n_stations=1, priority=np.array([5.0, 3.0, 2.0]),
                         priority_label=np.array(["高", "中", "低"]))
    win = Windows(np.zeros(3, np.int16), np.arange(3, dtype=np.int32),
                  np.zeros(3, np.int32), np.full(3, 119, np.int32),
                  [[(0, 119)]], 121, np.zeros(3, bool))
    return Problem.build(cfg, ds, win)


def plan(pb, targets):
    return Schedule.from_cells(pb.cfg, np.zeros(len(targets), int), targets,
                               np.arange(len(targets)) * 10, np.full(len(targets), 8))


class FrozenState:
    def __init__(self, sch):
        self.sch = sch

    def schedule(self, pb, name, runtime):
        return self.sch


@pytest.fixture
def sources(monkeypatch, tiny):
    """只替换来源生成和真实几何精修；融合 CP-SAT 默认仍然实际求解。"""
    calls = []
    weights = np.arange(8, dtype=float)

    def train(pb, **kw):
        calls.append(("ga", kw))
        return weights, {"fitness": 1.0}

    def greedy(pb, w, name, **kw):
        assert w is weights
        calls.append(("greedy", kw))
        sch = plan(pb, [1])
        return sch, FrozenState(sch)

    def gsolve(pb, w, **kw):
        assert w is weights
        calls.append(("g", kw))
        return plan(pb, [2])

    def hsolve(pb, **kw):
        calls.append(("h", kw))
        return plan(pb, [0]), {"history": [(0, 1.0, .5, 1.0)], "mu": np.zeros((1, 121))}

    def refine(pb, sch, **kw):
        calls.append(("finish", kw))
        return sch, {"refine": {"refined": True}, "synthetic_identity": True}

    monkeypatch.setattr(p.ga, "train", train)
    monkeypatch.setattr(p, "algo_weighted", greedy)
    monkeypatch.setattr(p.cpsat, "solve", gsolve)
    monkeypatch.setattr(p.cpsat, "to_state", lambda pb, sch: FrozenState(sch))
    monkeypatch.setattr(p.lagrange, "solve", hsolve)
    monkeypatch.setattr(p, "load_rule", lambda: lambda x: x)
    monkeypatch.setattr(p, "finish", refine)
    monkeypatch.setattr(p, "improve", lambda pb, state: {})
    return calls


def test_baseline_sources_generated_once_and_costs_counted(tiny, sources, monkeypatch):
    merge_calls = []

    def merge(pb, pool, cur, **kw):
        merge_calls.append(kw)
        return cur, []

    monkeypatch.setattr(p.fm, "merge_blocks", merge)
    original = deepcopy(p.PRESETS)
    sch, info = p.solve(tiny, {"preset": "fusion_baseline", "workers": 1})
    assert [name for name, _ in sources].count("ga") == 1
    assert [name for name, _ in sources].count("greedy") == 1
    assert [name for name, _ in sources].count("g") == 1
    assert [name for name, _ in sources].count("h") == 1
    assert [name for name, _ in sources].count("finish") == 1
    by_name = dict(sources)
    assert by_name["ga"] == {"n_scen": 3, "pop": 16, "gens": 12, "seed": 2026, "verbose": False}
    assert by_name["g"]["block_h"] == 4
    assert by_name["g"]["commit_h"] == 3
    assert by_name["g"]["time_limit"] == 10
    assert by_name["h"]["iters"] == 300 and by_name["h"]["fast"] is False
    assert len(merge_calls) == 3
    assert all(c["time_limit"] == 30 and c["budget_s"] == 300 for c in merge_calls)
    assert all(c["block_h"] == 24 and c["step_h"] == 12 and c["seed"] == 0 for c in merge_calls)
    assert p.PRESETS == original  # 运行时覆盖不会污染下次实验。
    stages = info["stage_timings_s"]
    assert {"ga_train", "e_construct", "e_local_search", "g_construct", "g_local_search", "hplus", "finish"} <= stages.keys()
    assert info["runtime_s"] >= sum(stages.values())
    assert sch.runtime_s == info["runtime_s"]
    assert info["stop_reason"] == "round_limit"
    assert info["famou"]["runtime_cloud_calls"] == 0
    assert "mu" not in info["details"]["hplus"]
    assert info["g_seed_control"] == "upstream_default_not_overridden"
    json.dumps(info, allow_nan=False)


def test_fast_only_runs_hplus_and_refinement(tiny, sources):
    events = []
    previous_threads = numba.get_num_threads()
    sch, info = p.solve(tiny, {"preset": "hplus_fast", "workers": 1, "seed": 7}, events.append)
    assert [name for name, _ in sources] == ["h", "finish"]
    assert dict(sources)["h"]["iters"] == 150
    assert dict(sources)["h"]["fast"] is True
    assert info["actual_config"]["fusion_seed"] == 7
    assert info["stop_reason"] == "no_fusion_requested"
    assert set(info["grid_final_origin"]["same_as"]) == {"H+"}
    assert numba.get_num_threads() == previous_threads
    assert not structural_violations(tiny, sch)
    assert {e["event"] for e in events} >= {"start", "complete"}


def test_real_fusion_recovers_complementary_sources(tiny, sources):
    # H 和 E 的原计划起始时间冲突；真实候选池允许缩短/平移并选出兼容组合。
    sch, info = p.solve(tiny, {"preset": "hplus_e", "workers": 1,
                              "fusion": {"rounds": 1, "time_limit_s": 2, "budget_s": 3,
                                         "shifts": [10], "local_search": False}})
    assert "g" not in [name for name, _ in sources]
    assert not structural_violations(tiny, sch)
    assert evaluate(tiny.cfg, tiny.ds, sch)["combined"] > max(v["combined"] for v in info["sources"].values())
    assert info["grid_final"]["covered"] == 2
    assert info["grid_final_origin"]["new_or_variant"] >= 1


def test_worse_round_cannot_replace_incumbent_and_stops(tiny, sources, monkeypatch):
    monkeypatch.setattr(p.fm, "merge_blocks", lambda pb, pool, cur, **kw: (np.zeros(0, np.int64), []))
    monkeypatch.setattr(p, "state_from_schedule", lambda pb, sch: FrozenState(sch))
    sch, info = p.solve(tiny, {"preset": "hplus_self", "workers": 1, "fusion": {"rounds": 4}})
    assert sch.target.tolist() == [0]
    assert len(info["rounds"]) == 1
    assert info["rounds"][0]["merged"]["tasks"] == 0
    assert info["rounds"][0]["gain"] == 0
    assert info["stop_reason"] == "low_marginal_gain"


@pytest.mark.parametrize("change", [
    {"workers": 0}, {"h_iters": 0}, {"fusion": {"budget_s": float("nan")}},
    {"ga": {"pop": 2}}, {"fusion": {"step_h": .00001}},
    {"sources": ["H+", "H+"]}, {"sources": []}, {"unknown_option": 5},
    {"fusion": {"durations": [3]}}, {"g": {"commit_h": 5}},
])
def test_invalid_specs_fail_before_expensive_work(tiny, sources, change):
    with pytest.raises(ValueError):
        p.solve(tiny, {"preset": "hplus_fast", "workers": 1, **change})
    assert sources == []


@pytest.mark.parametrize("change", [{"cell_s": 60}, {"w_cov": .4}, {"score_ref": 12345.0}])
def test_rejects_unsupported_objective_or_grid(tiny, sources, change):
    tiny.cfg = tiny.cfg.with_(**change)
    with pytest.raises(ValueError):
        p.solve(tiny, {"preset": "hplus_fast", "workers": 1})
    assert sources == []


def test_missing_adopted_rule_is_not_silently_h(tiny, sources, monkeypatch):
    monkeypatch.setattr(p, "load_rule", lambda: None)
    previous_threads = numba.get_num_threads()
    with pytest.raises(RuntimeError, match="recover_rule"):
        p.solve(tiny, {"preset": "hplus_fast", "workers": 1})
    assert numba.get_num_threads() == previous_threads
    assert sources == []
