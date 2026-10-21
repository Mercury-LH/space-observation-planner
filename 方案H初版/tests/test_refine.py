"""按秒精修：结果必须满足全部时间约束、不降分；真实边界探查要与 Skyfield 一致。"""
import numpy as np
import pytest

from obsplan import lagrange
from obsplan.refine import point_ok, refine, true_bounds
from obsplan.schedule import Schedule, evaluate
from obsplan.validator import validate


@pytest.fixture(scope="module")
def plans(sub):
    sch, _ = lagrange.solve(sub, iters=20, verbose=False)
    ref, info = refine(sub, sch)
    return sch, ref, info


def test_refine_keeps_constraints_and_never_worsens(cfg, ds, sub, plans):
    sch, ref, info = plans
    assert info["refined"]
    assert np.array_equal(ref.station, sch.station) and np.array_equal(ref.target, sch.target)
    assert ((ref.dur_s >= cfg.obs_min_s) & (ref.dur_s <= cfg.obs_max_s)).all()
    L, R = true_bounds(sub, sch)
    assert (ref.start_s >= L).all() and (ref.start_s + ref.dur_s <= R).all()
    end = ref.start_s + ref.dur_s
    for key, need in ((ref.station, cfg.switch_s), (ref.target, cfg.repeat_gap_h * 3600)):
        o = np.lexsort((ref.start_s, key))
        same = key[o[1:]] == key[o[:-1]]
        assert (ref.start_s[o[1:]][same] - end[o[:-1]][same] >= need).all()
    before, after = evaluate(cfg, ds, sch), evaluate(cfg, ds, ref)
    assert after["covered"] == before["covered"]
    assert after["score"] >= before["score"] - 1e-6


def test_probe_stops_where_conditions_fail(cfg, ds, sub, plans):
    sch, _, _ = plans
    L, R = true_bounds(sub, sch)
    i = np.arange(min(len(sch), 200))
    st, tg = sch.station[i].astype(np.int64), sch.target[i].astype(np.int64)
    inside_l = (L[i] >= 1)
    # 边界本身可见，再往外 1 秒（若在网格窗口两个采样点之间）不可见
    assert point_ok(cfg, ds, tg, st, L[i].astype(float)).all()
    assert point_ok(cfg, ds, tg, st, R[i].astype(float)).all()
    sel = inside_l & (L[i] % cfg.cell_s != 1)
    assert not point_ok(cfg, ds, tg[sel], st[sel], (L[i][sel] - 1).astype(float)).any()


@pytest.mark.slow
def test_refined_plan_passes_skyfield(cfg, ds, plans):
    _, ref, _ = plans
    idx = np.random.default_rng(1).choice(len(ref), min(60, len(ref)), replace=False)
    part = Schedule(ref.station[idx], ref.target[idx], ref.start_s[idx], ref.dur_s[idx])
    assert validate(cfg, ds, part)["legal"]


@pytest.mark.slow
def test_finish_inserts_second_level_targets_legally(cfg, ds, pb):
    from obsplan.greedy import algo_priority
    from obsplan.refine import finish
    sch = algo_priority(pb)
    out, info = finish(pb, sch)
    assert info["refine"]["refined"]
    end = out.start_s + out.dur_s
    assert ((out.dur_s >= cfg.obs_min_s) & (out.dur_s <= cfg.obs_max_s)).all()
    for key, need in ((out.station, cfg.switch_s), (out.target, cfg.repeat_gap_h * 3600)):
        o = np.lexsort((out.start_s, key))
        same = key[o[1:]] == key[o[:-1]]
        assert (out.start_s[o[1:]][same] - end[o[:-1]][same] >= need).all()
    new = np.nonzero(pb.max_obs[out.target] == 0)[0]            # 网格上不可见、按秒插入的目标
    assert len(new) == info["insert"]["added"]
    if len(new):
        part = Schedule(out.station[new], out.target[new], out.start_s[new], out.dur_s[new])
        assert validate(cfg, ds, part)["legal"]
    assert evaluate(cfg, ds, out)["combined"] >= evaluate(cfg, ds, sch)["combined"]
