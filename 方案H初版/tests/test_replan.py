"""突发事件重规划：冻结部分不变、新计划无结构违规、停机站点在停机时段内没有任务、必须观测的目标排上。"""
import numpy as np
import pytest

from obsplan import lagrange
from obsplan.checks import structural_violations
from obsplan.refine import finish
from obsplan.replan import Event, compare_plans, replan

T_EVENT = int(2.5 * 86400)


@pytest.fixture(scope="module")
def plan(sub):
    sch, _ = lagrange.solve(sub, iters=20, verbose=False)
    out, info = finish(sub, sch, near=False)         # 与界面一致：先按秒精修，再在它上面重排
    assert info["refine"]["refined"]
    return out


def _rows(sch, sel):
    return sorted(zip(sch.station[sel].tolist(), sch.target[sel].tolist(), sch.start_s[sel].tolist(),
                      sch.dur_s[sel].tolist()))


def _check_common(sub, plan, ev, res):
    new = res["schedule"]
    t = ev.t_s
    keep = plan.start_s < t
    if ev.kind == "down":                              # t 时正在停机站执行的任务中断，不算已执行
        keep &= ~((plan.station == ev.station) & (plan.start_s + plan.dur_s > t))
    assert _rows(new, new.start_s < t) == _rows(plan, keep)
    assert res["frozen_unchanged"]
    assert structural_violations(sub, new, by_second=True) == {}
    d = res["diff"]
    assert d["保留"] + d["移动"] + d["删除"] == d["原计划 t 之后"]
    assert d["保留"] + d["移动"] + d["新增"] == int((new.start_s >= t).sum())


@pytest.mark.parametrize("mode", ["minimal", "full"])
def test_station_outage(sub, plan, mode):
    later = plan.start_s >= T_EVENT
    s = int(np.bincount(plan.station[later], minlength=sub.n_stations).argmax())
    ev = Event("down", T_EVENT, station=s, hours=8)
    res = replan(sub, plan, ev, mode=mode, iters=20)
    _check_common(sub, plan, ev, res)
    new = res["schedule"]
    da, db = ev.down
    assert not np.any((new.station == s) & (new.start_s < db) & (new.start_s + new.dur_s > da))
    assert res["outage_tasks"] == 0 and res["diff"]["停机取消"] > 0
    if mode == "minimal":                              # 扰动最小：停机以外的原任务绝大多数原样保留
        others = later & ~((plan.station == s) & (plan.start_s < db))
        assert res["diff"]["保留"] >= 0.9 * others.sum()
        assert res["after"]["combined"] >= res["no_replan"]["combined"] - 1e-9


def test_priority_raise_keeps_plan_and_helps_target(sub, plan):
    later = plan.start_s >= T_EVENT
    low = np.nonzero(sub.ds.priority == 2.0)[0]
    once = [x for x in low if (plan.target == x).sum() == 1 and later[plan.target == x].all()]
    x = int(once[0])
    ev = Event("priority", T_EVENT, target=x)
    res = replan(sub, plan, ev, iters=20)
    _check_common(sub, plan, ev, res)
    assert res["ds"].priority[x] == 5.0 and sub.ds.priority[x] == 2.0
    obs = lambda lst: sum(d for _, _, d in lst)          # noqa: E731
    assert obs(res["target"]["after"]) >= obs(res["target"]["before"])
    assert res["diff"]["保留"] >= 0.9 * later.sum()


def _not_observed_after(sub, plan, t):
    """t 之后没有观测、但（与 t 前最后一次观测隔开 48 小时后）还有窗口的目标，没覆盖过的排在前面。"""
    cell, gap = sub.cfg.cell_s, sub.cfg.repeat_gap_h * 3600
    out = []
    for x in np.nonzero(sub.max_obs > 0)[0]:
        mine = plan.target == x
        if mine.sum() >= 3 or np.any(mine & (plan.start_s >= t)):
            continue
        lo = max([t] + [e + gap for e in plan.start_s[mine] + plan.dur_s[mine]])
        ws = sub.t_win[sub.t_ptr[x]:sub.t_ptr[x + 1]]
        if np.any((sub.w_k1[ws] - sub.m_min) * cell >= lo + 600):
            out.append((int(mine.sum()), int(x)))
    return [x for _, x in sorted(out)]


def test_must_observe_target_gets_scheduled(sub, plan):
    cand = _not_observed_after(sub, plan, T_EVENT)
    assert cand, "小场景里应有 t 之后没有观测、但还有窗口的目标"
    x = cand[0]
    ev = Event("must", T_EVENT, target=x)
    res = replan(sub, plan, ev, iters=20)
    _check_common(sub, plan, ev, res)
    new = res["schedule"]
    assert np.any((new.target == x) & (new.start_s >= T_EVENT)) and res["target"]["ok"]


@pytest.mark.parametrize("target,mode", [(3730, "full"), (3730, "minimal"), (7877, "full")])
def test_must_observe_survives_local_repair(sub, plan, target, mode):
    """回归：这几个目标 t 之前已有观测，强行插入的那一次按正常价值很便宜，曾在随后的修补里被别的目标挤掉
    （结果里 forced 为真，计划里却没有它）。"""
    ev = Event("must", T_EVENT, target=target)
    res = replan(sub, plan, ev, mode=mode, iters=15)
    _check_common(sub, plan, ev, res)
    new = res["schedule"]
    assert res["target"]["forced"] and res["target"]["ok"]
    assert np.any((new.target == target) & (new.start_s >= T_EVENT))


@pytest.mark.parametrize("when", ["start", "end"])
def test_event_at_horizon_edges(sub, plan, when):
    t = 0 if when == "start" else int(sub.cfg.days * 86400) - 1800
    s = int(np.bincount(plan.station, minlength=sub.n_stations).argmax())
    ev = Event("down", t, station=s, hours=48)                # 末尾那次停机一直延续到规划期结束之后
    res = replan(sub, plan, ev, iters=10)
    _check_common(sub, plan, ev, res)
    new = res["schedule"]
    da, db = ev.down
    assert not np.any((new.station == s) & (new.start_s < db) & (new.start_s + new.dur_s > da))


def test_compare_plans_identical_is_all_kept(plan):
    sel = plan.start_s >= T_EVENT
    o, n = compare_plans(plan, sel, plan, sel)
    assert set(o[sel]) == {"保留"} and set(n[sel]) == {"保留"} and set(o[~sel]) <= {"冻结"}
