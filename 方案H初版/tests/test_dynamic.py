"""动态重规划：多天重排拼起来的执行记录必须合法（构造的随机云量，不联网）。"""
import numpy as np

from obsplan.checks import structural_violations
from obsplan.dynamic import calibrate, clip_windows, simulate, task_succeeds
from obsplan.schedule import Schedule


def _sched(pb, tasks, only_ok=False):
    rows = [t for t in tasks if t[4] or not only_ok]
    c = pb.cfg.cell_s
    return Schedule(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
                    np.array([r[2] * c for r in rows]), np.array([r[3] * c for r in rows]))


def test_dynamic_execution_is_legal(cfg, sub):
    rng = np.random.default_rng(1)
    H = int(cfg.days * 24)
    truth = rng.choice([10.0, 90.0], size=(sub.n_stations, H), p=[0.6, 0.4])
    fc = np.where(rng.random(truth.shape) < 0.8, truth, 100 - truth)
    clim = np.full(sub.n_stations, 0.6)
    cal = calibrate(fc, truth, 30)
    res, status = simulate(sub, truth, "dynamic_forecast", clim, fc1=fc, fc2=fc, cal1=cal, cal2=cal,
                           iters=20, return_status=True)
    assert res["executed"] > 0 and res["covered"] > 0
    bad_all = structural_violations(sub, _sched(sub, status.tasks))
    bad_all.pop("48 小时间隔不足", None)          # 失败的尝试不计入 48 小时规则
    bad_all.pop("同目标超过 3 次", None)
    assert bad_all == {}
    assert structural_violations(sub, _sched(sub, status.tasks, only_ok=True)) == {}


def test_clear_sky_dynamic_matches_plan_success(cfg, sub):
    H = int(cfg.days * 24)
    truth = np.zeros((sub.n_stations, H))
    clim = np.ones(sub.n_stations)
    res = simulate(sub, truth, "dynamic_clim", clim, iters=20)
    assert res["failed"] == 0 and res["covered"] > 0


def test_clipped_windows_only_contain_clear_hours(cfg, sub):
    rng = np.random.default_rng(2)
    H = int(cfg.days * 24)
    truth = rng.choice([10.0, 90.0], size=(sub.n_stations, H), p=[0.5, 0.5])
    pc = clip_windows(sub, truth <= 30)
    assert 0 < len(pc.w_k0) and np.all(pc.w_k1 - pc.w_k0 >= pc.m_min)
    for i in rng.choice(len(pc.w_k0), min(300, len(pc.w_k0)), replace=False):
        a = int(rng.integers(pc.w_k0[i], pc.w_k1[i] - pc.m_min + 1))
        assert task_succeeds(truth, int(pc.w_st[i]), a, pc.m_min, cfg.cell_s, 30)


def test_hindsight_never_fails(cfg, sub):
    rng = np.random.default_rng(3)
    H = int(cfg.days * 24)
    truth = rng.choice([10.0, 90.0], size=(sub.n_stations, H), p=[0.6, 0.4])
    res = simulate(sub, truth, "hindsight", np.full(sub.n_stations, 0.6), iters=20)
    assert res["failed"] == 0 and res["covered"] > 0
