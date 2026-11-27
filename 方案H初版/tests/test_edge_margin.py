"""按秒边界余量（Config.edge_margin_s）：边界两端各往里收 B 秒；B = 0 时与原来完全相同。"""
from pathlib import Path

import numpy as np
import pytest

from obsplan.refine import probe_bounds


def _sample(pb, n=400, seed=3):
    idx = np.sort(np.random.default_rng(seed).choice(len(pb.w_st), n, replace=False))
    return (pb.w_st[idx].astype(np.int64), pb.w_tg[idx].astype(np.int64),
            pb.w_k0[idx].astype(np.int64), pb.w_k1[idx].astype(np.int64))


def test_scenario_key_unchanged_without_margin(cfg):
    assert "edge" not in cfg.scenario_key()
    assert cfg.with_(edge_margin_s=2).scenario_key() == cfg.scenario_key() + "_edge2"


@pytest.mark.parametrize("B", [1, 2, 3])
def test_probe_bounds_shrinks_both_ends_except_horizon(cfg, ds, pb, B):
    """同一批网格窗口：留 B 秒余量的边界 = 不留余量的边界往里各收 B 秒；规划期的起点 0 和最晚结束时刻不收。"""
    st, tg, k0, k1 = _sample(pb)
    L0, R0 = probe_bounds(cfg, ds, st, tg, k0, k1)
    LB, RB = probe_bounds(cfg.with_(edge_margin_s=B), ds, st, tg, k0, k1)
    end = cfg.end_s                                               # 最晚结束时刻（规范截止 08-06 23:59:59）不收
    assert np.array_equal(LB, np.where(L0 > 0, L0 + B, L0))
    assert np.array_equal(RB, np.where(R0 < end, R0 - B, R0))
    assert R0.max() <= end and RB.max() <= end


def test_probe_bounds_horizon_edges_not_shrunk(cfg, ds, pb):
    """从规划期第 0 秒就开始的窗口，起点不收（规划期的起点不是可见边界）。"""
    first = np.nonzero(pb.w_k0 == 0)[0][:20]
    if len(first) == 0:
        pytest.skip("没有从第 0 秒开始的窗口")
    args = (pb.w_st[first].astype(np.int64), pb.w_tg[first].astype(np.int64),
            pb.w_k0[first].astype(np.int64), pb.w_k1[first].astype(np.int64))
    L0, _ = probe_bounds(cfg, ds, *args)
    LB, _ = probe_bounds(cfg.with_(edge_margin_s=2), ds, *args)
    assert np.array_equal(LB[L0 == 0], L0[L0 == 0])


@pytest.mark.slow
def test_margin_windows_stay_inside_shrunk_bounds(cfg, ds):
    """留 2 秒余量的网格窗口：首末采样点都落在"真实边界往里 2 秒"之内，窗口至少 4 格；
    探查出的按秒边界离不留余量的真实边界至少 2 秒。"""
    from obsplan.visibility import get_windows
    B = 2
    c2 = cfg.with_(edge_margin_s=B)
    base, w = get_windows(cfg, ds), get_windows(c2, ds)
    z = np.load(Path(cfg.cache_dir) / f"winbounds_{cfg.scenario_key()}.npz")
    L, R = z["L"], z["R"]
    end = cfg.end_s
    raw = {(int(s), int(t), int(a)): (int(lo), int(hi)) for s, t, a, lo, hi in
           zip(base.station, base.target, base.k0, L, R)}
    assert (w.k1 - w.k0 >= cfg.min_cells).all() and len(w.k0) <= len(base.k0)
    # 收紧后的窗口与原窗口一一对应：起点不变或往后一格
    hit = 0
    for s, t, a, b in zip(w.station[::50], w.target[::50], w.k0[::50], w.k1[::50]):
        lo, hi = raw.get((int(s), int(t), int(a)), raw.get((int(s), int(t), int(a) - 1), (None, None)))
        assert lo is not None
        assert a * cfg.cell_s >= (lo + B if lo > 0 else lo)
        assert b * cfg.cell_s <= (hi - B if hi < end else hi)
        hit += 1
    assert hit > 1000
    # 探查出的按秒边界，正好是真实边界往里收 2 秒（规划期起止不收）
    idx = np.arange(0, len(w.k0), 997)
    LB, RB = probe_bounds(c2, ds, w.station[idx].astype(np.int64), w.target[idx].astype(np.int64),
                          w.k0[idx].astype(np.int64), w.k1[idx].astype(np.int64))
    for i, lb, rb in zip(idx, LB, RB):
        s, t, a = int(w.station[i]), int(w.target[i]), int(w.k0[i])
        lo, hi = raw.get((s, t, a), raw.get((s, t, a - 1)))
        assert lb == (lo + B if lo > 0 else lo) and rb == (hi - B if hi < end else hi)
        assert lb <= w.k0[i] * cfg.cell_s and rb >= w.k1[i] * cfg.cell_s


@pytest.mark.parametrize("v,ok", [("0", 0), ("3", 3), ("10", 10), ("-1", None), ("11", None), ("1.5", None), ("x", None)])
def test_cli_edge_margin_range(v, ok):
    """--edge-margin 只收 0～10 的整数秒；越界在解析参数时就报错，不会悄悄当成 0。"""
    import argparse
    from obsplan.__main__ import _edge_margin, main
    if ok is not None:
        assert _edge_margin(v) == ok
        return
    with pytest.raises(argparse.ArgumentTypeError):
        _edge_margin(v)
    with pytest.raises(SystemExit) as e:
        main(["windows", "--edge-margin", v])
    assert e.value.code == 2


@pytest.mark.slow
def test_margin_windows_without_short_cache(cfg, ds, tmp_path):
    """缓存里只有不留余量的窗口文件、没有短窗口文件时，留余量的窗口照样能算，并补上短窗口文件。"""
    import shutil
    from obsplan.visibility import get_windows
    c0 = cfg.with_(days=1)
    src = Path(cfg.cache_dir)
    from obsplan.ephemeris import ensure_de421
    get_windows(c0, ds)                                           # 新机器上先把不留余量的 1 天窗口算好
    ensure_de421(src)
    shutil.copy(src / f"windows_{c0.scenario_key()}.npz", tmp_path)
    try:                                                          # Windows 上没有权限建符号链接时就复制
        (tmp_path / "de421.bsp").symlink_to(src / "de421.bsp")
    except OSError:
        shutil.copy(src / "de421.bsp", tmp_path)
    c2 = c0.with_(cache_dir=str(tmp_path), edge_margin_s=2)
    w = get_windows(c2, ds)
    assert (tmp_path / f"short_{c0.scenario_key()}.npz").exists()
    assert (tmp_path / f"short_{c2.scenario_key()}.npz").exists()
    ref = get_windows(c0.with_(edge_margin_s=2), ds)              # 正常缓存下算出的同一组窗口
    assert np.array_equal(w.k0, ref.k0) and np.array_equal(w.k1, ref.k1)
    a, b = np.load(tmp_path / f"short_{c2.scenario_key()}.npz"), np.load(src / f"short_{c2.scenario_key()}.npz")
    assert all(np.array_equal(a[k], b[k]) for k in ("station", "target", "k0", "k1"))


def test_windows_and_bounds_stop_at_spec_end(cfg, ds, pb):
    """默认规划期：网格窗口的末个采样点和按秒边界都不超过 08-06 23:59:59，所以计划不会排到 8/7 0 点那一秒。"""
    from obsplan.second_bounds import all_bounds
    assert cfg.end_s == cfg.n_cells * cfg.cell_s - 1
    assert int(pb.w_k1.max()) * cfg.cell_s <= cfg.end_s
    L, R = all_bounds(pb, Path(cfg.cache_dir) / f"bounds_all_{cfg.scenario_key()}.npz", verbose=False)
    assert int(R.max()) == cfg.end_s and (L < R).all()
