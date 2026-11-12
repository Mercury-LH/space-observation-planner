"""全部网格窗口的按秒真实可见区间 [L, R]（秒）。

refine.true_bounds 只探查"计划里用到的窗口"；跨站弹射链（obsplan/chain.py）和按秒上界要用到全部窗口，
这里把 52 万个窗口都探查一遍（两端各向外逐秒最多 29 秒，判定与 refine.point_ok 完全相同），结果存盘复用。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .problem import Problem
from .refine import probe_bounds


def _job(args):
    cfg, ds, st, tg, k0, k1 = args
    return probe_bounds(cfg, ds, st, tg, k0, k1)


def all_bounds(pb: Problem, path: Path | None = None, workers: int = 4, chunk_targets: int = 100,
               verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """返回 (L, R)，下标与 pb.w_* 一致。path 给定且存在时直接读回（窗口数对得上才用）。"""
    W = len(pb.w_st)
    if path is not None and Path(path).exists():
        z = np.load(path)
        if len(z["L"]) == W:
            return z["L"], z["R"]
    jobs, parts = [], []
    for i0 in range(0, pb.n_targets, chunk_targets):
        ws = pb.t_win[pb.t_ptr[i0]:pb.t_ptr[min(i0 + chunk_targets, pb.n_targets)]]
        if len(ws):
            parts.append(ws)
            jobs.append((pb.cfg, pb.ds, pb.w_st[ws], pb.w_tg[ws], pb.w_k0[ws], pb.w_k1[ws]))
    L = np.zeros(W, np.int64)
    R = np.zeros(W, np.int64)
    with ProcessPoolExecutor(workers) as ex:
        for n, (ws, (lo, hi)) in enumerate(zip(parts, ex.map(_job, jobs))):
            L[ws], R[ws] = lo, hi
            if verbose and n % 20 == 0:
                print(f"  [按秒边界] {n}/{len(jobs)} 块", flush=True)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, L=L, R=R)
    return L, R


def short_windows(pb: Problem) -> dict:
    """网格上只有 4 个采样点（90 秒）、按秒探查后够 120 秒的短窗口：{"station", "target", "k0", "L", "R"}。
    这些窗口不在 pb 的窗口表里；用了它们的计划做结构检查时要把它们作为 extra_windows 传给 structural_violations。"""
    from .bounds import _short_windows
    st, tg, k0, k1 = (np.asarray(x).astype(np.int64) for x in _short_windows(pb))
    keep = k1 - k0 == pb.m_min - 1
    st, tg, k0, k1 = st[keep], tg[keep], k0[keep], k1[keep]
    L, R = probe_bounds(pb.cfg, pb.ds, st, tg, k0, k1)
    ok = R - L >= pb.cfg.obs_min_s
    return {"station": st[ok], "target": tg[ok], "k0": k0[ok], "L": L[ok], "R": R[ok]}
