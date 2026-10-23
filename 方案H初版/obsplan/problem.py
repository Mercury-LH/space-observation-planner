"""算法共用的问题实例：窗口、目标价值、按目标/站点的索引。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .data import Dataset
from .schedule import score_ref
from .visibility import Windows


@dataclass
class Problem:
    cfg: Config
    ds: Dataset
    win: Windows
    n_targets: int
    n_stations: int
    n_samples: int
    P: np.ndarray            # (N,) 优先级基础分
    vc: float                # 覆盖一颗目标在综合指标里的价值
    vs: float                # 1 分累计得分在综合指标里的价值
    frac: np.ndarray         # frac[m]：观测 m 格的得分比例
    m_min: int
    m_max: int
    gap: int                 # 48 小时对应格数
    # 窗口（按站点、开始时间排序）
    w_st: np.ndarray
    w_tg: np.ndarray
    w_k0: np.ndarray
    w_k1: np.ndarray
    # 按目标的窗口索引（CSR）
    t_ptr: np.ndarray
    t_win: np.ndarray        # 窗口下标，每个目标内按时间排序
    max_obs: np.ndarray      # 每颗目标在 48 小时间隔下最多能观测几次

    @classmethod
    def build(cls, cfg: Config, ds: Dataset, win: Windows) -> "Problem":
        n = ds.n_targets
        frac = np.zeros(cfg.max_cells + 1)
        for m in range(cfg.min_cells, cfg.max_cells + 1):
            frac[m] = 0.6 + 0.4 * (m * cfg.cell_s - cfg.obs_min_s) / (cfg.obs_max_s - cfg.obs_min_s)
        order = np.lexsort((win.k0, win.target))
        counts = np.bincount(win.target, minlength=n)
        t_ptr = np.zeros(n + 1, np.int64)
        np.cumsum(counts, out=t_ptr[1:])
        p = cls(cfg, ds, win, n, ds.n_stations, win.n_samples, ds.priority.astype(np.float64),
                cfg.w_cov / n, cfg.w_score / score_ref(cfg, ds), frac, cfg.min_cells, cfg.max_cells,
                cfg.gap_cells, win.station.astype(np.int64), win.target.astype(np.int64),
                win.k0.astype(np.int64), win.k1.astype(np.int64), t_ptr, order.astype(np.int64),
                np.zeros(n, np.int64))
        p.max_obs = p._max_obs()
        return p

    def _max_obs(self) -> np.ndarray:
        """最早结束优先：每次选能最早结束的那次观测，得到每颗目标最多能排几次（用于上界）。"""
        out = np.zeros(self.n_targets, np.int64)
        for i in range(self.n_targets):
            ws = self.t_win[self.t_ptr[i]:self.t_ptr[i + 1]]
            if len(ws) == 0:
                continue
            k0, k1 = self.w_k0[ws], self.w_k1[ws]
            last_end = -10**9
            while True:
                a = np.maximum(k0, last_end + self.gap)
                ok = a + self.m_min <= k1
                if not ok.any():
                    break
                last_end = int((a[ok] + self.m_min).min())
                out[i] += 1
        return out

    def obs_value(self, target, m, first: bool) -> np.ndarray:
        v = self.vs * self.P[target] * self.frac[m]
        return v + self.vc if first else v
