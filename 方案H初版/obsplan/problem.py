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
    k_floor: int = 3         # k_max 的下限：由原问题重建的问题（剩余、裁剪、子问题、放宽）沿用原问题的 k_max，见 rebuild

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

    @property
    def k_max(self) -> int:
        """每颗目标最多排几次观测（各算法里按目标记录观测的数组宽度）。6 天规划期里两两隔 48 小时最多 3 次；
        7 天（规范 v3.0 的默认规划期）里大多数目标能排 4 次。取 max(3, 最多能排的次数)，6 天时与原来完全相同。"""
        return max(self.k_floor, int(self.max_obs.max())) if len(self.max_obs) else self.k_floor

    def rebuild(self, win: Windows, cfg: Config | None = None) -> "Problem":
        """用新的窗口（剩余问题、按天气裁剪、子问题、按秒放宽问题）重建问题。新问题按自己的窗口重算 max_obs，
        但按目标记录观测的数组宽度 k_max 不小于原问题的：从原计划带过来的观测（冻结的、固定的、放回网格的）
        一颗目标最多可能有原问题的 k_max 次，剩余窗口算出来的上限可能更小（7 天计划过了一天多以后只剩 3 次）。"""
        p = Problem.build(self.cfg if cfg is None else cfg, self.ds, win)
        p.k_floor = max(p.k_floor, self.k_max)
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
