"""方案 D：遗传算法学习 AI 价值函数的 8 个权重（规划方案 7.4、7.5 节）。

训练场景按方案 7.5 构造：随机抽一部分目标、随机屏蔽一部分站点，多场景取平均，
每个个体在每个场景上跑一次完整贪心调度，适应度 = 各场景综合指标相对人工权重的平均提升。
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from .greedy import MANUAL_WEIGHTS, ai_values, candidates, feature_matrix, run_greedy, window_features
from .lagrange import objective
from .problem import Problem
from .visibility import Windows


def sub_problem(pb: Problem, rng, target_frac=0.3, n_st=3) -> Problem:
    keep_t = rng.random(pb.n_targets) < target_frac
    keep_s = np.zeros(pb.n_stations, bool)
    keep_s[rng.choice(pb.n_stations, n_st, replace=False)] = True
    sel = keep_t[pb.w_tg] & keep_s[pb.w_st]
    win = pb.win
    w2 = Windows(pb.w_st[sel].astype(np.int16), pb.w_tg[sel].astype(np.int32), pb.w_k0[sel].astype(np.int32),
                 pb.w_k1[sel].astype(np.int32), win.dark, win.n_samples, win.sgp4_failed)
    return Problem.build(pb.cfg, pb.ds, w2)


class Scenario:
    def __init__(self, pb: Problem):
        self.pb = pb
        self.cand = candidates(pb)
        self.F = feature_matrix(pb, window_features(pb), *self.cand)

    def evaluate(self, w) -> float:
        v1, v2 = ai_values(self.F, w)
        st = run_greedy(self.pb, *self.cand, v1, v2, best_fit=True)
        return objective(self.pb, st)


def train(pb: Problem, n_scen=3, pop=16, gens=12, seed=2026, verbose=True) -> tuple[np.ndarray, dict]:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    scens = [Scenario(sub_problem(pb, rng)) for _ in range(n_scen)]
    base = np.array([s.evaluate(MANUAL_WEIGHTS) for s in scens])

    def fitness(w):
        return float(np.mean([s.evaluate(w) for s in scens] / base))

    P = [MANUAL_WEIGHTS.copy()] + [rng.uniform(0, 2, 8) for _ in range(pop - 1)]
    fit = np.array([fitness(w) for w in P])
    hist = []
    for g in range(gens):
        order = np.argsort(-fit)
        new = [P[order[0]].copy(), P[order[1]].copy()]          # 精英保留
        while len(new) < pop:
            def pick():
                c = rng.choice(pop, 3, replace=False)
                return P[c[np.argmax(fit[c])]]
            p1, p2 = pick(), pick()
            alpha = rng.uniform(-0.25, 1.25, 8)                    # BLX 交叉
            child = p1 + alpha * (p2 - p1)
            child += rng.normal(0, 0.15, 8) * (rng.random(8) < 0.3)  # 变异
            new.append(np.clip(child, 0, 2))
        P = new
        fit = np.array([fitness(w) for w in P])
        hist.append((g, float(fit.max()), float(fit.mean())))
        if verbose:
            print(f"  [GA] 第{g + 1:2d}代 最好 {fit.max():.4f} 平均 {fit.mean():.4f}  {time.time() - t0:.0f}s")
    best = P[int(np.argmax(fit))]
    return best, {"history": hist, "train_s": time.time() - t0, "fitness": float(fit.max())}
