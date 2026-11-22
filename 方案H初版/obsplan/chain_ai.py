"""弹射链的 AI 预测器：试一颗目标之前，先用小神经网络估计"这颗能不能成功加上一次观测"。

弹射链一遍要试一千多颗高优先级目标，约两成找不到划算的链；这些失败的尝试往往要把大半份计划搜一遍，
占了一遍弹射链将近四成的时间。预测器用十来个几微秒就能算出的特征（还剩几个满足 48 小时间隔的窗口、
能不能直接塞进空当、最大空当多大、窗口里有没有可以删掉的低价值重复观测、之前失败过几次……）
给每颗目标打一个成功概率，弹射链按概率从高到低去试（时间有限时先拿容易的分），概率很低的直接跳过。

训练数据来自弹射链自己的运行记录（collect）：每试一颗目标记下当时的特征和成败。
网络与评审网络共用 obsplan/judge.py 里的 numpy 多层感知机（标准化 + Adam），模型存在 obsplan/models/chain_skip.npz。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from numba import njit

from .chain import Timeline, _slots

FEATURES = ["已有观测次数", "候选窗口数", "满足48小时的候选数", "能直接塞进空当", "最大空当（秒，截断）",
            "有空当的位置数", "平均候选区间长（秒）", "可删的低价值重复观测数", "之前失败次数", "最大空当−120秒是否≥0的个数"]
MODEL_PATH = Path(__file__).resolve().parent / "models" / "chain_skip.npz"


@njit(cache=True)
def _feats(x, c_ptr, c_unit, c_l, c_r, tk_s, tk_e, tk_tg, tg_tasks, tg_cnt, P, gap, dmin, seq2, A2, B2, ulen,
           unit_ok, out):
    sa = np.empty(4, np.int64)
    sb = np.empty(4, np.int64)
    n_cand = 0
    n_slot = 0
    direct = 0
    max_room = -100000
    n_pos = 0
    n_fit = 0
    len_sum = 0.0
    n_drop = 0
    for ci in range(c_ptr[x], c_ptr[x + 1]):
        u = c_unit[ci]
        if not unit_ok[u]:
            continue
        n_cand += 1
        len_sum += c_r[ci] - c_l[ci]
        ns = _slots(x, -1, c_l[ci], c_r[ci], tk_s, tk_e, tg_tasks, gap, dmin, sa, sb)
        if ns > 0:
            n_slot += 1
        n = ulen[u]
        for g in range(ns):
            a, b = sa[g], sb[g]
            lo, hi = 0, n
            while lo < hi:                      # 第一个 B[k] >= a 的空当
                mid = (lo + hi) // 2
                if B2[u, mid] >= a:
                    hi = mid
                else:
                    lo = mid + 1
            k = lo
            while k <= n and A2[u, k] <= b:
                room = min(b, B2[u, k]) - max(a, A2[u, k])
                if room > max_room:
                    max_room = room
                if room >= 0:
                    n_pos += 1
                if room >= dmin:
                    direct = 1
                    n_fit += 1
                if k < n:                        # 空当后面那个任务：低价值的重复观测可以直接删掉腾位置
                    y = seq2[u, k]
                    if y >= 0 and P[tk_tg[y]] < P[x] and tg_cnt[tk_tg[y]] >= 2:
                        n_drop += 1
                k += 1
    out[0] = tg_cnt[x]
    out[1] = n_cand
    out[2] = n_slot
    out[3] = direct
    out[4] = max(min(max_room, 600), -3600)
    out[5] = n_pos
    out[6] = len_sum / max(n_cand, 1)
    out[7] = n_drop
    out[9] = n_fit


def features(tl: Timeline, xs) -> np.ndarray:
    """当前状态下一批目标的特征（每颗几微秒）。"""
    xs = np.asarray(xs, np.int64)
    F = np.zeros((len(xs), len(FEATURES)), np.float64)
    fails = getattr(tl, "fail_count", None)
    for i, x in enumerate(xs):
        _feats(int(x), tl.c_ptr, tl.c_unit, tl.c_l, tl.c_r, tl.tk_s, tl.tk_e, tl.tk_tg, tl.tg_tasks, tl.tg_cnt, tl.P,
               tl.gap, tl.dmin, tl.seq2, tl.A2, tl.B2, tl.ulen, tl.unit_ok, F[i])
        F[i, 8] = 0 if fails is None else fails[x]
    return F


def collect(tl: Timeline, xs, **kw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按顺序逐颗试（与 run_pass 相同），记下每次尝试前的特征、成败和用时。"""
    F, y, t = [], [], []
    for x in np.asarray(xs, np.int64):
        if tl.tg_cnt[x] >= 3:
            continue
        F.append(features(tl, [x])[0])
        t0 = time.perf_counter()
        g = tl.try_add(int(x), **kw)
        t.append(time.perf_counter() - t0)
        y.append(1.0 if g > 0 else 0.0)
    return np.array(F), np.array(y), np.array(t)


class Predictor:
    """加载训练好的模型，给目标打成功概率。"""

    def __init__(self, path: Path | str | None = None):
        z = np.load(path or MODEL_PATH, allow_pickle=False)
        self.p = {k: z[k] for k in z.files if k[0] in "Wb" and k[1:].isdigit()}
        self.mean, self.std = z["mean"], z["std"]
        self.meta = json.loads(str(z["meta"]))

    def prob(self, F: np.ndarray) -> np.ndarray:
        from .judge import forward, standardize
        return 1.0 / (1.0 + np.exp(-forward(self.p, standardize(F, self.mean, self.std))))


def load_predictor(path: Path | str | None = None) -> Predictor | None:
    p = Path(path) if path else MODEL_PATH
    return Predictor(p) if p.exists() else None


def save(path: Path, p: dict, mean: np.ndarray, std: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean.astype(np.float32), std=std.astype(np.float32),
             meta=np.array(json.dumps(meta, ensure_ascii=False)), **p)
