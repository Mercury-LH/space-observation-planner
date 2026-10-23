"""伐谋评测（第二个演化对象）：方案 H 的"价格引导放置规则"。

H 的可行解恢复（`lagrange.recover`）是一个两遍贪心：第一遍按"价值 − 价格"从高到低放置正收益的候选，
第二遍按真实价值补空；之后再做共用的局部搜索。第一遍的排序规则是人工写的，这里交给伐谋演化。
候选代码只给每个候选打两个分（目标未覆盖时 / 已覆盖时），放置与全部硬约束仍由本地引擎保证。

除了原有的窗口特征，还给出价格协调本身的信息：候选的最低价格成本、所在窗口的需求拥挤度，
以及每颗目标在当前价格下"子问题"的选择（是否选了这个窗口、选了几次、净收益）。

评测：对每个子场景先用 H 跑 300 次价格迭代得到收敛价格（固定，不随候选变化），
再用候选的打分做两遍贪心 + 局部搜索，按综合指标与人工规则比较：
combined_score = 0.5 × 各场景(候选 / 人工规则)的平均，人工规则本身得 0.5。
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from .config import Config
from .data import load
from .famou_eval import CandidateFailure, run_candidate
from .ga import sub_problem
from .greedy import candidates, feature_matrix, run_greedy, window_features
from .lagrange import _candidate_cost, _demand, _subproblem, objective, solve
from .local_search import improve
from .problem import Problem
from .visibility import get_windows

FEATURES = ["priority", "duration_cells", "task_score", "cover_value", "price_cost", "price_per_cell",
            "window_cells", "slack_cells", "target_windows", "scarcity", "urgency", "conflict", "night_pos",
            "time_frac", "max_obs", "sub_pick", "sub_pick_same_duration", "sub_count", "target_dual",
            "window_demand"]
# 子场景（目标比例, 站数, 种子）。v1 为第一次实验所用：20% 目标 × 3 站，比全量宽松，偏重覆盖的规则在这里被高估，
# 到全量上不成立；v2 取全部目标、只抽 3–4 个站，拥挤程度与全量一致，各规则的优劣排序与全量相同。
CASE_SETS = {
    "v1": {"train": ((0.2, 3, 101), (0.2, 3, 202)), "holdout": ((0.2, 3, 303), (0.2, 3, 404), (0.2, 3, 505))},
    "v2": {"train": ((1.0, 3, 11), (1.0, 4, 11)), "holdout": ((1.0, 3, 22), (1.0, 4, 22), (0.8, 6, 22))},
}
TRAIN_SEEDS = tuple(x[2] for x in CASE_SETS["v1"]["train"])        # 兼容旧调用
HOLDOUT_SEEDS = tuple(x[2] for x in CASE_SETS["v1"]["holdout"])


def baseline_rule(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """人工规则（与 lagrange.recover 第一遍相同）：未覆盖 = 得分 + 覆盖价值 − 价格成本，已覆盖 = 得分 − 价格成本。"""
    c = {n: i for i, n in enumerate(FEATURES)}
    ts, cv, pc = rows[:, c["task_score"]], rows[:, c["cover_value"]], rows[:, c["price_cost"]]
    return ts + cv - pc, ts - pc


def price_features(pb: Problem, mu: np.ndarray, cw: np.ndarray, cm: np.ndarray) -> np.ndarray:
    """在价格 mu 下给每个候选 (窗口, 时长) 算特征，单位换成"分"（与 task_score 相同）。"""
    S, T = mu.shape
    vs = pb.vs
    pref = np.concatenate([np.zeros((S, 1)), np.cumsum(mu, axis=1)], axis=1)
    cost = _candidate_cost(pref, cw, cm, pb.w_st, pb.w_k0, pb.w_k1) / vs
    # 子问题：每颗目标在价格 mu 下的最优观测链
    N = pb.n_targets
    out_total = np.zeros(N)
    out_task = np.zeros((N, 3, 3), np.int64)
    out_cnt = np.zeros(N, np.int64)
    _subproblem(pref, pb.t_ptr, pb.t_win, pb.w_st, pb.w_k0, pb.w_k1, vs * pb.P, pb.frac, np.full(N, pb.vc), pb.gap,
                pb.m_min, pb.m_max, out_total, out_task, out_cnt, np.zeros((1, 1)), False, np.zeros(N),
                np.full(N, 3, np.int64))
    tg = pb.w_tg[cw]
    st, k0, k1 = pb.w_st[cw], pb.w_k0[cw], pb.w_k1[cw]
    pick = np.zeros(len(cw), bool)
    pick_m = np.zeros(len(cw), bool)
    for c in range(3):
        ok = out_cnt[tg] > c
        s_c, a_c, m_c = out_task[tg, c, 0], out_task[tg, c, 1], out_task[tg, c, 2]
        hit = ok & (s_c == st) & (a_c >= k0) & (a_c + m_c <= k1)
        pick |= hit
        pick_m |= hit & (m_c == cm)
    dem = _demand(out_task, out_cnt, S, T)
    dpref = np.concatenate([np.zeros((S, 1)), np.cumsum(dem, axis=1)], axis=1)
    wdem = (dpref[st, k1 + 1] - dpref[st, k0]) / (k1 - k0 + 1)
    return np.column_stack([cost, cost / (cm + 1), pick, pick_m, out_cnt[tg], out_total[tg] / vs, wdem])


def night_position(pb: Problem) -> np.ndarray:
    """窗口在所处黑夜段中的位置：0/1 靠近黄昏/黎明，0.5 为午夜前后（与 famou_eval 相同）。"""
    night = np.full(len(pb.w_k0), 0.5)
    for s, runs in enumerate(pb.win.dark):
        if not runs:
            continue
        d0 = np.array([a for a, _ in runs])
        d1 = np.array([b for _, b in runs])
        idx = np.nonzero(pb.w_st == s)[0]
        mid = (pb.w_k0[idx] + pb.w_k1[idx]) / 2
        r = np.clip(np.searchsorted(d0, mid, side="right") - 1, 0, len(d0) - 1)
        night[idx] = np.clip((mid - d0[r]) / np.maximum(d1[r] - d0[r], 1), 0, 1)
    return night


def feature_rows(pb: Problem, mu: np.ndarray, cw: np.ndarray, cm: np.ndarray) -> np.ndarray:
    tg = pb.w_tg[cw]
    wf = window_features(pb)
    F = feature_matrix(pb, wf, cw, cm)
    ts = pb.P[tg] * pb.frac[cm]
    pf = price_features(pb, mu, cw, cm)
    wc = (pb.w_k1 - pb.w_k0)[cw]
    X = np.column_stack([
        pb.P[tg], cm, ts, np.full(len(cw), pb.vc / pb.vs), pf[:, 0], pf[:, 1], wc, wc - cm, wf["n_win"][tg],
        F[:, 3], F[:, 4], F[:, 6], night_position(pb)[cw], pb.w_k0[cw] / pb.n_samples, pb.max_obs[tg],
        pf[:, 2], pf[:, 3], pf[:, 4], pf[:, 5], pf[:, 6],
    ])
    return np.round(X, 4)


def run_rule(pb: Problem, mu: np.ndarray, cw, cm, u: np.ndarray, c: np.ndarray, local_search=True):
    """两遍贪心（第一遍用规则打分、只放正值；第二遍按真实价值补空）+ 局部搜索，返回状态。"""
    v = pb.vs * pb.P[pb.w_tg[cw]] * pb.frac[cm]
    st = run_greedy(pb, cw, cm, u, np.minimum(u, c), best_fit=True, cutoff=0.0, price=mu)
    run_greedy(pb, cw, cm, v + pb.vc, v, best_fit=True, state=st, price=mu)
    if local_search:
        improve(pb, st)
    return st


class HCase:
    def __init__(self, pb: Problem, iters: int = 300):
        self.pb = pb
        _, info = solve(pb, iters=iters, verbose=False, local_search=False)
        self.mu = info["mu"]
        self.cw, self.cm = candidates(pb)
        self.X = feature_rows(pb, self.mu, self.cw, self.cm)
        self.base = self.objective(*baseline_rule(self.X))

    @property
    def rows(self):
        return self.X.tolist()

    def objective(self, u, c) -> float:
        return objective(self.pb, run_rule(self.pb, self.mu, self.cw, self.cm, np.asarray(u, float),
                                           np.asarray(c, float)))


class FullCase:
    """全量实例（10000 目标 × 10 站 × 6 天）：价格固定为人工规则 H 跑 300 次迭代的结果；
    候选规则放置 → 快速局部搜索 → 按秒精修（finish）→ 综合指标。赛题只有这一个实例，直接在它上面评测最可靠。"""

    def __init__(self, pb: Problem, iters: int = 300):
        self.pb = pb
        _, info = solve(pb, iters=iters, verbose=False, local_search=False)
        self.mu = info["mu"]
        self.cw, self.cm = candidates(pb)
        self.X = feature_rows(pb, self.mu, self.cw, self.cm)
        self.base = self.objective(*baseline_rule(self.X))

    @property
    def rows(self):
        return self.X.tolist()

    def schedule(self, u, c):
        from .refine import finish
        st = run_rule(self.pb, self.mu, self.cw, self.cm, np.asarray(u, float), np.asarray(c, float),
                      local_search=False)
        improve(self.pb, st, repeat_min_priority=3.0)
        ref, _ = finish(self.pb, st.schedule(self.pb, "H + 伐谋放置规则", 0.0))
        return ref

    def objective(self, u, c) -> float:
        from .schedule import evaluate
        return float(evaluate(self.pb.cfg, self.pb.ds, self.schedule(u, c))["combined"])


def load_full_case(cfg: Config | None = None) -> FullCase:
    cfg = cfg or Config()
    path = _cache_path(cfg, "full")
    if path.exists():
        return pickle.loads(path.read_bytes())
    ds = load(cfg)
    case = FullCase(Problem.build(cfg, ds, get_windows(cfg, ds)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(case))
    return case


def _cache_path(cfg: Config, tag: str) -> Path:
    return Path(cfg.cache_dir) / f"famou_h_cases_{tag}_{cfg.scenario_key()}.pkl"


def load_cases(cfg: Config | None = None, seeds=TRAIN_SEEDS, tag="train", version: str | None = None) -> list[HCase]:
    """version 给定时按 CASE_SETS[version][tag] 建子场景；否则按旧接口（20% 目标 × 3 站 × seeds）。"""
    cfg = cfg or Config()
    specs = CASE_SETS[version][tag] if version else tuple((0.2, 3, s) for s in seeds)
    path = _cache_path(cfg, f"{version}_{tag}" if version else tag)
    if path.exists():
        return pickle.loads(path.read_bytes())
    ds = load(cfg)
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    cases = [HCase(sub_problem(pb, np.random.default_rng(s), target_frac=f, n_st=n)) for f, n, s in specs]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(cases))
    return cases


def load_payload(cfg: Config | None = None, version: str = "v2", tag: str = "train") -> bytes:
    """送进沙箱的输入 JSON（上百万行，预先写到缓存目录，评测时直接读）。"""
    cfg = cfg or Config()
    path = _cache_path(cfg, f"{version}_{tag}").with_suffix(".json")
    if not path.exists():
        cases = [load_full_case(cfg)] if version == "full" else load_cases(cfg, version=version, tag=tag)
        path.write_bytes(json.dumps({"names": FEATURES, "cases": [c.rows for c in cases]}).encode())
    return path.read_bytes()


def evaluate_source(source: bytes, cases: list[HCase] | None = None, version: str | None = None,
                    timeout_s: float = 40.0) -> dict:
    """version："v1"/"v2" 为子场景，"full" 为全量实例（综合指标含按秒精修）。"""
    t0 = time.time()
    payload = load_payload(version=version) if (version and cases is None) else None
    if cases is None:
        cases = [load_full_case()] if version == "full" else load_cases(version=version, tag="train")
    try:
        result = run_candidate(source, cases, names=FEATURES, timeout_s=timeout_s, payload=payload,
                               fsize=256 << 20)
        if not isinstance(result, list) or len(result) != len(cases):
            raise CandidateFailure("返回格式应为每个场景一对 (uncovered, covered) 列表")
        ratios = []
        for case, (u, c) in zip(cases, result):
            n = len(case.cw)
            if len(u) != n or len(c) != n:
                raise CandidateFailure("打分列表长度必须等于候选数")
            v1, v2 = np.asarray(u, float), np.asarray(c, float)
            if not (np.isfinite(v1).all() and np.isfinite(v2).all()):
                raise CandidateFailure("打分里有 NaN 或无穷大")
            ratios.append(case.objective(v1, v2) / case.base)
        ratio = float(np.mean(ratios))
        return {"combined_score": float(min(1.0, max(0.0, 0.5 * ratio))), "validity": 1,
                "ratio_vs_manual": ratio, "case_ratios": [round(r, 6) for r in ratios],
                "eval_s": round(time.time() - t0, 2)}
    except CandidateFailure as e:
        return {"combined_score": 0.0, "validity": 0, "error_info": str(e)[:800]}
