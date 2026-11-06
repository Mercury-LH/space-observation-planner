"""突发事件下的动态重规划：冻结已执行的任务，按事件更新问题，用价格协调重排之后的时段，并统计对原计划的扰动。

正式赛题没有突发情况（答疑 #32），这里作为产品演示与加分项，对应企业赛题初稿里
"持续监测 → 状态更新 → 动态重规划"的闭环和"尽量降低对既有计划的扰动"的要求。

三类事件（事件时刻 t：相对规划开始的秒数）：
- down：某站从 t 起停机若干小时。t 时正在该站执行的任务中断，停机时段内该站的任务取消；
- priority：某目标从 t 起升为高优先级，计分按新优先级；
- must：新增一个必须观测的目标，t 之后至少安排一次（t 之前观测过的，须与上次观测间隔 48 小时）。

两种重排方式，都先冻结 t 之前开始的任务（中断的除外），在网格上规划 t 之后的时段，最后按秒精修：
- minimal（扰动最小，默认）：原计划 t 之后仍然可行的任务先原样放回，价格协调只往空档里补任务；
  然后只为受事件影响的目标做局部修补（挤让、缩短、挪走或删掉少量价值低的任务）。
  按秒精修时冻结的和原样保留的任务固定不动，只有与新任务挨得太近的才跟着微调；
- full（全面重排）：t 之后从头规划（与"每天重排"同样的做法），改动大；它用人工放置规则（伐谋规则不支持
  按目标给覆盖奖励），在默认的 H+ 结果上实测综合指标反而比扰动最小略低。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import timedelta

import numpy as np

from . import lagrange
from .checks import structural_violations
from .dynamic import Status, clip_cells, residual_problem, state_factory
from .greedy import candidates, run_greedy
from .lns import state_from_schedule
from .local_search import _attach, extend_all, repair_coverage
from .problem import Problem
from .refine import refine, true_bounds
from .schedule import Schedule, evaluate, score_ref

KINDS = {"down": "站点停机", "priority": "目标升为高优先级", "must": "新增必须观测的目标"}
MODES = {"minimal": "扰动最小", "full": "全面重排"}
STATUS_ORDER = ["冻结", "保留", "移动", "删除", "新增"]
TOL_S = 60            # 同一站、开始与结束时刻都相差不超过 60 秒，算"保留"
MUST_VC = 100.0       # 必须观测目标的首次观测奖励（覆盖奖励的倍数），价格协调时用
MUST_P = 1000.0       # 强行插入必须观测目标时它的临时优先级分，足以挤开任何两个任务


@dataclass(frozen=True)
class Event:
    kind: str                 # "down" | "priority" | "must"
    t_s: int                  # 事件时刻：相对规划开始的秒数
    station: int = -1         # down：站点下标
    hours: float = 0.0        # down：停机时长（小时）
    target: int = -1          # priority / must：目标下标
    level: str = "高"         # priority：新的优先级

    @property
    def down(self) -> tuple[int, int]:
        return int(self.t_s), int(self.t_s + round(self.hours * 3600))

    def describe(self, cfg, ds) -> str:
        when = f"{cfg.start_dt + timedelta(seconds=int(self.t_s)):%m-%d %H:%M} UTC"
        if self.kind == "down":
            return f"{when} 起 {self.station + 1} 号站停机 {self.hours:g} 小时"
        name = f"目标 {ds.satnum[self.target]}"
        if self.kind == "priority":
            return f"{when} 起{name}由{ds.priority_label[self.target]}优先级升为{self.level}优先级"
        return f"{when} 新增必须观测的{name}（{ds.priority_label[self.target]}优先级）"


def suggest_targets(pb: Problem, sch: Schedule, t_s: int, kind: str, limit: int = 300) -> list[int]:
    """事件目标的推荐（界面与实验脚本共用）。
    priority：原计划只观测 1 次、这次在事件之后、48 小时后还有窗口的中低优先级目标（升级后有机会多观测一次），后续窗口多的在前；
    must：事件之后没有观测、但还有可行窗口的目标，原计划没覆盖的在前，其次按优先级。"""
    cell, gap = pb.cfg.cell_s, int(pb.cfg.repeat_gap_h * 3600)
    start = sch.start_s.astype(np.int64)
    end = start + sch.dur_s.astype(np.int64)
    tg = sch.target.astype(np.int64)
    cnt = np.bincount(tg, minlength=pb.n_targets)
    out = []
    if kind == "priority":
        top = max(pb.cfg.priority_score.values())
        for i in np.nonzero((start >= t_s) & (pb.P[tg] < top) & (cnt[tg] == 1))[0]:
            x = int(tg[i])
            ws = pb.t_win[pb.t_ptr[x]:pb.t_ptr[x + 1]]
            n_later = int(((pb.w_k1[ws] - pb.m_min) * cell >= end[i] + gap).sum())
            if n_later:
                out.append((-n_later, int(start[i]), x))
    else:
        after = np.zeros(pb.n_targets, bool)
        after[tg[start >= t_s]] = True
        last = np.full(pb.n_targets, -10 ** 12, np.int64)
        np.maximum.at(last, tg, end)
        for x in np.nonzero((pb.max_obs > 0) & ~after & (cnt < 3))[0]:
            lo = max(t_s, int(last[x]) + gap) if cnt[x] else t_s
            ws = pb.t_win[pb.t_ptr[x]:pb.t_ptr[x + 1]]
            if np.any((pb.w_k1[ws] - pb.m_min) * cell >= lo + 600):
                out.append((int(cnt[x] > 0), -float(pb.P[x]), int(x)))
    return [x for *_, x in sorted(out)[:limit]]


def busiest_station(sch: Schedule, n_stations: int, t_s: int, hours: float) -> int:
    """事件之后 hours 小时内任务最多的站点（演示停机事件用）。"""
    sel = (sch.start_s >= t_s) & (sch.start_s < t_s + hours * 3600)
    return int(np.bincount(sch.station[sel].astype(np.int64), minlength=n_stations).argmax())


def event_problem(pb: Problem, ev: Event) -> Problem:
    """事件改了优先级时，返回按新优先级计分的问题（窗口不变）；否则原样返回。"""
    if ev.kind != "priority":
        return pb
    ds = pb.ds
    pr, lab = ds.priority.copy(), list(ds.priority_label)
    pr[ev.target], lab[ev.target] = pb.cfg.priority_score[ev.level], ev.level
    ds2 = replace(ds, priority=pr, priority_label=lab)
    return replace(pb, ds=ds2, P=pr.astype(np.float64), vs=pb.cfg.w_score / score_ref(pb.cfg, ds2))


def _ceil(x, c):
    return -(-np.asarray(x, np.int64) // c)


def _subset(sch: Schedule, idx) -> Schedule:
    idx = np.asarray(idx, np.int64)
    return Schedule(sch.station[idx].astype(np.int64), sch.target[idx].astype(np.int64),
                    sch.start_s[idx].astype(np.int64), sch.dur_s[idx].astype(np.int64), algo=sch.algo)


# ------------------------------------------------------------------ 原样保留的任务放回网格
def _snap(pr: Problem, sch: Schedule, idx: np.ndarray, base_owner: np.ndarray, succ: dict):
    """把原计划 t 之后的任务（按秒）就近放回网格：起止各取最近的网格点，再挪进所在窗口；与邻居冲突时把网格上的
    占位缩短一两格再试（输出时没改动的任务仍用原来的按秒时刻）。
    放不回的（窗口对不上、缩短后仍冲突或不满足 48 小时）作为固定任务：不参与调整，占用的格整格封锁。
    返回 grid：[(原下标, 站, 目标, a, m, 窗口)]，fixed：原下标列表。"""
    cfg = pr.cfg
    cell, half = cfg.cell_s, cfg.cell_s // 2
    start = sch.start_s.astype(np.int64)
    end = start + sch.dur_s.astype(np.int64)
    prov = {}
    for i in idx:
        s, t = int(sch.station[i]), int(sch.target[i])
        a, e = int((start[i] + half) // cell), int((end[i] + half) // cell)
        ws = pr.t_win[pr.t_ptr[t]:pr.t_ptr[t + 1]]
        ws = ws[(pr.w_st[ws] == s) & (pr.w_k0[ws] <= e) & (pr.w_k1[ws] >= a)]
        if not len(ws):
            continue
        w = int(ws[np.argmax(np.minimum(pr.w_k1[ws], e) - np.maximum(pr.w_k0[ws], a))])
        k0, k1 = int(pr.w_k0[w]), int(pr.w_k1[w])
        if a < k0:
            a = k0
            e = max(e, a + pr.m_min)
        if e > k1:
            e = k1
            a = min(a, e - pr.m_min)
        e = min(e, a + pr.m_max)
        if a >= k0 and e <= k1 and e - a >= pr.m_min:
            prov[int(i)] = (s, t, a, e - a, w)
    fixed = sorted(set(int(i) for i in idx) - set(prov))
    while True:
        owner = base_owner.copy()
        obs = {t: list(v) for t, v in succ.items()}
        for i in fixed:
            s = int(sch.station[i])
            owner[s, start[i] // cell:int(_ceil(end[i] + cfg.switch_s, cell))] = -3
            obs.setdefault(int(sch.target[i]), []).append((int(start[i] // cell), int(_ceil(end[i], cell) - start[i] // cell)))
        bad, placed = [], {}
        for i in sorted(prov, key=lambda i: (prov[i][0], prov[i][2])):
            s, t, a, m, w = prov[i]
            for a2, m2 in ((a, m), (a + 1, m - 1), (a, m - 1), (a + 1, m - 2)):
                if m2 >= pr.m_min and (owner[s, a2:a2 + m2 + 1] == -1).all() and \
                        all(a2 >= b + mb + pr.gap or a2 + m2 + pr.gap <= b for b, mb in obs.get(t, ())):
                    break
            else:
                bad.append(i)
                continue
            placed[i] = (s, t, a2, m2, w)
            owner[s, a2:a2 + m2 + 1] = i
            obs.setdefault(t, []).append((a2, m2))
        if not bad:
            break
        for i in bad:
            del prov[i]
        fixed = sorted(fixed + bad)
    grid = [(i, *placed[i]) for i in sorted(placed, key=lambda i: (placed[i][0], placed[i][2]))]
    return grid, fixed


def _factory(pr: Problem, status: Status, blocks, fixed_obs, grid):
    """返回 State 构造函数：过去的时段、停机和固定任务封锁，已执行与固定的观测记入目标，可调整的保留任务放好。"""
    base = state_factory(pr, status)

    def make():
        st = base()
        for s, c0, c1 in blocks:
            st.owner[s, c0:c1] = -3
        for t, a, m in fixed_obs:
            x = st.t_cnt[t]
            st.t_obs_a[t, x], st.t_obs_m[t, x] = a, m
            st.t_cnt[t] = x + 1
        for _, s, t, a, m, w in grid:
            k = st.n
            st.st[k], st.tg[k], st.a[k], st.m[k], st.w[k] = s, t, a, m, w
            st.alive[k] = True
            _attach(k, st.st, st.tg, st.a, st.m, st.owner, st.t_cnt, st.t_obs_a, st.t_obs_m, st.t_task)
            st.n += 1
        return st
    return make


def _repair_args(st):
    return (st.st, st.tg, st.a, st.m, st.w, st.alive, st.n, st.owner, st.t_cnt, st.t_obs_a, st.t_obs_m, st.t_task)


def _fixed_record(st, t: int, a: int, m: int, on: bool) -> None:
    """在目标 t 的观测记录里加上或去掉一条固定观测（t_task 为 -1 的记录）。"""
    c = st.t_cnt[t]
    if on:
        st.t_obs_a[t, c], st.t_obs_m[t, c], st.t_task[t, c] = a, m, -1
        st.t_cnt[t] = c + 1
        return
    j = next(j for j in range(c) if st.t_task[t, j] == -1 and st.t_obs_a[t, j] == a and st.t_obs_m[t, j] == m)
    st.t_obs_a[t, j], st.t_obs_m[t, j], st.t_task[t, j] = st.t_obs_a[t, c - 1], st.t_obs_m[t, c - 1], st.t_task[t, c - 1]
    st.t_task[t, c - 1] = -1
    st.t_cnt[t] = c - 1


def _force(pr: Problem, st, x: int, after_cell: int, fixed_info: dict, protect) -> tuple[bool, list[int]]:
    """必须观测：x 在 t 之后还没有观测时，临时把它的价值调到极高，用覆盖修复挤出一个 120 秒的位置。
    挤不出来时，让开挡在 x 的窗口里的固定任务（原样保留但对不上网格的任务，还没执行）再试；没被占用的固定任务原样放回。
    fixed_info：{原下标: (站, 封锁起格, 封锁止格, 目标, 记录 a, 记录 m)}；protect()：重新封锁过去与停机时段。
    返回 (是否排上, 被让开、不再保留的固定任务)。"""
    alive = np.nonzero(st.alive[:st.n])[0]
    if np.any((st.tg[alive] == x) & (st.a[alive] >= after_cell)) or any(v[3] == x for v in fixed_info.values()):
        return True, []
    if st.t_cnt[x] >= 3:
        return False, []
    P = pr.P.copy()
    P[x] = MUST_P

    def insert() -> bool:
        st.n, ins = repair_coverage(np.array([x], np.int64), *_repair_args(st), pr.w_st, pr.w_k0, pr.w_k1, pr.t_ptr,
                                    pr.t_win, P, pr.frac, pr.vc, pr.vs, pr.gap, pr.m_min, 1 << 30)
        return ins > 0
    if insert():
        return True, []
    ws = pr.t_win[pr.t_ptr[x]:pr.t_ptr[x + 1]]
    rel = [f for f, (s, c0, c1, *_) in fixed_info.items()
           if np.any((pr.w_st[ws] == s) & (pr.w_k0[ws] < c1) & (pr.w_k1[ws] >= c0))]
    if not rel:
        return False, []
    for f in rel:
        s, c0, c1, t, a, m = fixed_info[f]
        st.owner[s, c0:c1] = -1
        _fixed_record(st, t, a, m, on=False)
    for f, (s, c0, c1, *_) in fixed_info.items():          # 相邻固定任务共用的边界格、过去与停机时段仍然封锁
        if f not in rel:
            st.owner[s, c0:c1] = -3
    protect(st)
    ok = insert()
    gone = []
    for f in rel:
        s, c0, c1, t, a, m = fixed_info[f]
        sep = all(a >= st.t_obs_a[t, j] + st.t_obs_m[t, j] + pr.gap or a + m + pr.gap <= st.t_obs_a[t, j]
                  for j in range(st.t_cnt[t]))
        if (st.owner[s, c0:c1] < 0).all() and st.t_cnt[t] < 3 and sep:      # 没有被任务占用（边界格可能已封锁）
            st.owner[s, c0:c1] = -3
            _fixed_record(st, t, a, m, on=True)
        else:
            gone.append(f)
    return ok, gone


def _local_repair(pr: Problem, st, focus: np.ndarray, n_keep: int, snap: dict):
    """扰动最小的修补：只为未覆盖的目标和受事件影响的目标挤让少量任务，再往空档里补任务，最后只延长新增和改动过的任务。"""
    has_win = np.diff(pr.t_ptr) > 0
    unc = np.nonzero((st.t_cnt == 0) & has_win)[0]
    unc = unc[np.argsort(-pr.P[unc], kind="stable")]
    st.n, _ = repair_coverage(unc, *_repair_args(st), pr.w_st, pr.w_k0, pr.w_k1, pr.t_ptr, pr.t_win,
                              pr.P, pr.frac, pr.vc, pr.vs, pr.gap, pr.m_min, 40)
    rep = focus[(st.t_cnt[focus] > 0) & (st.t_cnt[focus] < 3)]
    rep = rep[np.argsort(-pr.P[rep], kind="stable")]
    st.n, _ = repair_coverage(rep, *_repair_args(st), pr.w_st, pr.w_k0, pr.w_k1, pr.t_ptr, pr.t_win,
                              pr.P, pr.frac, pr.vc, pr.vs, pr.gap, pr.m_min, 40)
    cw, cm = candidates(pr, [pr.m_min])
    v = pr.vs * pr.P[pr.w_tg[cw]] * pr.frac[cm]
    run_greedy(pr, cw, cm, v + pr.vc, np.full(len(cw), -1.0), best_fit=True, state=st, cutoff=0.0)   # 新覆盖
    run_greedy(pr, cw, cm, v + pr.vc, v, best_fit=True, state=st)                                    # 重复观测
    fs = set(focus.tolist())
    changed = [k for k in range(st.n) if st.alive[k] and (k >= n_keep or snap[k] != (st.st[k], st.a[k], st.m[k])
                                                          or int(st.tg[k]) in fs)]
    order = np.array(sorted(changed, key=lambda k: -pr.P[st.tg[k]]), np.int64)
    extend_all(order, st.st, st.tg, st.a, st.m, st.w, st.alive, st.n, st.owner, st.t_cnt, st.t_obs_a, st.t_obs_m,
               st.t_task, pr.w_k0, pr.w_k1, pr.gap, pr.m_max)


# ------------------------------------------------------------------ 新旧计划对比
def compare_plans(old: Schedule, old_sel: np.ndarray, new: Schedule, new_sel: np.ndarray, tol_s: int = TOL_S):
    """逐目标配对 t 之后的任务：同站且起止都相差 ≤ tol_s 秒为"保留"，同一目标的其余任务按时间先后配成"移动"
    （换站、换时间或改时长），原计划剩下的为"删除"，新计划剩下的为"新增"。返回两份计划逐任务的状态。"""
    o_st = np.where(old_sel, "", "冻结").astype(object)
    n_st = np.where(new_sel, "", "冻结").astype(object)
    os_, oe = old.start_s.astype(np.int64), (old.start_s + old.dur_s).astype(np.int64)
    ns_, ne = new.start_s.astype(np.int64), (new.start_s + new.dur_s).astype(np.int64)
    groups: dict[int, tuple[list, list]] = {}
    for i in np.nonzero(old_sel)[0]:
        groups.setdefault(int(old.target[i]), ([], []))[0].append(int(i))
    for j in np.nonzero(new_sel)[0]:
        groups.setdefault(int(new.target[j]), ([], []))[1].append(int(j))
    for O, N in groups.values():
        pairs = sorted((abs(os_[i] - ns_[j]) + abs(oe[i] - ne[j]), i, j) for i in O for j in N
                       if old.station[i] == new.station[j] and abs(os_[i] - ns_[j]) <= tol_s and abs(oe[i] - ne[j]) <= tol_s)
        used_o, used_n = set(), set()
        for _, i, j in pairs:
            if i not in used_o and j not in used_n:
                o_st[i] = n_st[j] = "保留"
                used_o.add(i)
                used_n.add(j)
        ro = sorted((i for i in O if i not in used_o), key=lambda i: os_[i])
        rn = sorted((j for j in N if j not in used_n), key=lambda j: ns_[j])
        for i, j in zip(ro, rn):
            o_st[i] = n_st[j] = "移动"
        for i in ro[len(rn):]:
            o_st[i] = "删除"
        for j in rn[len(ro):]:
            n_st[j] = "新增"
    return o_st, n_st


def _unpin_conflicts(sch: Schedule, pinned: np.ndarray, grid_pos: dict, cfg) -> None:
    """按秒精修前：固定在原时刻的保留任务若与网格上的新任务挨得太近（不足切换时间或 48 小时），改为跟着微调。
    改为网格位置后可能又碰到它另一侧的邻居，所以反复检查直到没有冲突。原地修改 sch 与 pinned。"""
    gap = int(round(cfg.repeat_gap_h * 3600))
    while True:
        hit = set()
        for key, need in ((sch.station, cfg.switch_s), (sch.target, gap)):
            order = np.lexsort((sch.start_s, key))
            p, q = order[:-1], order[1:]
            bad = (key[p] == key[q]) & (sch.start_s[q] < sch.start_s[p] + sch.dur_s[p] + need)
            for i, j in zip(p[bad], q[bad]):
                hit.update(k for k in (int(i), int(j)) if pinned[k] and k in grid_pos)
        if not hit:
            return
        for k in hit:
            pinned[k] = False
            sch.start_s[k], sch.dur_s[k] = grid_pos[k]


# ------------------------------------------------------------------ 主流程
def replan(pb: Problem, sch: Schedule, ev: Event, mode: str = "minimal", iters: int = 150, fast: bool = True,
           refine_seconds: bool = True, extra_windows: dict | None = None, progress=None) -> dict:
    """在原计划 sch（可以是按秒精修过的）上处理事件 ev，返回新计划与对比统计。
    extra_windows：按秒补覆盖目标的真实可见区间（refine.near_windows），只用于结构检查。"""
    t_all = time.time()
    clock = [t_all]
    timing = {}

    def tick(name):
        now = time.time()
        timing[name] = round(timing.get(name, 0.0) + now - clock[0], 2)
        clock[0] = now

    def say(f, text):
        if progress is not None:
            progress(min(1.0, f), text)

    cfg = pb.cfg
    cell, sw = cfg.cell_s, cfg.switch_s
    pe = event_problem(pb, ev)
    S, n = pb.n_stations, len(sch)
    start = sch.start_s.astype(np.int64)
    end = start + sch.dur_s.astype(np.int64)
    t = int(ev.t_s)

    # ---- 1. 状态更新：哪些任务已执行（冻结）、中断、取消
    frozen = start < t
    interrupted = np.zeros(n, bool)
    cancelled = np.zeros(n, bool)
    if ev.kind == "down":
        da, db = ev.down
        overlap = (sch.station == ev.station) & (start < db) & (end > da)
        interrupted, cancelled = frozen & overlap, ~frozen & overlap
        frozen &= ~interrupted
    later = ~frozen & ~interrupted & ~cancelled
    fi = np.nonzero(frozen)[0]
    c_now = np.full(S, int(_ceil(t, cell)), np.int64)
    np.maximum.at(c_now, sch.station[fi].astype(np.int64), _ceil(end[fi] + sw, cell))
    succ: dict[int, list] = {}
    for i in fi[np.argsort(start[fi], kind="stable")]:
        a = int(start[i] // cell)
        succ.setdefault(int(sch.target[i]), []).append((a, int(_ceil(end[i], cell)) - a))
    status = Status(succ=succ, c_now=c_now)
    say(0.02, "状态更新：冻结已执行的任务，生成剩余问题……")
    pr = residual_problem(pe, status)
    blocks = []
    if ev.kind == "down":
        da, db = ev.down
        b0, b1 = da // cell, int(_ceil(db, cell))
        ok = np.ones((S, pb.n_samples + 2), bool)
        ok[ev.station, b0:b1] = False
        pr = clip_cells(pr, ok)
        blocks.append((ev.station, b0 + 1, b1))
    n_frozen = np.bincount(sch.target[fi].astype(np.int64), minlength=pb.n_targets)
    vct = np.where(n_frozen > 0, 0.0, pr.vc)
    if ev.kind == "must":
        vct[ev.target] = MUST_VC * pr.vc
    left = np.maximum(0, 3 - n_frozen)
    tick("剩余问题")

    # ---- 2. 扰动最小：t 之后仍然可行的原任务放回网格
    grid, fixed = [], []
    if mode == "minimal":
        base_owner = state_factory(pr, status)().owner
        for s, c0, c1 in blocks:
            base_owner[s, c0:c1] = -3
        grid, fixed = _snap(pr, sch, np.nonzero(later)[0], base_owner, succ)
    keep_blocks = list(blocks)                 # 停机时段（固定任务让开后要重新封锁）
    fixed_info = {i: (int(sch.station[i]), int(start[i] // cell), int(_ceil(end[i] + sw, cell)), int(sch.target[i]),
                      int(start[i] // cell), int(_ceil(end[i], cell) - start[i] // cell)) for i in fixed}
    blocks += [v[:3] for v in fixed_info.values()]
    fixed_obs = [v[3:] for v in fixed_info.values()]
    make = _factory(pr, status, blocks, fixed_obs, grid)

    def protect(st_):
        for s_ in range(S):
            st_.owner[s_, :c_now[s_]] = -3
        for s_, c0, c1 in keep_blocks:
            st_.owner[s_, c0:c1] = -3
    tick("放回原任务")

    # ---- 3. 价格协调重排 t 之后的时段
    def cb(f, text):                       # 只报迭代进度：迭代中的目标值是剩余时段的内部口径，与全计划指标不可比
        say(0.05 + 0.75 * f, "价格协调：" + text.split("，")[0])
    sch_g, info = lagrange.solve(pr, iters=iters, fast=fast, verbose=False, progress=cb, vc_target=vct,
                                 max_obs_left=left, init_state=make, local_search=mode == "full")
    tick("价格协调")

    # ---- 4. 事件相关的修补
    say(0.82, "局部修补：为受影响的目标挤出位置……")
    st = _factory(pr, status, blocks, fixed_obs, [])()
    state_from_schedule(pr, sch_g, st)
    n_keep = len(grid)
    snap = {k: (g[1], g[3], g[4]) for k, g in enumerate(grid)}
    for k, g in enumerate(grid):           # 价格协调只往空档里补任务，放回的任务排在最前、原封不动
        assert (st.st[k], st.tg[k], st.a[k], st.m[k]) == (g[1], g[2], g[3], g[4])
    must_ok, gone = None, []
    if ev.kind == "must":
        must_ok, gone = _force(pr, st, ev.target, int(_ceil(t, cell)), fixed_info, protect)
        fixed = [i for i in fixed if i not in set(gone)]
    focus = set()
    if mode == "minimal":
        focus = {int(x) for x in sch.target[interrupted | cancelled]} | {int(sch.target[i]) for i in gone}
        if ev.kind in ("priority", "must"):
            focus.add(ev.target)
        _local_repair(pr, st, np.array(sorted(focus), np.int64), n_keep, snap)
    elif ev.kind == "must":                 # 全面重排后强行插入挤掉的目标，再补一次
        _local_repair(pr, st, np.zeros(0, np.int64), 0, {})
    tick("局部修补")

    # ---- 5. 按秒精修：冻结、固定和原样保留的任务不动，其余任务按秒重新分配起止
    say(0.88, "按秒精修……")
    alive = np.nonzero(st.alive[:st.n])[0]
    unchanged = [int(k) for k in alive if k < n_keep and snap[k] == (st.st[k], st.a[k], st.m[k])]
    unchanged_set = set(unchanged)
    moved_k = [int(k) for k in alive if int(k) not in unchanged_set]
    rows = ([(int(i), "frozen") for i in fi] + [(int(i), "fixed") for i in fixed]
            + [(grid[k][0], "kept") for k in unchanged])
    m_idx = np.array([r[0] for r in rows], np.int64)
    old_part = _subset(sch, m_idx)
    comb = Schedule(np.concatenate([old_part.station, st.st[moved_k].astype(np.int64)]),
                    np.concatenate([old_part.target, st.tg[moved_k].astype(np.int64)]),
                    np.concatenate([old_part.start_s, st.a[moved_k].astype(np.int64) * cell]),
                    np.concatenate([old_part.dur_s, st.m[moved_k].astype(np.int64) * cell]),
                    algo=sch.algo + " · 重排")
    kinds = np.array([r[1] for r in rows] + ["grid"] * len(moved_k), object)
    pinned = kinds != "grid"
    n_pre = len(fi) + len(fixed)
    grid_pos = {n_pre + j: (int(st.a[k]) * cell, int(st.m[k]) * cell) for j, k in enumerate(unchanged)}
    _unpin_conflicts(comb, pinned, grid_pos, cfg)
    free = np.nonzero(~pinned)[0]
    refined = {"refined": False, "reason": "未开启"}
    if refine_seconds and len(free):
        L = comb.start_s.astype(np.int64).copy()
        R = (comb.start_s + comb.dur_s).astype(np.int64)
        fl, fr = true_bounds(pr, _subset(comb, free))
        fl = np.maximum(fl, t)
        if ev.kind == "down":
            da, db = ev.down
            on = comb.station[free] == ev.station
            after = comb.start_s[free] >= db
            fl = np.where(on & after, np.maximum(fl, db), fl)
            fr = np.where(on & ~after, np.minimum(fr, da), fr)
        L[free], R[free] = fl, fr
        out, refined = refine(pr, comb, bounds=(L, R), fixed=pinned)
        if refined.get("refined"):
            comb = Schedule(out.station, out.target, out.start_s, out.dur_s, algo=comb.algo)
    tick("按秒精修")

    # ---- 6. 对比与校验
    say(0.95, "约束校验与对比……")
    new_sel = kinds != "frozen"
    old_sel = ~frozen
    o_st, n_st = compare_plans(sch, old_sel, comb, new_sel)
    kept_exact = int(sum(1 for j in np.nonzero(n_st == "保留")[0] if pinned[j] and kinds[j] in ("kept", "fixed")))
    viol = structural_violations(pe, comb, by_second=True, extra_windows=extra_windows)
    fz = np.nonzero(kinds == "frozen")[0]
    frozen_same = bool(np.array_equal(comb.start_s[fz], sch.start_s[m_idx[:len(fi)]])
                       and np.array_equal(comb.dur_s[fz], sch.dur_s[m_idx[:len(fi)]])
                       and np.array_equal(comb.station[fz], sch.station[m_idx[:len(fi)]])
                       and np.array_equal(comb.target[fz], sch.target[m_idx[:len(fi)]]))
    outage_tasks = 0
    if ev.kind == "down":
        da, db = ev.down
        ce = comb.start_s + comb.dur_s
        outage_tasks = int(((comb.station == ev.station) & (comb.start_s < db) & (ce > da)).sum())
    keep_old = np.nonzero(~(interrupted | cancelled))[0]
    before = evaluate(cfg, pe.ds, sch)
    no_replan = evaluate(cfg, pe.ds, _subset(sch, keep_old))
    after = evaluate(cfg, pe.ds, comb)
    diff = {k: int((n_st == k).sum()) for k in ("保留", "移动", "新增")}
    diff["删除"] = int((o_st == "删除").sum())
    diff["冻结"] = int(len(fi))
    diff["原计划 t 之后"] = int(old_sel.sum())
    diff["完全不变"] = kept_exact
    diff["停机中断"] = int(interrupted.sum())
    diff["停机取消"] = int(cancelled.sum())
    diff["取消后改到别处"] = int(((o_st == "移动") & (interrupted | cancelled)).sum())
    tick("对比与校验")
    res = {
        "event": ev, "mode": mode, "describe": ev.describe(cfg, pb.ds), "schedule": comb, "old_status": o_st,
        "new_status": n_st, "pinned": pinned, "ds": pe.ds, "before": before, "no_replan": no_replan, "after": after,
        "diff": diff, "violations": viol, "frozen_unchanged": frozen_same, "outage_tasks": outage_tasks,
        "refine": refined, "seconds": round(time.time() - t_all, 1), "timing": timing,
        "residual": {"windows": int(len(pr.w_k0)), "targets": int((np.diff(pr.t_ptr) > 0).sum()),
                     "kept_on_grid": len(grid), "kept_fixed": len(fixed), "ub_relaxed": float(info["ub_relaxed"])},
    }
    if ev.kind in ("priority", "must"):
        x = ev.target
        mine_old = np.nonzero(sch.target == x)[0]
        mine_new = np.nonzero(comb.target == x)[0]
        res["target"] = {
            "before": [(int(sch.station[i]), int(start[i]), int(sch.dur_s[i])) for i in mine_old[np.argsort(start[mine_old])]],
            "after": [(int(comb.station[i]), int(comb.start_s[i]), int(comb.dur_s[i]))
                      for i in mine_new[np.argsort(comb.start_s[mine_new])]],
            "windows_after_t": int(pr.t_ptr[x + 1] - pr.t_ptr[x]),
            "frozen_obs": int(n_frozen[x]),
        }
        if ev.kind == "must":
            res["target"]["ok"] = bool(np.any(comb.target[comb.start_s >= t] == x))
            res["target"]["forced"] = must_ok
            res["target"]["released_fixed"] = len(gone)
    say(1.0, f"完成，用时 {res['seconds']:.0f} 秒")
    return res
