"""按秒精修：网格计划之后，把每个任务的起止按秒外推到真实可见边界，再用线性规划重新分配时长。

30 秒网格上，一个窗口的首、末采样点之外往往还有 0–29 秒是真正可见的，网格计划用不上这些时间。
精修分两步：
1. 对每个任务所在的窗口，在两端按 1 秒逐点检查（仰角、太阳角、目标受照、月光、外推合理性，余量与窗口计算相同），
   得到真实可见区间 [L, R]；
2. 任务集合和先后顺序都不变，解线性规划：最大化 Σ 优先级 × 观测时长，约束为任务落在 [L, R] 内、
   时长 120–240 秒、同站相邻任务之间留 30 秒切换、同一目标相邻两次观测间隔 48 小时（上次结束到下次开始）。
   约束都是"两个变量之差 ≤ 整数"的形式，最优解自然落在整秒上。
"""
from __future__ import annotations

import time

import numpy as np
from sgp4.api import Satrec

from .config import Config
from .data import Dataset
from .problem import Problem
from .schedule import Schedule
from .visibility import (WGS84_A, elevation_deg, gmst82, julian, moon_ecef, station_ecef, sun_elevation,
                         sun_unit_teme, teme_to_ecef)


def _task_windows(pb: Problem, sch: Schedule) -> np.ndarray:
    """每个任务所在的网格窗口下标。"""
    cell = pb.cfg.cell_s
    a = (sch.start_s // cell).astype(np.int64)
    b = ((sch.start_s + sch.dur_s) // cell).astype(np.int64)
    out = np.empty(len(sch), np.int64)
    for i, (s, t) in enumerate(zip(sch.station, sch.target)):
        ws = pb.t_win[pb.t_ptr[t]:pb.t_ptr[t + 1]]
        hit = ws[(pb.w_st[ws] == s) & (pb.w_k0[ws] <= a[i]) & (pb.w_k1[ws] >= b[i])]
        out[i] = hit[0]
    return out


def point_ok(cfg: Config, ds: Dataset, target: np.ndarray, station: np.ndarray, sec: np.ndarray) -> np.ndarray:
    """逐点判断 (目标, 站点, 秒) 是否满足全部观测条件（含安全余量）。"""
    ok = np.ones(len(sec), bool)
    st_pos, st_up = station_ecef(ds.st_lon, ds.st_lat, ds.st_alt_m)
    lo, hi = cfg.elev_min + cfg.elev_margin, cfg.elev_max - cfg.elev_margin
    if cfg.use_sun:
        sun_el = sun_elevation(cfg, ds, sec)                       # (n, S)
        ok &= sun_el[np.arange(len(sec)), station] <= cfg.sun_max - cfg.sun_margin
    jd, fr = julian(cfg, sec)
    theta = gmst82(jd, fr)
    sun = sun_unit_teme(jd, fr)
    moon = moon_ecef(cfg, sec) if cfg.moon_sep_deg > 0 else None
    for t in np.unique(target):
        idx = np.nonzero(target == t)[0]
        sat = Satrec.twoline2rv(ds.line1[t], ds.line2[t])
        e, r, _ = sat.sgp4_array(jd[idx], fr[idx])
        a_km = (398600.4418 / (sat.no_kozai / 60.0) ** 2) ** (1 / 3)
        rn = np.linalg.norm(r, axis=1)
        good = (e == 0) & (rn >= 0.9 * a_km * (1 - sat.ecco)) & (rn <= 1.1 * a_km * (1 + sat.ecco))
        if cfg.use_sunlit:
            proj = np.einsum("nk,nk->n", r, sun[idx])
            perp2 = np.einsum("nk,nk->n", r, r) - proj ** 2
            good &= (proj > 0) | (perp2 > (WGS84_A + cfg.sunlit_margin_km) ** 2)
        recef = teme_to_ecef(r, theta[idx])
        for s in np.unique(station[idx]):
            k = idx[station[idx] == s]
            sel = station[idx] == s
            el = elevation_deg(recef[sel], st_pos[s], st_up[s])
            g = good[sel] & (el >= lo) & (el <= hi)
            if moon is not None:
                mv = moon[k] - st_pos[s]
                mu_ = mv / np.linalg.norm(mv, axis=1, keepdims=True)
                d = recef[sel] - st_pos[s]
                cs = np.einsum("nk,nk->n", d, mu_) / np.linalg.norm(d, axis=1)
                moon_up = mu_ @ st_up[s] > -np.sin(np.radians(cfg.moon_margin_deg))   # 与窗口计算相同的余量
                g &= ~moon_up | (cs <= np.cos(np.radians(cfg.moon_sep_deg + cfg.moon_margin_deg)))
            ok[k] &= g
    return ok


def probe_bounds(cfg: Config, ds: Dataset, st: np.ndarray, tg: np.ndarray, k0: np.ndarray, k1: np.ndarray):
    """网格窗口 [k0, k1]（采样点）两端各按 1 秒向外探查最多 cell_s−1 秒，返回真实可见区间 [L, R]（秒）。"""
    cell = cfg.cell_s
    d = np.arange(1, cell)
    horizon = cfg.n_cells * cell
    secs = np.concatenate([k0[:, None] * cell - d[None, :], k1[:, None] * cell + d[None, :]], axis=1).astype(np.float64)
    flat = secs.ravel()
    t_rep, s_rep = np.repeat(tg, secs.shape[1]), np.repeat(st, secs.shape[1])
    inside = (flat >= 0) & (flat <= horizon)
    ok = np.zeros(len(flat), bool)
    if inside.any():
        ok[inside] = point_ok(cfg, ds, t_rep[inside], s_rep[inside], flat[inside])
    ok = ok.reshape(secs.shape)
    nl, pad = cell - 1, np.zeros((len(k0), 1), bool)
    ext_l = np.argmin(np.concatenate([ok[:, :nl], pad], axis=1), axis=1)    # 从边界向外连续可见的秒数
    ext_r = np.argmin(np.concatenate([ok[:, nl:], pad], axis=1), axis=1)
    return (k0 * cell - ext_l).astype(np.int64), (k1 * cell + ext_r).astype(np.int64)


def true_bounds(pb: Problem, sch: Schedule) -> tuple[np.ndarray, np.ndarray]:
    """每个任务所在网格窗口的真实可见区间 [L, R]（秒）。"""
    w = _task_windows(pb, sch)
    uw, inv = np.unique(w, return_inverse=True)
    L, R = probe_bounds(pb.cfg, pb.ds, pb.w_st[uw], pb.w_tg[uw], pb.w_k0[uw], pb.w_k1[uw])
    return L[inv], R[inv]


def refine(pb: Problem, sch: Schedule, verbose: bool = False, bounds=None) -> tuple[Schedule, dict]:
    """返回按秒精修后的计划和统计信息。任务集合与顺序不变，只调整起止时间。
    bounds：每个任务的真实可见区间 (L, R)；不给时从网格窗口探查。"""
    from ortools.linear_solver import pywraplp
    cfg, ds = pb.cfg, pb.ds
    t0 = time.time()
    n = len(sch)
    L, R = true_bounds(pb, sch) if bounds is None else bounds
    t1 = time.time()
    s0, e0 = sch.start_s.astype(np.int64), (sch.start_s + sch.dur_s).astype(np.int64)
    sw, dmin, dmax = cfg.switch_s, cfg.obs_min_s, cfg.obs_max_s
    gap = int(round(cfg.repeat_gap_h * 3600))
    P = ds.priority[sch.target].astype(float)
    lp = pywraplp.Solver.CreateSolver("GLOP")
    sv = [lp.NumVar(float(L[i]), float(R[i] - dmin), "") for i in range(n)]
    ev = [lp.NumVar(float(L[i] + dmin), float(R[i]), "") for i in range(n)]
    pairs = []                                # (前, 后, 最小间隔)
    for i in range(n):
        lp.Add(ev[i] - sv[i] >= dmin)
        lp.Add(ev[i] - sv[i] <= dmax)
    order = np.lexsort((s0, sch.station))
    for p, q in zip(order[:-1], order[1:]):
        if sch.station[p] == sch.station[q]:
            pairs.append((p, q, sw))          # 同站：上一任务结束 + 切换 ≤ 下一任务开始
    order = np.lexsort((s0, sch.target))
    for p, q in zip(order[:-1], order[1:]):
        if sch.target[p] == sch.target[q]:
            pairs.append((p, q, gap))         # 同目标：上次结束 + 48 小时 ≤ 下次开始
    for p, q, g in pairs:
        lp.Add(sv[q] - ev[p] >= g)
    obj = lp.Objective()
    for i in range(n):
        obj.SetCoefficient(ev[i], P[i])
        obj.SetCoefficient(sv[i], -P[i])
    obj.SetMaximization()
    status = lp.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        return sch, {"refined": False, "reason": f"线性规划状态 {status}"}
    s1 = np.array([round(v.solution_value()) for v in sv], np.int64)
    e1 = np.array([round(v.solution_value()) for v in ev], np.int64)
    # 取整后复核全部约束；万一不满足（理论上不会），回退原计划
    d = e1 - s1
    bad = int(((d < dmin) | (d > dmax) | (s1 < L) | (e1 > R)).sum())
    bad += sum(int(s1[q] - e1[p] < g) for p, q, g in pairs)
    if bad:
        return sch, {"refined": False, "reason": f"取整后 {bad} 处约束不满足"}
    out = Schedule(sch.station.copy(), sch.target.copy(), s1, d, algo=sch.algo + " + 按秒精修",
                   runtime_s=sch.runtime_s + time.time() - t0, info=dict(sch.info))
    info = {"refined": True, "seconds_bounds": round(t1 - t0, 1), "seconds_lp": round(time.time() - t1, 1),
            "dur_before": float((e0 - s0).mean()), "dur_after": float(d.mean()),
            "tasks_longer": int((d > e0 - s0).sum())}
    if verbose:
        print(f"  [精修] 探查真实边界 {info['seconds_bounds']}s、线性规划 {info['seconds_lp']}s；"
              f"平均时长 {info['dur_before']:.1f} → {info['dur_after']:.1f} 秒")
    return out, info



def near_windows(pb: Problem) -> dict:
    """网格上没有 ≥120 秒窗口、但按秒探查后有的目标窗口（网格长度 m_min−1 格，两端各外推最多 cell−1 秒）。结果缓存。"""
    from dataclasses import replace
    from pathlib import Path
    from .visibility import compute_windows
    cfg, ds = pb.cfg, pb.ds
    path = Path(cfg.cache_dir) / f"near_{cfg.scenario_key()}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    raw = Path(cfg.cache_dir) / f"short_{cfg.scenario_key()}.npz"     # get_windows 顺带算好的短窗口
    if raw.exists():
        z = np.load(raw)
        w_st, w_tg, w_k0, w_k1 = z["station"], z["target"], z["k0"], z["k1"]
    else:
        nw = compute_windows(replace(cfg, obs_min_s=(pb.m_min - 1) * cfg.cell_s), ds, verbose=False)
        w_st, w_tg, w_k0, w_k1 = nw.station, nw.target, nw.k0, nw.k1
    sel = np.nonzero((pb.max_obs[w_tg] == 0) & (w_k1 - w_k0 == pb.m_min - 1))[0]
    st, tg = w_st[sel].astype(np.int64), w_tg[sel].astype(np.int64)
    L, R = probe_bounds(cfg, ds, st, tg, w_k0[sel].astype(np.int64), w_k1[sel].astype(np.int64))
    ok = R - L >= cfg.obs_min_s
    out = {"station": st[ok], "target": tg[ok], "L": L[ok], "R": R[ok]}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def _passes(idx, lo, hi, dmin, sw):
    """同站任务链（按时间排序的下标 idx）：全部压到最短时长时，各任务最早结束 EF 与最晚开始 LS。不可行时返回 None。"""
    n = len(idx)
    EF = np.empty(n)
    LS = np.empty(n)
    prev = -1e18
    for j, i in enumerate(idx):
        s = max(lo[i], prev + sw)
        EF[j] = s + dmin
        if EF[j] > hi[i]:
            return None
        prev = EF[j]
    nxt = 1e18
    for j in range(n - 1, -1, -1):
        i = idx[j]
        e = min(hi[i], nxt - sw)
        LS[j] = e - dmin
        if LS[j] < lo[i]:
            return None
        nxt = LS[j]
    return EF, LS


def insert_near(pb: Problem, sch: Schedule, L: np.ndarray, R: np.ndarray, verbose: bool = False):
    """把"按秒才可见"的目标插进站点时间线：同站任务可压到 120 秒、在各自真实窗口内移动；
    腾不出位置时，可删掉一个价值更低的重复观测。返回新计划、对应的真实可见区间和统计。"""
    import bisect
    cfg, ds = pb.cfg, pb.ds
    sw, dmin = cfg.switch_s, cfg.obs_min_s
    gap = int(round(cfg.repeat_gap_h * 3600))
    near = near_windows(pb)
    st, tg = list(sch.station.astype(int)), list(sch.target.astype(int))
    s0 = list(sch.start_s.astype(float))
    lo, hi = list(L.astype(float)), list(R.astype(float))
    alive = [True] * len(st)
    # 同一目标的相邻观测：按当前位置保持 48 小时（保守）
    order = np.lexsort((sch.start_s, sch.target))
    e0 = sch.start_s + sch.dur_s
    for p, q in zip(order[:-1], order[1:]):
        if sch.target[p] == sch.target[q]:
            hi[p] = min(hi[p], float(sch.start_s[q] - gap))
            lo[q] = max(lo[q], float(e0[p] + gap))
    cnt = np.bincount(sch.target, minlength=pb.n_targets)
    by_st = {s: sorted((i for i in range(len(st)) if st[i] == s), key=lambda i: s0[i]) for s in range(pb.n_stations)}
    P = ds.priority.astype(float)
    val_new = pb.vc + pb.vs * P[near["target"]] * pb.frac[pb.m_min]
    added, deleted, tried = [], 0, 0
    for c in np.argsort(-val_new, kind="stable"):
        t, s = int(near["target"][c]), int(near["station"][c])
        if cnt[t] > 0:
            continue
        Lc, Rc = float(near["L"][c]), float(near["R"][c])
        tried += 1
        idx = by_st[s]
        starts = [s0[i] for i in idx]
        k0 = bisect.bisect_left(starts, Lc)

        def try_insert(idx_):
            ps = _passes(idx_, lo, hi, dmin, sw)
            if ps is None:
                return None
            EF, LS = ps
            st_ = [s0[i] for i in idx_]
            k = bisect.bisect_left(st_, Lc)
            for kk in range(max(0, k - 1), min(len(idx_), k + 1) + 1):
                x_lo = max(Lc, (EF[kk - 1] + sw) if kk > 0 else 0.0)
                x_hi = min(Rc, (LS[kk] - sw) if kk < len(idx_) else 1e18) - dmin
                if x_lo <= x_hi:
                    return kk, x_lo
            return None
        res = try_insert(idx)
        drop = None
        if res is None:
            # 删掉一个或两个相邻的、价值更低的重复观测再试
            def rep_val(b):
                return pb.vs * P[tg[b]] * pb.frac[min(pb.m_max, int(round((e0[b] if b < len(e0) else s0[b] + dmin) - s0[b])) // cfg.cell_s)] \
                    if b < len(sch) else 1e9
            near_j = [j for j in range(max(0, k0 - 3), min(len(idx), k0 + 3)) if cnt[tg[idx[j]]] >= 2]
            best = None
            combos = [(j,) for j in near_j] + [(j, j2) for j in near_j for j2 in near_j if j2 > j and tg[idx[j]] != tg[idx[j2]]]
            for cb in combos:
                vb = sum(rep_val(idx[j]) for j in cb)
                if vb >= val_new[c] or (best is not None and vb >= best[0]):
                    continue
                r2 = try_insert([i for jj, i in enumerate(idx) if jj not in cb])
                if r2 is not None:
                    best = (vb, cb, r2)
            if best is not None:
                _, cb, res = best
                drop = [idx[j] for j in cb]
        if res is None:
            continue
        if drop is not None:
            for d_ in drop:
                alive[d_] = False
                cnt[tg[d_]] -= 1
                deleted += 1
            idx = [i for i in idx if i not in drop]
        kk, x = res
        st.append(s); tg.append(t); s0.append(x); lo.append(Lc); hi.append(Rc); alive.append(True)
        L = np.append(L, int(Lc)); R = np.append(R, int(Rc))
        new = len(st) - 1
        idx.insert(kk, new)
        by_st[s] = idx
        cnt[t] += 1
        added.append(new)
    keep = np.nonzero(alive)[0]
    out = Schedule(np.array(st, np.int64)[keep], np.array(tg, np.int64)[keep],
                   np.round(np.array(s0)[keep]).astype(np.int64), np.full(len(keep), dmin, np.int64),
                   algo=sch.algo, runtime_s=sch.runtime_s, info=dict(sch.info))
    # 原有任务保留原来的时长（线性规划会重新分配），新插入的先给 120 秒
    dur = np.concatenate([sch.dur_s.astype(np.int64), np.full(len(st) - len(sch), dmin, np.int64)])[keep]
    out.dur_s = dur
    info = {"near_candidates": int(len(near["target"])), "tried": tried, "added": len(added), "deleted_repeats": deleted}
    if verbose:
        print(f"  [按秒补覆盖] 候选 {len(near['target'])} 个窗口，新增覆盖 {len(added)} 颗，删去重复观测 {deleted} 次")
    return out, (L[keep], R[keep]), info, np.array([k for k, i in enumerate(keep) if i >= len(sch)], np.int64)


def finish(pb: Problem, sch: Schedule, verbose: bool = False, near: bool = True) -> tuple[Schedule, dict]:
    """网格计划的按秒收尾：插入"按秒才可见"的目标，再用线性规划统一定时；不可行时逐个撤回插入。
    near=False 时只做精修（演示用的子问题必须关掉：被抽掉的目标同样"没有窗口"，会被误当成候选）。"""
    from .schedule import evaluate
    L, R = true_bounds(pb, sch)
    base, bi = refine(pb, sch, bounds=(L, R))
    if not near or not bi.get("refined"):
        return base, {"refine": bi, "insert": {"added": 0, "skipped": True}}
    cand, (L2, R2), ii, new_idx = insert_near(pb, sch, L, R, verbose=verbose)
    out, ri = refine(pb, cand, bounds=(L2, R2))
    while not ri.get("refined") and len(new_idx):
        keep = np.ones(len(cand), bool)
        keep[new_idx[-1]] = False
        new_idx = np.array([k - (k > new_idx[-1]) for k in new_idx[:-1]], np.int64)
        cand = Schedule(cand.station[keep], cand.target[keep], cand.start_s[keep], cand.dur_s[keep], algo=cand.algo,
                        runtime_s=cand.runtime_s, info=cand.info)
        L2, R2 = L2[keep], R2[keep]
        ii["added"] -= 1
        out, ri = refine(pb, cand, bounds=(L2, R2))
    if not ri.get("refined"):
        return base, {"refine": bi, "insert": {**ii, "added": 0, "fallback": True}}
    if evaluate(pb.cfg, pb.ds, out)["combined"] < evaluate(pb.cfg, pb.ds, base)["combined"]:
        return base, {"refine": bi, "insert": {**ii, "added": 0, "worse": True}}
    return out, {"refine": ri, "insert": ii}
