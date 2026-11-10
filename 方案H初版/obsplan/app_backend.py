"""产品界面的后端：运行规划、保存/读取结果、生成甘特图数据与任务解释。界面代码只调用这里。"""
from __future__ import annotations

import io
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np

from . import ga, lagrange
from .config import Config
from .data import Dataset
from .greedy import MANUAL_WEIGHTS, algo_priority, algo_weighted
from .local_search import improve
from .problem import Problem
from .schedule import Schedule, evaluate, export_csv, score_ref
from .target_status import orbit_class  # noqa: F401（界面沿用 app_backend.orbit_class）
from .visibility import Windows

ALGOS = {
    "H": "H 价格协调（推荐）",
    "E": "E AI 进化权重贪心 + 局部搜索",
    "B": "B 固定优先级贪心（基线）",
}


@dataclass
class Run:
    run_id: str
    algo: str
    cfg: dict
    subset: dict
    station: np.ndarray
    target: np.ndarray
    start_s: np.ndarray
    dur_s: np.ndarray
    metrics: dict
    runtime_s: float
    mu: np.ndarray | None = None
    validation: dict | None = None
    notes: dict = field(default_factory=dict)
    per_task: np.ndarray | None = None      # Skyfield 复核的 (最低仰角, 最高仰角, 最高太阳角)

    def schedule(self) -> Schedule:
        return Schedule(self.station, self.target, self.start_s, self.dur_s, algo=ALGOS.get(self.algo, self.algo),
                        runtime_s=self.runtime_s)

    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        meta = {k: v for k, v in asdict(self).items() if not isinstance(v, np.ndarray) and k not in ("mu", "per_task")}
        path = folder / f"{self.run_id}.npz"
        np.savez_compressed(path, station=self.station, target=self.target, start_s=self.start_s, dur_s=self.dur_s,
                            mu=self.mu if self.mu is not None else np.zeros((0, 0)),
                            per_task=self.per_task if self.per_task is not None else np.zeros((0, 3)),
                            meta=np.frombuffer(json.dumps(meta, ensure_ascii=False, default=float).encode(), np.uint8))
        return path

    @classmethod
    def load(cls, path: Path) -> "Run":
        z = np.load(path)
        meta = json.loads(bytes(z["meta"]).decode())
        mu = z["mu"] if z["mu"].size else None
        pt = z["per_task"] if "per_task" in z and z["per_task"].size else None
        return cls(station=z["station"], target=z["target"], start_s=z["start_s"], dur_s=z["dur_s"], mu=mu,
                   per_task=pt, **meta)


def run_name(run: Run) -> str:
    """结果的显示名称：有标签用标签，否则是"算法 模式 · 月-日 时:分"（H+ = 用伐谋放置规则的方案 H）。"""
    if run.notes.get("label"):
        return run.notes["label"]
    algo = run.algo
    if algo == "H":
        algo = "H 按天气" if run.notes.get("weather_aware") else ("H+" if run.notes.get("famou_rule") else "H")
    rid = run.run_id
    when = f"{rid[4:6]}-{rid[6:8]} {rid[9:11]}:{rid[11:13]}" if len(rid) >= 15 and rid[8] == "-" else rid
    return " ".join(x for x in (algo, run.notes.get("mode", "")) if x) + f" · {when}"


def scope_metrics(cfg: Config, ds: Dataset, run: Run) -> dict:
    """结果所在范围：参与规划的目标（mask，全量为 None）、目标数，以及按这个范围归一的综合指标。
    演示结果只排了一部分目标，按全部目标归一的综合值会低得没有意义，所以改用参与目标的个数和满分参考。"""
    m = run.metrics
    if not run.subset:
        return {"mask": None, "targets": ds.n_targets, "combined": float(m["combined"])}
    mask = subset_mask(ds.n_targets, run.subset["target_frac"], run.subset.get("seed", 0))
    n = int(mask.sum())
    ref = score_ref(cfg, ds) * float(ds.priority[mask].sum()) / max(float(ds.priority.sum()), 1e-9)
    return {"mask": mask, "targets": n,
            "combined": float(cfg.w_cov * m["covered"] / max(n, 1) + cfg.w_score * m["score"] / max(ref, 1e-9))}


def list_runs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_problem(cfg: Config) -> tuple[Dataset, Problem, dict]:
    """读数据、算（或读缓存的）可见窗口、建问题实例，并记下各步用时（界面"演示"页用）。"""
    from .data import load
    from .visibility import get_windows
    t0 = time.time()
    ds = load(cfg)
    t1 = time.time()
    cached = (Path(cfg.cache_dir) / f"windows_{cfg.scenario_key()}.npz").exists()
    win = get_windows(cfg, ds)
    t2 = time.time()
    pb = Problem.build(cfg, ds, win)
    t3 = time.time()
    return ds, pb, {"load_s": t1 - t0, "windows_s": t2 - t1, "windows_cached": cached, "build_s": t3 - t2}


def subset_mask(n_targets: int, target_frac: float, seed: int = 0) -> np.ndarray:
    """演示模式下参与规划的目标（固定种子，同一比例每次选到同一批）。"""
    if target_frac >= 1:
        return np.ones(n_targets, bool)
    return np.random.default_rng(seed).random(n_targets) < target_frac


def subset_problem(pb: Problem, target_frac: float, stations: list[int], seed: int = 0) -> Problem:
    """演示模式：只保留部分目标和指定站点。"""
    keep_t = subset_mask(pb.n_targets, target_frac, seed)
    keep_s = np.zeros(pb.n_stations, bool)
    keep_s[list(stations)] = True
    sel = keep_t[pb.w_tg] & keep_s[pb.w_st]
    w = pb.win
    w2 = Windows(pb.w_st[sel].astype(np.int16), pb.w_tg[sel].astype(np.int32), pb.w_k0[sel].astype(np.int32),
                 pb.w_k1[sel].astype(np.int32), w.dark, w.n_samples, w.sgp4_failed)
    return Problem.build(pb.cfg, pb.ds, w2)


CLOUD_THRESHOLD = 30.0


def weather_stats(pb: Problem, threshold: float = CLOUD_THRESHOLD):
    """过去 10 年同期逐小时云量（首次需联网，之后用缓存）与各站晴空比例。"""
    from .environment import fetch_cloud_history
    cloud = fetch_cloud_history(pb.cfg, pb.ds, verbose=False)
    return cloud, np.nanmean(cloud <= threshold, axis=(0, 2))


def replay_weather(pb: Problem, run: Run, threshold: float = CLOUD_THRESHOLD) -> dict:
    from .environment import evaluate_under_weather
    cloud, _ = weather_stats(pb, threshold)
    return evaluate_under_weather(pb.cfg, pb.ds, run.schedule(), cloud, threshold)


def simulate_days(pb: Problem, year: int, threshold: float = CLOUD_THRESHOLD, progress=None) -> dict:
    """把"一次排完、照单执行"和"每天按执行结果重排"放到某一年的真实天气里逐日模拟。
    气候统计用其余年份（不含被检验的这一年）。"""
    from .dynamic import DAY_H, daily, simulate
    cloud, _ = weather_stats(pb, threshold)
    years = list(range(2016, 2016 + cloud.shape[0]))
    yi = years.index(year)
    truth = cloud[yi]
    clim = np.nanmean(np.delete(cloud, yi, axis=0) <= threshold, axis=(0, 2))
    days = int(np.ceil(truth.shape[1] / DAY_H))
    out = {"year": year, "days": days}
    plan = (("static", "一次排完、照单执行"), ("dynamic_clim", "每天按执行结果重排"))
    for j, (policy, name) in enumerate(plan):
        def cb(d, n, j=j, name=name):
            if progress is not None:
                progress((j + d / n) / len(plan), f"{name}：第 {d}/{n} 天")
        if progress is not None:
            progress(j / len(plan), f"{name}：规划中……")
        res, status = simulate(pb, truth, policy, clim, threshold, iters=150, fast=True, return_status=True,
                               on_day=cb)
        out[policy] = {"name": name, **{k: float(v) for k, v in res.items()}, "daily": daily(pb, status, days)}
    return out


def run_planning(pb: Problem, algo: str, iters: int = 300, progress=None, weather: bool = False,
                 fast: bool = False, refine_seconds: bool = True, full_problem: bool = True,
                 evolved_rule: bool = True) -> Run:
    """refine_seconds：网格计划之后做按秒精修（窗口两端外推到真实边界，线性规划重新分配时长）。
    full_problem：pb 是否为完整问题；演示子集上不做"按秒补覆盖"。
    evolved_rule：H 的放置规则用伐谋演化并已采纳的版本（obsplan/evolved/），否则用人工的"价值 − 价格"。"""
    cfg = pb.cfg
    t0 = time.time()
    mu = None
    notes = {}
    if algo == "H":
        kw = {}
        if weather:
            if progress:
                progress(0.0, "读取历史云量……")
            _, sc = weather_stats(pb)
            kw = {"cell_weight": np.repeat(sc[:, None], pb.n_samples + 2, axis=1), "backup": np.zeros(pb.n_targets)}
            notes["weather_aware"] = True
        elif evolved_rule:
            from .evolved import load_rule
            kw["rule"] = load_rule()
            kw["portfolio"] = True          # 稳健模式：人工规则也排一份，取综合高的（两份局部搜索并发，用时基本不变）
            notes["famou_rule"] = kw["rule"] is not None
        sch, info = lagrange.solve(pb, iters=iters, verbose=False, progress=progress, fast=fast, **kw)
        if "portfolio" in info:
            pf = info["portfolio"]
            notes["famou_rule"] = pf["picked"] == "rule"
            notes["rule_pick"] = {"picked": pf["picked"], "objective": {k: float(v) for k, v in pf["objective"].items()}}
        notes["mode"] = "快速" if fast else "高质量"
        mu = info["mu"].astype(np.float32)
        notes["ub_relaxed"] = float(info["ub_relaxed"])
    elif algo == "E":
        if progress:
            progress(0.1, "遗传算法学习打分权重……")
        w, gi = ga.train(pb, verbose=False)
        if progress:
            progress(0.6, "贪心构造与局部搜索……")
        sch, st = algo_weighted(pb, w, "E", return_state=True)
        improve(pb, st, repeat_min_priority=3.0 if fast else None)
        sch = st.schedule(pb, "E", 0)
        notes = {"weights": [float(x) for x in w], "mode": "快速" if fast else "高质量"}
    else:
        sch = algo_priority(pb)
    if refine_seconds:
        from .refine import finish
        if progress:
            progress(0.97, "按秒精修：探查真实可见边界、插入按秒才可见的目标、重新分配观测时长……")
        sch, fi = finish(pb, sch, near=full_problem)
        notes["refine"], notes["insert"] = fi["refine"], fi["insert"]
    rt = time.time() - t0
    m = evaluate(cfg, pb.ds, sch)
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{algo}"
    return Run(run_id, algo, json.loads(cfg.to_json()), {}, sch.station, sch.target, sch.start_s, sch.dur_s,
               m, rt, mu=mu, notes=notes)


# ------------------------------------------------------------------ 展示数据
def utc(cfg: Config, seconds: float):
    return cfg.start_dt + timedelta(seconds=float(seconds))




def station_nights(pb: Problem) -> list[tuple[int, int, int, int]]:
    """(站点, 第几夜, 开始格, 结束格)。"""
    out = []
    for s, runs in enumerate(pb.win.dark):
        for n, (a, b) in enumerate(runs):
            if b - a >= 10:
                out.append((s, n, a, b))
    return out


def task_rows(pb: Problem, run: Run, orbit: np.ndarray) -> list[dict]:
    cfg, ds = pb.cfg, pb.ds
    order = np.lexsort((run.start_s, run.target))
    first = set()
    kind = np.empty(len(run.target), object)
    for i in order:
        t = int(run.target[i])
        kind[i] = "首次覆盖" if t not in first else "48 小时后重复"
        first.add(t)
    frac = 0.6 + 0.4 * np.clip(run.dur_s - cfg.obs_min_s, 0, cfg.obs_max_s - cfg.obs_min_s) / (cfg.obs_max_s - cfg.obs_min_s)
    rows = []
    for i in range(len(run.target)):
        t = int(run.target[i])
        rows.append({
            "站点": int(run.station[i]) + 1, "目标编号": int(ds.satnum[t]), "名称": ds.names[t],
            "优先级": ds.priority_label[t], "轨道": orbit[t], "开始": utc(cfg, run.start_s[i]),
            "结束": utc(cfg, run.start_s[i] + run.dur_s[i]), "时长(秒)": int(run.dur_s[i]),
            "得分": round(float(ds.priority[t] * frac[i]), 2), "类型": kind[i],
        })
    return rows


def explain_target(pb: Problem, run: Run, t: int) -> dict:
    """解释一颗目标为什么排在这里（或为什么没排上）。"""
    cfg, ds = pb.cfg, pb.ds
    cell = cfg.cell_s
    mine = np.nonzero(run.target == t)[0]
    ws = pb.t_win[pb.t_ptr[t]:pb.t_ptr[t + 1]]
    mu = run.mu
    pref = None
    if mu is not None:
        pref = np.concatenate([np.zeros((mu.shape[0], 1)), np.cumsum(mu, axis=1)], axis=1)
    # 站点占用：每格被哪个任务占用
    occ = {}
    for i in range(len(run.target)):
        s = int(run.station[i])
        a, m = int(run.start_s[i] // cell), int(run.dur_s[i] // cell)
        occ.setdefault(s, []).append((a, a + m, i))
    windows = []
    for w in ws:
        s, k0, k1 = int(pb.w_st[w]), int(pb.w_k0[w]), int(pb.w_k1[w])
        price = None
        if pref is not None and k1 - k0 >= pb.m_min:
            costs = pref[s, k0 + pb.m_min + 1:k1 + 2] - pref[s, k0:k1 - pb.m_min + 1]
            price = float(costs.min()) if len(costs) else None
        users = [i for (a, b, i) in occ.get(s, []) if a <= k1 and b >= k0]
        # 按秒精修后任务起点可能比网格窗口早最多 cell−1 秒
        used_here = [i for i in mine if int(run.station[i]) == s and (k0 - 1) * cell < run.start_s[i] <= k1 * cell]
        busy = sum(min(b, k1) - max(a, k0) + 1 for (a, b, i) in occ.get(s, []) if a <= k1 and b >= k0)
        windows.append({
            "站点": s + 1, "开始": utc(cfg, k0 * cell), "结束": utc(cfg, k1 * cell), "长度(秒)": (k1 - k0) * cell,
            "最低价格": price, "占用率": round(min(1.0, busy / max(k1 - k0 + 1, 1)), 2),
            "本目标在此观测": bool(used_here), "占用任务数": len(users) - len(used_here),
            "_users": [int(u) for u in users if u not in used_here],
        })
    pmax = max([w["最低价格"] or 0 for w in windows] + [1e-12])
    for w in windows:
        if w["最低价格"] is not None:
            w["相对价格"] = round(w["最低价格"] / pmax, 3)
    name = f"{ds.satnum[t]}（{ds.names[t] or '无名称'}，{ds.priority_label[t]}优先级）"
    lines = []
    if pb.win.sgp4_failed[t]:
        lines.append("该目标的轨道外推到规划期时 SGP4 报错或给出不合理的位置，无法可靠计算可见窗口，因此不安排观测。")
    elif not len(ws):
        lines.append("规划期内没有同时满足仰角 10°–85°、站点天黑和目标受照、且持续 ≥120 秒的窗口，无法观测。")
    elif len(mine):
        lines.append(f"共安排 {len(mine)} 次观测（可见窗口 {len(ws)} 个，分布在 {len({w['站点'] for w in windows})} 个站）。")
        prices = sorted(w["最低价格"] for w in windows if w["最低价格"] is not None)
        for k, i in enumerate(sorted(mine, key=lambda i: run.start_s[i])):
            s = int(run.station[i])
            a = int(run.start_s[i] // cell)
            txt = f"第 {k + 1} 次：{s + 1} 号站，{utc(cfg, run.start_s[i]):%m-%d %H:%M:%S} UTC 起，{run.dur_s[i]} 秒"
            home = next((w for w, wi in zip(windows, ws) if int(pb.w_st[wi]) == s and pb.w_k0[wi] <= a <= pb.w_k1[wi]), None)
            if home is not None and home["最低价格"] is not None and prices:
                rank = sum(1 for p in prices if p < home["最低价格"] - 1e-15) + 1
                txt += f"；所在窗口的价格在该目标 {len(prices)} 个窗口中排第 {rank} 低"
                if rank > 1:
                    txt += "（更便宜的窗口与该目标其他观测不满足 48 小时间隔，或空档已被占满）"
            lines.append(txt + "。")
        if len(mine) > 1:
            lines.append("各次观测之间满足\"上次结束 48 小时后\"的重复计分条件。")
        elif pb.max_obs[t] > 1:
            lines.append("该目标理论上还可以再观测，但对应时段的站点时间被价值更高的任务占用。")
    else:
        full = [w for w in windows if w["占用率"] >= 0.99]
        if len(full) == len(windows):
            lines.append(f"有 {len(ws)} 个可见窗口，但都已被其他任务占满，没有排上。")
        else:
            lines.append(f"有 {len(ws)} 个可见窗口，但没有排上：{len(full)} 个已被其他任务占满，"
                         f"其余 {len(windows) - len(full)} 个里的空隙都不足 150 秒（120 秒观测 + 30 秒切换）。")
        rivals = {}
        for w in windows:
            for u in w["_users"]:
                rivals[u] = rivals.get(u, 0) + 1
        if rivals:
            top = sorted(rivals, key=lambda u: -ds.priority[run.target[u]])[:5]
            desc = "、".join(f"{ds.satnum[run.target[u]]}（{ds.priority_label[run.target[u]]}）" for u in top)
            lines.append(f"占用这些时段的任务包括：{desc} 等。")
    for w in windows:
        w.pop("_users", None)
    return {"name": name, "lines": lines, "windows": windows}


def pass_profile(pb: Problem, run: Run, i: int, pad_s: int = 300) -> dict:
    """第 i 个任务前后各 pad_s 秒内按秒计算的几何量：目标仰角、站点太阳高度角、目标是否受照（与窗口计算同一模型）。"""
    from sgp4.api import Satrec
    from .visibility import (WGS84_A, elevation_deg, gmst82, julian, station_ecef, sun_elevation, sun_unit_teme,
                             teme_to_ecef)
    cfg, ds = pb.cfg, pb.ds
    s, t = int(run.station[i]), int(run.target[i])
    a0, e0 = float(run.start_s[i]), float(run.start_s[i] + run.dur_s[i])
    horizon = cfg.n_cells * cfg.cell_s
    sec = np.arange(max(0.0, a0 - pad_s), min(horizon, e0 + pad_s) + 1, 1.0)
    jd, fr = julian(cfg, sec)
    sat = Satrec.twoline2rv(ds.line1[t], ds.line2[t])
    err, r, _ = sat.sgp4_array(jd, fr)
    pos, up = station_ecef(ds.st_lon, ds.st_lat, ds.st_alt_m)
    el = elevation_deg(teme_to_ecef(r, gmst82(jd, fr)), pos[s], up[s])
    sun_el = sun_elevation(cfg, ds, sec)[:, s]
    sun = sun_unit_teme(jd, fr)
    proj = np.einsum("nk,nk->n", r, sun)
    perp2 = np.einsum("nk,nk->n", r, r) - proj ** 2
    lit = (proj > 0) | (perp2 > WGS84_A ** 2)
    el[err != 0] = np.nan
    obs = (sec >= a0) & (sec <= e0)
    return {"sec": sec, "el": el, "sun_el": sun_el, "lit": lit, "obs": obs, "start": a0, "end": e0,
            "station": s, "target": t,
            "summary": {"el_min": float(np.nanmin(el[obs])), "el_max": float(np.nanmax(el[obs])),
                        "sun_max": float(sun_el[obs].max()), "lit_all": bool(lit[obs].all()),
                        "dur": int(round(e0 - a0))}}


def validate_run(pb: Problem, run: Run) -> Run:
    from .validator import validate
    t0 = time.time()
    chk = validate(pb.cfg, pb.ds, run.schedule())
    run.validation = {"legal": chk["legal"], "violations": chk["violations"],
                      "margins": [chk["min_elev_margin"], chk["max_elev_margin"], chk["sun_margin"]],
                      "seconds": round(time.time() - t0, 1)}
    run.per_task = chk["per_task"].astype(np.float32)
    return run


def near_for(pb: Problem, run: Run) -> dict | None:
    """全量运行里"按秒补覆盖"插入的目标在网格上没有窗口，结构检查要对照它们的真实可见区间。"""
    if run.subset or not run.notes.get("insert", {}).get("added"):
        return None
    from .refine import near_windows
    return near_windows(pb)


def structural_check(pb: Problem, run: Run) -> dict[str, int]:
    """快速结构检查（obsplan/checks.py，按秒精修后的口径）：各类违规的任务数，空字典表示全部合法。"""
    from .checks import structural_violations
    bad = structural_violations(pb, run.schedule(), by_second=True, extra_windows=near_for(pb, run))
    return {k: len(v) for k, v in bad.items()}


# ------------------------------------------------------------------ 突发事件重规划（obsplan/replan.py）
def replan_run(pb: Problem, run: Run, event, mode: str = "minimal", fast: bool = True, progress=None) -> dict:
    """在当前结果上处理突发事件：冻结事件时刻前已执行的任务，用价格协调重排之后的时段。"""
    from .replan import replan
    return replan(pb, run.schedule(), event, mode=mode, iters=150 if fast else 300, fast=fast,
                  extra_windows=near_for(pb, run), progress=progress)


def replan_changed(res: dict) -> np.ndarray:
    """新计划里与原计划不完全相同的任务（移动、新增，以及跟着按秒微调过的保留任务）。"""
    st = res["new_status"]
    return np.nonzero(((st == "移动") | (st == "新增") | ((st == "保留") & ~res["pinned"])))[0]


def validate_replan(pb: Problem, res: dict, include_kept: bool = False) -> dict:
    """Skyfield 逐秒校验重排后的计划：默认只校验改动过的任务（其余任务与原计划完全相同），也可校验事件后的全部任务。"""
    from .validator import validate
    sch = res["schedule"]
    idx = np.nonzero(res["new_status"] != "冻结")[0] if include_kept else replan_changed(res)
    part = Schedule(sch.station[idx], sch.target[idx], sch.start_s[idx], sch.dur_s[idx])
    t0 = time.time()
    chk = validate(pb.cfg, res["ds"], part) if len(idx) else {"legal": True, "violations": {}}
    return {"legal": chk["legal"], "violations": chk["violations"], "tasks": int(len(idx)),
            "scope": "事件后的全部任务" if include_kept else "改动过的任务", "seconds": round(time.time() - t0, 1)}


def schedule_csv_bytes(cfg: Config, ds: Dataset, sch: Schedule, name: str) -> bytes:
    tmp = Path(cfg.cache_dir) / f"_export_{name}.csv"
    export_csv(cfg, ds, sch, tmp)
    data = tmp.read_bytes()
    tmp.unlink(missing_ok=True)
    return data


def save_replan(run: Run, res: dict, folder: Path) -> Run:
    """把重排后的计划另存为一份结果，其他页签（甘特图、目标状态、校验、导出）都能直接查看。"""
    sch = res["schedule"]
    refined = bool(res["refine"].get("refined") or run.notes.get("refine", {}).get("refined"))
    notes = {k: run.notes[k] for k in ("mode", "famou_rule", "rule_pick") if k in run.notes}
    notes.update({"label": f"重排：{res['describe']}", "refine": {"refined": refined},
                  "replan": {"from": run.run_id, "mode": res["mode"], "event": res["describe"], "diff": res["diff"]}})
    new = Run(time.strftime("%Y%m%d-%H%M%S") + f"-{run.algo}R", run.algo, dict(run.cfg), dict(run.subset),
              sch.station.astype(np.int64), sch.target.astype(np.int64), sch.start_s.astype(np.int64),
              sch.dur_s.astype(np.int64), res["after"], float(res["seconds"]), notes=notes)
    new.save(folder)
    return new


def csv_bytes(pb: Problem, run: Run) -> bytes:
    tmp = Path(pb.cfg.cache_dir) / f"_export_{run.run_id}.csv"
    check = {"per_task": run.per_task} if run.per_task is not None else None
    export_csv(pb.cfg, pb.ds, run.schedule(), tmp, check)
    data = tmp.read_bytes()
    tmp.unlink(missing_ok=True)
    return data


def status_rows(pb: Problem, run: Run) -> list[dict]:
    """目标状态分析（obsplan/target_status.py）：每颗目标的观测完成情况、时长与覆盖情况。"""
    from .target_status import target_status
    return target_status(pb, run.schedule())


def status_csv_bytes(rows: list[dict]) -> bytes:
    import csv
    from .target_status import COLUMNS
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS)
    w.writeheader()
    w.writerows(rows)
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


def summary_json(run: Run) -> bytes:
    d = {"run_id": run.run_id, "algo": ALGOS.get(run.algo, run.algo), "metrics": run.metrics,
         "runtime_s": run.runtime_s, "config": run.cfg, "subset": run.subset, "validation": run.validation,
         "notes": run.notes}
    return json.dumps(d, ensure_ascii=False, indent=2, default=float).encode("utf-8")
