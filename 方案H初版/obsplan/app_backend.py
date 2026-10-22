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
from .schedule import Schedule, evaluate, export_csv
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


def list_runs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)


def subset_problem(pb: Problem, target_frac: float, stations: list[int], seed: int = 0) -> Problem:
    """演示模式：只保留部分目标和指定站点。"""
    rng = np.random.default_rng(seed)
    keep_t = rng.random(pb.n_targets) < target_frac if target_frac < 1 else np.ones(pb.n_targets, bool)
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
            notes["famou_rule"] = kw["rule"] is not None
        sch, info = lagrange.solve(pb, iters=iters, verbose=False, progress=progress, fast=fast, **kw)
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


def orbit_class(ds: Dataset) -> np.ndarray:
    mm = np.array([float(l[52:63]) for l in ds.line2])
    return np.where(mm > 11.25, "低轨", np.where((mm > 0.9) & (mm < 1.1), "地球同步", "中高轨"))


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
    chk = validate(pb.cfg, pb.ds, run.schedule())
    run.validation = {"legal": chk["legal"], "violations": chk["violations"],
                      "margins": [chk["min_elev_margin"], chk["max_elev_margin"], chk["sun_margin"]]}
    run.per_task = chk["per_task"].astype(np.float32)
    return run


def csv_bytes(pb: Problem, run: Run) -> bytes:
    tmp = Path(pb.cfg.cache_dir) / f"_export_{run.run_id}.csv"
    check = {"per_task": run.per_task} if run.per_task is not None else None
    export_csv(pb.cfg, pb.ds, run.schedule(), tmp, check)
    data = tmp.read_bytes()
    tmp.unlink(missing_ok=True)
    return data


def summary_json(run: Run) -> bytes:
    d = {"run_id": run.run_id, "algo": ALGOS.get(run.algo, run.algo), "metrics": run.metrics,
         "runtime_s": run.runtime_s, "config": run.cfg, "subset": run.subset, "validation": run.validation,
         "notes": run.notes}
    return json.dumps(d, ensure_ascii=False, indent=2, default=float).encode("utf-8")
