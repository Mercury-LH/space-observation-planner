"""目标状态分析：每颗目标的观测完成情况、观测时长与覆盖情况。

答疑 #26：赛题名称中的"目标状态分析"按观测任务状态理解（目标观测完成情况、观测时长及覆盖情况），
不涉及轨道状态估计。这里对一份计划逐目标给出：观测了几次（只计满足 48 小时间隔的观测，与计分口径相同）、
理论上最多几次、总观测时长、得分、首次与最近一次观测时间、所用站点、可见窗口情况，
以及没有覆盖或没有观测满的原因。
"""
from __future__ import annotations

import csv
from datetime import timedelta
from pathlib import Path

import numpy as np

from .data import Dataset
from .problem import Problem
from .schedule import Schedule, counted_mask, task_score

STATUS = {"SGP4_FAILED": "外推异常", "NO_WINDOW": "无可见窗口", "UNCOVERED": "未覆盖",
          "OBSERVED_1": "已观测 1 次", "OBSERVED_2": "已观测 2 次", "OBSERVED_3": "已观测 3 次"}
COLUMNS = ["target_id", "target_name", "priority", "orbit_class", "status", "status_cn", "n_obs", "max_obs",
           "observed_seconds", "task_score", "first_obs_utc", "last_obs_utc", "stations", "n_windows",
           "window_seconds", "n_stations_visible", "note"]


def orbit_class(ds: Dataset) -> np.ndarray:
    """按 TLE 平均运动划分：低轨（> 11.25 圈/天）、地球同步（约 1 圈/天）、其余为中高轨。"""
    mm = np.array([float(l[52:63]) for l in ds.line2])
    return np.where(mm > 11.25, "低轨", np.where((mm > 0.9) & (mm < 1.1), "地球同步", "中高轨"))


def _free_segments(pb: Problem, sch: Schedule) -> list[list[tuple[float, float]]]:
    """每个站点的空闲时段（秒），已扣除与相邻任务之间的 30 秒切换。"""
    cfg = pb.cfg
    horizon = pb.n_samples * cfg.cell_s
    out = []
    for s in range(pb.n_stations):
        idx = np.nonzero(sch.station == s)[0]
        idx = idx[np.argsort(sch.start_s[idx])]
        segs, t = [], 0.0
        for i in idx:
            a = float(sch.start_s[i]) - cfg.switch_s
            if a > t:
                segs.append((t, a))
            t = max(t, float(sch.start_s[i] + sch.dur_s[i]) + cfg.switch_s)
        if t < horizon:
            segs.append((t, float(horizon)))
        out.append(segs)
    return out


def target_status(pb: Problem, sch: Schedule, valid_mask: np.ndarray | None = None) -> list[dict]:
    cfg, ds = pb.cfg, pb.ds
    cell = cfg.cell_s
    counted = counted_mask(cfg, sch, valid_mask)
    sc = task_score(cfg, ds.priority[sch.target], sch.dur_s) * counted
    orbit = orbit_class(ds)
    free = None
    obs: dict[int, list[int]] = {}
    for i in np.nonzero(counted)[0]:
        obs.setdefault(int(sch.target[i]), []).append(int(i))
    rows = []
    for t in range(pb.n_targets):
        ws = pb.t_win[pb.t_ptr[t]:pb.t_ptr[t + 1]]
        mine = sorted(obs.get(t, []), key=lambda i: sch.start_s[i])
        n = len(mine)
        win_sec = int(((pb.w_k1[ws] - pb.w_k0[ws]) * cell).sum()) if len(ws) else 0
        if pb.win.sgp4_failed[t]:
            code, note = "SGP4_FAILED", "规划期内 SGP4 外推报错或位置不合理，无法计算可见窗口"
        elif not len(ws):
            code, note = "NO_WINDOW", "规划期内没有同时满足仰角、站点黑夜、目标受照且不短于 120 秒的窗口"
        elif n == 0:
            code = "UNCOVERED"
            if free is None:
                free = _free_segments(pb, sch)
            gap = any(min(b, (pb.w_k1[w]) * cell) - max(a, pb.w_k0[w] * cell) >= cfg.obs_min_s
                      for w in ws for a, b in free[int(pb.w_st[w])])
            note = (f"{len(ws)} 个可见窗口中仍有不短于 120 秒的空隙，但插入会与该目标或其他任务的约束冲突" if gap else
                    f"{len(ws)} 个可见窗口（共 {win_sec} 秒）的站点时间都已被其他任务占用")
        else:
            code = f"OBSERVED_{n}"
            k = int(pb.max_obs[t])
            note = ("已达到规划期内理论最多次数" if n >= k else
                    f"理论上最多 {k} 次，其余可见时段的站点时间已被其他任务占用")
        t0 = cfg.start_dt
        rows.append({
            "target_id": int(ds.satnum[t]), "target_name": ds.names[t], "priority": ds.priority_label[t],
            "orbit_class": str(orbit[t]), "status": code, "status_cn": STATUS[code], "n_obs": n,
            "max_obs": int(pb.max_obs[t]), "observed_seconds": int(sch.dur_s[mine].sum()) if n else 0,
            "task_score": round(float(sc[mine].sum()), 2) if n else 0.0,
            "first_obs_utc": (t0 + timedelta(seconds=int(sch.start_s[mine[0]]))).strftime("%Y-%m-%dT%H:%M:%SZ") if n else "",
            "last_obs_utc": (t0 + timedelta(seconds=int(sch.start_s[mine[-1]]))).strftime("%Y-%m-%dT%H:%M:%SZ") if n else "",
            "stations": ";".join(str(int(sch.station[i]) + 1) for i in mine),
            "n_windows": int(len(ws)), "window_seconds": win_sec,
            "n_stations_visible": int(len(np.unique(pb.w_st[ws]))) if len(ws) else 0, "note": note,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    """按优先级、轨道类别统计各状态的目标数，以及覆盖率、平均观测时长等。"""
    def count(sel):
        out = {v: 0 for v in STATUS.values()}
        for r in sel:
            out[r["status_cn"]] += 1
        return out
    pri = sorted({r["priority"] for r in rows}, key=lambda p: "高中低".find(p) if p in "高中低" else 9)
    obs = [r for r in rows if r["status"] not in ("SGP4_FAILED", "NO_WINDOW")]
    cov = [r for r in obs if r["n_obs"] > 0]
    by_pri = {p: count([r for r in rows if r["priority"] == p]) for p in pri}
    return {
        "targets": len(rows), "observable": len(obs), "covered": len(cov),
        "covered_rate_of_observable": round(len(cov) / max(len(obs), 1), 6),
        "observations": int(sum(r["n_obs"] for r in rows)),
        "targets_observed_2plus": sum(1 for r in rows if r["n_obs"] >= 2),
        "mean_observed_seconds_per_covered": round(float(np.mean([r["observed_seconds"] for r in cov])), 1) if cov else 0.0,
        "by_priority": by_pri,
        "by_orbit": {o: count([r for r in rows if r["orbit_class"] == o]) for o in sorted({r["orbit_class"] for r in rows})},
        "uncovered_observable": [{"target_id": r["target_id"], "priority": r["priority"], "note": r["note"]}
                                 for r in obs if r["n_obs"] == 0][:50],
    }


def export(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def describe(summary: dict) -> str:
    """一段话的状态摘要（命令行与界面用）。"""
    hi = summary["by_priority"].get("高", {})
    hi_total = sum(hi.values())
    hi_cov = sum(v for k, v in hi.items() if k.startswith("已观测"))
    return (f"可观测 {summary['observable']} 颗，已覆盖 {summary['covered']} 颗；"
            f"高优先级 {hi_cov}/{hi_total} 颗已覆盖；观测 2 次以上的目标 {summary['targets_observed_2plus']} 颗；"
            f"已覆盖目标平均观测 {summary['mean_observed_seconds_per_covered']:.0f} 秒")
