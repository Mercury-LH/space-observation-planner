"""计划表示、评分、上界和导出。所有算法输出同一种 Schedule，由同一套代码评分。"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np

from .config import Config
from .data import Dataset


@dataclass
class Schedule:
    station: np.ndarray   # int
    target: np.ndarray    # int（数据集下标）
    start_s: np.ndarray   # int 秒，相对规划开始
    dur_s: np.ndarray     # int 秒，有效观测时长
    algo: str = ""
    runtime_s: float = 0.0
    info: dict = field(default_factory=dict)

    @classmethod
    def from_cells(cls, cfg: Config, station, target, a, m, **kw) -> "Schedule":
        return cls(np.asarray(station, np.int64), np.asarray(target, np.int64),
                   np.asarray(a, np.int64) * cfg.cell_s, np.asarray(m, np.int64) * cfg.cell_s, **kw)

    def __len__(self):
        return len(self.station)


def task_score(cfg: Config, P: np.ndarray, dur_s: np.ndarray) -> np.ndarray:
    d = np.asarray(dur_s, float)
    frac = 0.6 + 0.4 * np.clip(d - cfg.obs_min_s, 0, cfg.obs_max_s - cfg.obs_min_s) / (cfg.obs_max_s - cfg.obs_min_s)
    return np.where(d >= cfg.obs_min_s, P * frac, 0.0)


def score_ref(cfg: Config, ds: Dataset) -> float:
    return float(ds.priority.sum()) if cfg.score_ref == "sumP" else float(cfg.score_ref)


def counted_mask(cfg: Config, sch: Schedule, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """哪些任务计分：合法，且与同一目标上一次计分观测的结束相隔 48 小时以上（按时间顺序贪心认定）。"""
    keep = np.ones(len(sch), bool) if valid_mask is None else valid_mask.copy()
    order = np.lexsort((sch.start_s, sch.target))
    last_end = {}
    counted = np.zeros(len(sch), bool)
    gap = cfg.repeat_gap_h * 3600
    for i in order:
        if not keep[i]:
            continue
        t = int(sch.target[i])
        if t in last_end and sch.start_s[i] < last_end[t] + gap:
            continue
        counted[i] = True
        last_end[t] = sch.start_s[i] + sch.dur_s[i]
    return counted


def evaluate(cfg: Config, ds: Dataset, sch: Schedule, valid_mask: np.ndarray | None = None) -> dict:
    """按规则计分：只有满足 48 小时间隔的观测才计分（按时间顺序贪心认定）。"""
    counted = counted_mask(cfg, sch, valid_mask)
    sc = task_score(cfg, ds.priority[sch.target], sch.dur_s) * counted
    covered = np.unique(sch.target[counted])
    n = ds.n_targets
    total = float(sc.sum())
    res = {
        "tasks": int(counted.sum()),
        "covered": int(len(covered)),
        "coverage_rate": len(covered) / n,
        "score": total,
        "combined": cfg.w_cov * len(covered) / n + cfg.w_score * total / score_ref(cfg, ds),
        "repeat_tasks": int(counted.sum() - len(covered)),
        "mean_dur_s": float(sch.dur_s[counted].mean()) if counted.any() else 0.0,
    }
    for lab in cfg.priority_score:
        m = np.array([l == lab for l in ds.priority_label])
        res[f"covered_{lab}"] = int(m[covered].sum())
        res[f"total_{lab}"] = int(m.sum())
    busy = np.bincount(sch.station[counted], weights=sch.dur_s[counted] + cfg.switch_s, minlength=ds.n_stations)
    res["station_busy_h"] = (busy / 3600).round(1).tolist()
    return res


def upper_bound(cfg: Config, ds: Dataset, win, max_obs: np.ndarray) -> dict:
    """松弛上界：忽略窗口对齐，只保留站点总可用时间、每目标最多观测次数和计分规则。
    物品按"每秒收益"排序做分数背包，结果对任何合法计划都成立。"""
    cap = sum((b - a) * cfg.cell_s + cfg.switch_s for runs in win.dark for a, b in runs)
    n = ds.n_targets
    sref = score_ref(cfg, ds)
    vc, vs = cfg.w_cov / n, cfg.w_score / sref
    base_t = cfg.obs_min_s + cfg.switch_s
    ext_t = cfg.obs_max_s - cfg.obs_min_s
    items = []  # (每秒价值, 占用时间, 价值)
    for i in range(n):
        k = int(max_obs[i])
        if k == 0:
            continue
        P = ds.priority[i]
        items.append(((vc + vs * 0.6 * P) / base_t, base_t, vc + vs * 0.6 * P))
        for _ in range(k - 1):
            items.append((vs * 0.6 * P / base_t, base_t, vs * 0.6 * P))
        for _ in range(k):
            items.append((vs * 0.4 * P / ext_t, ext_t, vs * 0.4 * P))
    items.sort(key=lambda x: -x[0])
    left, val = float(cap), 0.0
    for dens, t, v in items:
        if left <= 0:
            break
        take = min(1.0, left / t)
        val += take * v
        left -= take * t
    return {"combined_ub": val, "capacity_h": cap / 3600}


def export_csv(cfg: Config, ds: Dataset, sch: Schedule, path: Path, check: dict | None = None):
    order = np.lexsort((sch.start_s, sch.station))
    t0 = cfg.start_dt
    sc = task_score(cfg, ds.priority[sch.target], sch.dur_s)
    seen = set()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "station_id", "target_id", "target_name", "priority", "observation_start_utc",
                    "observation_end_utc", "duration_seconds", "switch_release_time_utc", "elevation_min",
                    "elevation_max", "sun_elevation_max", "task_score", "coverage_type", "algorithm_name"])
        by_time = np.argsort(sch.start_s, kind="stable")
        first = {}
        for i in by_time:
            first.setdefault(int(sch.target[i]), i)
        for n, i in enumerate(order, 1):
            s = t0 + timedelta(seconds=int(sch.start_s[i]))
            e = s + timedelta(seconds=int(sch.dur_s[i]))
            el = check["per_task"][i] if check else (np.nan, np.nan, np.nan)
            w.writerow([n, int(sch.station[i]) + 1, int(ds.satnum[sch.target[i]]), ds.names[sch.target[i]],
                        ds.priority_label[sch.target[i]], s.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        e.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        int(sch.dur_s[i]), (e + timedelta(seconds=cfg.switch_s)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        f"{el[0]:.3f}", f"{el[1]:.3f}", f"{el[2]:.3f}", f"{sc[i]:.2f}",
                        "FIRST" if first[int(sch.target[i])] == i else "REPEAT_48H", sch.algo])
