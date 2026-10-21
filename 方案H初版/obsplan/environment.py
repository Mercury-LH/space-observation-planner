"""加分项：天气（云量）与天体环境（月光）。

天气：用 Open-Meteo 历史再分析数据（ERA5 等）取各站过去若干年同一日历时段（默认 10 月 1–7 日）的逐小时总云量，
- 每一年当作一个"天气情景"：一个任务只有在它覆盖的每个小时云量都 ≤ 阈值时才算观测成功；
- 各年平均得到每站每小时的晴空概率，用于"考虑天气的规划"（期望价值最大化）。
只向外部服务发送四舍五入到 0.5° 的站点经纬度，不发送任何其他数据；结果缓存在 cache/。

月光：在 visibility.py 中作为可选约束（目标与月亮的角距 ≥ moon_sep_deg，月亮在地平线以上时生效）。
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .config import Config
from .data import Dataset
from .schedule import Schedule, task_score

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def _round_half(x: float) -> float:
    return round(x * 2) / 2


def fetch_cloud_history(cfg: Config, ds: Dataset, years=range(2016, 2026), verbose=True) -> np.ndarray:
    """返回 (年数, 站点数, 小时数) 的总云量（%），小时从规划开始时刻起算。"""
    hours = int(round(cfg.days * 24))
    path = Path(cfg.cache_dir) / f"cloud_{cfg.start_utc[5:10]}_{hours}h_{min(years)}-{max(years)}.npz"
    if path.exists():
        return np.load(path)["cloud"]
    start = cfg.start_dt
    out = np.full((len(years), ds.n_stations, hours), np.nan)
    for j in range(ds.n_stations):
        lat, lon = _round_half(float(ds.st_lat[j])), _round_half(float(ds.st_lon[j]))
        a = datetime(min(years), start.month, start.day, tzinfo=timezone.utc)
        b = datetime(max(years), start.month, start.day, tzinfo=timezone.utc) + timedelta(hours=hours)
        url = (f"{ARCHIVE}?latitude={lat}&longitude={lon}&start_date={a:%Y-%m-%d}&end_date={b:%Y-%m-%d}"
               f"&hourly=cloud_cover&timezone=GMT")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    d = json.loads(r.read())
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
        times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in d["hourly"]["time"]]
        idx = {t: k for k, t in enumerate(times)}
        vals = d["hourly"]["cloud_cover"]
        for y, year in enumerate(years):
            t0 = start.replace(year=year)
            for h in range(hours):
                k = idx.get(t0 + timedelta(hours=h))
                if k is not None and vals[k] is not None:
                    out[y, j, h] = vals[k]
        if verbose:
            print(f"  [天气] {j + 1}号站 ({lat}, {lon}) 平均云量 {np.nanmean(out[:, j]):.0f}%")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, cloud=out, years=np.array(list(years)))
    return out


def clear_probability(cfg: Config, cloud: np.ndarray, threshold: float) -> np.ndarray:
    """(站点, 格) 晴空概率：各年中该小时云量 ≤ 阈值的比例，按时间格展开。"""
    clear = (cloud <= threshold).astype(float)
    clear[np.isnan(cloud)] = np.nan
    p_hour = np.nanmean(clear, axis=0)                       # (S, H)
    cells_per_hour = int(3600 / cfg.cell_s)
    T = cfg.n_cells + 2
    q = np.repeat(p_hour, cells_per_hour, axis=1)
    if q.shape[1] < T:
        q = np.concatenate([q, np.repeat(q[:, -1:], T - q.shape[1], axis=1)], axis=1)
    return np.nan_to_num(q[:, :T], nan=0.5)


def evaluate_under_weather(cfg: Config, ds: Dataset, sch: Schedule, cloud: np.ndarray, threshold: float) -> dict:
    """逐年回放：任务覆盖的每个小时云量都 ≤ 阈值才算成功。返回各年与平均的覆盖数、得分。"""
    h0 = (sch.start_s // 3600).astype(int)
    h1 = ((sch.start_s + sch.dur_s - 1) // 3600).astype(int)
    sc = task_score(cfg, ds.priority[sch.target], sch.dur_s)
    cov, score = [], []
    for y in range(cloud.shape[0]):
        ok = np.array([np.all(cloud[y, s, a:b + 1] <= threshold) for s, a, b in zip(sch.station, h0, h1)])
        cov.append(int(np.unique(sch.target[ok]).size))
        score.append(float(sc[ok].sum()))
    return {"coverage_mean": float(np.mean(cov)), "score_mean": float(np.mean(score)),
            "coverage_by_year": cov, "score_by_year": score,
            "success_rate": float(np.mean([c for c in cov]) / max(np.unique(sch.target).size, 1))}
