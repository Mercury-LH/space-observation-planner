"""独立约束校验器，以 Skyfield 为参考标准（答疑 #26：可见窗口以 Skyfield 计算结果为参考）。

刻意不复用规划器的窗口代码：逐任务按 1 秒采样，用 Skyfield 重新计算
目标仰角（TEME -> GCRS -> ITRS -> 站心）、站点太阳高度角（DE421 星历）和目标是否受照（is_sunlit）。
另外用 TLE 自身的近地点/远地点检查 SGP4 结果是否可信（Skyfield 不检查 SGP4 错误码）。
"""
from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from sgp4.api import Satrec

from .config import Config
from .data import Dataset
from .schedule import Schedule


@lru_cache(maxsize=1)
def _sky(cache_dir: str):
    from skyfield.api import Loader
    load = Loader(cache_dir, verbose=False)
    return load.timescale(builtin=True), load("de421.bsp")


def _report(bad, per_task, cfg, *, geometry_checked: bool) -> dict:
    """Return a consistent result even when malformed input has no task rows."""
    n = len(per_task)
    invalid = np.zeros(n, bool)
    for indices in bad.values():
        for i in indices:
            if 0 <= i < n:
                invalid[i] = True

    def bound(column, threshold, minimum=False):
        values = per_task[:, column]
        values = values[np.isfinite(values)]
        if not len(values):
            return None
        return float(values.min() - threshold if minimum else threshold - values.max())

    return {
        "legal": not bad,
        "violations": {k: len(set(v)) for k, v in bad.items()},
        "invalid_mask": invalid,
        "per_task": per_task,
        "geometry_checked": geometry_checked,
        "min_elev_margin": bound(0, cfg.elev_min, minimum=True),
        "max_elev_margin": bound(1, cfg.elev_max),
        "sun_margin": bound(2, cfg.sun_max) if cfg.use_sun else None,
    }


def _sample_times(start: float, duration: float, step_s: float) -> np.ndarray:
    """Sample at most step_s apart and include the exact end, also off-grid."""
    end = start + duration
    samples = np.arange(start, end, step_s)
    return np.r_[samples[samples < end], end]


def validate(cfg: Config, ds: Dataset, sch: Schedule, step_s: float = 1.0) -> dict:
    """Check structure first, then independent geometry on a <= 1-second mesh.

    Structurally invalid schedules return geometry_checked=False without
    indexing unknown IDs or loading ephemerides. Such a report must never be
    ranked as a partially validated plan. Endpoints are included; finite-step
    geometric sampling remains a numerical check, not an analytic proof.
    """
    if not math.isfinite(step_s) or not 0 < step_s <= 1.0:
        raise ValueError("Validation step_s must be finite and in (0, 1] seconds")
    bad = defaultdict(list)
    try:
        station, target, start, duration = [np.asarray(getattr(sch, field)) for field in
                                             ("station", "target", "start_s", "dur_s")]
    except (AttributeError, TypeError, ValueError):
        bad["计划数组结构不合法"].append(0)
        return _report(bad, np.empty((0, 3)), cfg, geometry_checked=False)
    arrays = (station, target, start, duration)
    n = len(station) if station.ndim else 0
    per_task = np.full((n, 3), np.nan)
    if any(a.ndim != 1 or len(a) != n for a in arrays):
        bad["计划数组形状不一致"].extend(range(n) if n else [0])
        return _report(bad, per_task, cfg, geometry_checked=False)
    if not n:
        return _report(bad, per_task, cfg, geometry_checked=True)
    if any(not np.issubdtype(a.dtype, np.integer) for a in (station, target)):
        bad["编号必须为整数数组"].extend(range(n) if n else [0])
        return _report(bad, per_task, cfg, geometry_checked=False)
    if any(not (np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.floating))
           for a in (start, duration)):
        bad["时间必须为实数数组"].extend(range(n) if n else [0])
        return _report(bad, per_task, cfg, geometry_checked=False)
    # Float conversion prevents integer overflow in start + duration below.
    start, duration = start.astype(float), duration.astype(float)
    with np.errstate(over="ignore", invalid="ignore"):
        end = start + duration
    finite = np.isfinite(start) & np.isfinite(duration) & np.isfinite(end)
    bad_ids = (station < 0) | (station >= ds.n_stations) | (target < 0) | (target >= ds.n_targets)
    for i in range(n):
        if bad_ids[i]:
            bad["编号不存在"].append(i)
        if not finite[i]:
            bad["时间包含非有限数值"].append(i)
            continue
        if start[i] < 0 or end[i] > cfg.days * 86400:
            bad["超出规划时间范围"].append(i)
        if duration[i] < cfg.obs_min_s:
            bad["观测时长不足120秒"].append(i)
    if bad:
        return _report(bad, per_task, cfg, geometry_checked=False)

    # Check against the furthest preceding end, not merely the previous row:
    # a long task can contain several shorter tasks, each of which conflicts.
    for j in np.unique(station):
        idx = np.nonzero(station == j)[0]
        idx = idx[np.argsort(start[idx], kind="stable")]
        latest_end = -np.inf
        for i in idx:
            if start[i] < latest_end + cfg.switch_s:
                bad["站点时间冲突或切换不足"].append(int(i))
            latest_end = max(latest_end, end[i])
    by_target = defaultdict(list)
    for i in range(n):
        by_target[int(target[i])].append(i)
    gap = cfg.repeat_gap_h * 3600
    for idx in by_target.values():
        latest_end = -np.inf
        for i in sorted(idx, key=lambda i: start[i]):
            if start[i] < latest_end + gap:
                bad["同目标间隔不足48小时"].append(i)
            latest_end = max(latest_end, end[i])
    if bad:
        return _report(bad, per_task, cfg, geometry_checked=False)

    from skyfield.api import EarthSatellite, wgs84
    ts, eph = _sky(str(Path(cfg.cache_dir)))
    earth, sun, moon = eph["earth"], eph["sun"], eph["moon"]
    st0 = cfg.start_dt

    def times(secs):
        return ts.utc(st0.year, st0.month, st0.day, st0.hour, st0.minute, st0.second + secs)

    task_secs = [_sample_times(start[i], duration[i], step_s) for i in range(n)]
    sites = [wgs84.latlon(float(ds.st_lat[j]), float(ds.st_lon[j]), elevation_m=float(ds.st_alt_m[j]))
             for j in range(ds.n_stations)]

    # 站点太阳高度角：每站一次性计算
    for j in range(ds.n_stations):
        idx = np.nonzero(station == j)[0]
        if len(idx) == 0:
            continue
        secs = np.concatenate([task_secs[i] for i in idx])
        alt = (earth + sites[j]).at(times(secs)).observe(sun).apparent().altaz()[0].degrees
        pos = 0
        for i in idx:
            k = len(task_secs[i])
            per_task[i, 2] = alt[pos:pos + k].max()
            pos += k
            if not np.isfinite(alt[pos - k:pos]).all():
                bad["太阳几何计算非有限"].append(i)
            if cfg.use_sun and per_task[i, 2] >= cfg.sun_max:
                bad["站点太阳高度角不满足"].append(i)

    for t, idx in by_target.items():
        sat = EarthSatellite(ds.line1[t], ds.line2[t], ts=ts)
        raw = Satrec.twoline2rv(ds.line1[t], ds.line2[t])
        a_km = (398600.4418 / (raw.no_kozai / 60.0) ** 2) ** (1 / 3)
        secs = np.concatenate([task_secs[i] for i in idx])
        tt = times(secs)
        geo = sat.at(tt)
        rn = np.linalg.norm(geo.position.km, axis=0)
        implausible = ~np.isfinite(rn) | (rn < 0.9 * a_km * (1 - raw.ecco)) | (rn > 1.1 * a_km * (1 + raw.ecco))
        lit = geo.is_sunlit(eph) if cfg.use_sunlit else np.ones(len(secs), bool)
        pos = 0
        for i in idx:
            k = len(task_secs[i])
            sl = slice(pos, pos + k)
            pos += k
            j = int(station[i])
            alt = (sat - sites[j]).at(tt[sl]).altaz()[0].degrees
            per_task[i, 0], per_task[i, 1] = alt.min(), alt.max()
            if not np.isfinite(alt).all():
                bad["仰角计算非有限"].append(i)
            if implausible[sl].any():
                bad["轨道传播失败或不可信"].append(i)
            if alt.min() < cfg.elev_min:
                bad["仰角低于下限"].append(i)
            if alt.max() > cfg.elev_max:
                bad["仰角高于上限"].append(i)
            if not np.all(lit[sl]):
                bad["目标未被太阳照亮"].append(i)
            if cfg.moon_sep_deg > 0:
                obs = (earth + sites[j]).at(tt[sl])
                mpos = obs.observe(moon).apparent()
                malt = mpos.altaz()[0].degrees
                sv = (sat - sites[j]).at(tt[sl]).position.km
                mv = mpos.position.km
                cos = np.einsum("ij,ij->j", sv, mv) / (np.linalg.norm(sv, axis=0) * np.linalg.norm(mv, axis=0))
                sep = np.degrees(np.arccos(np.clip(cos, -1, 1)))
                if not np.isfinite(malt).all() or not np.isfinite(sep).all():
                    bad["月光几何计算非有限"].append(i)
                if np.any((malt > 0) & (sep < cfg.moon_sep_deg)):
                    bad["离月亮太近"].append(i)

    return _report(bad, per_task, cfg, geometry_checked=True)
