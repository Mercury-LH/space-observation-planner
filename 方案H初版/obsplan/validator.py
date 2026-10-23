"""独立约束校验器，以 Skyfield 为参考标准（答疑 #26：可见窗口以 Skyfield 计算结果为参考）。

刻意不复用规划器的窗口代码：逐任务按 1 秒采样，用 Skyfield 重新计算
目标仰角（TEME -> GCRS -> ITRS -> 站心）、站点太阳高度角（DE421 星历）和目标是否受照（is_sunlit）。
另外用 TLE 自身的近地点/远地点检查 SGP4 结果是否可信（Skyfield 不检查 SGP4 错误码）。
"""
from __future__ import annotations

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


def validate(cfg: Config, ds: Dataset, sch: Schedule, step_s: float = 1.0) -> dict:
    from skyfield.api import EarthSatellite, wgs84

    ts, eph = _sky(str(Path(cfg.cache_dir)))
    earth, sun, moon = eph["earth"], eph["sun"], eph["moon"]
    st0 = cfg.start_dt
    n = len(sch)
    horizon = cfg.days * 86400
    bad = defaultdict(list)
    per_task = np.full((n, 3), np.nan)   # 最低仰角、最高仰角、最高太阳高度角

    def times(secs):
        return ts.utc(st0.year, st0.month, st0.day, st0.hour, st0.minute, st0.second + secs)

    for i in range(n):
        if not (0 <= sch.station[i] < ds.n_stations and 0 <= sch.target[i] < ds.n_targets):
            bad["编号不存在"].append(i)
        if sch.start_s[i] < 0 or sch.start_s[i] + sch.dur_s[i] > horizon:
            bad["超出规划时间范围"].append(i)
        if sch.dur_s[i] < cfg.obs_min_s:
            bad["观测时长不足120秒"].append(i)

    task_secs = [np.arange(sch.start_s[i], sch.start_s[i] + sch.dur_s[i] + 1e-9, step_s) for i in range(n)]
    sites = [wgs84.latlon(float(ds.st_lat[j]), float(ds.st_lon[j]), elevation_m=float(ds.st_alt_m[j]))
             for j in range(ds.n_stations)]

    # 站点太阳高度角：每站一次性计算
    for j in range(ds.n_stations):
        idx = np.nonzero(sch.station == j)[0]
        if len(idx) == 0:
            continue
        secs = np.concatenate([task_secs[i] for i in idx])
        alt = (earth + sites[j]).at(times(secs)).observe(sun).apparent().altaz()[0].degrees
        pos = 0
        for i in idx:
            k = len(task_secs[i])
            per_task[i, 2] = alt[pos:pos + k].max()
            pos += k
            if cfg.use_sun and per_task[i, 2] >= cfg.sun_max:
                bad["站点太阳高度角不满足"].append(i)

    by_target = defaultdict(list)
    for i in range(n):
        by_target[int(sch.target[i])].append(i)
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
            j = int(sch.station[i])
            alt = (sat - sites[j]).at(tt[sl]).altaz()[0].degrees
            per_task[i, 0], per_task[i, 1] = alt.min(), alt.max()
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
                if np.any((malt > 0) & (sep < cfg.moon_sep_deg)):
                    bad["离月亮太近"].append(i)

    for j in range(ds.n_stations):
        idx = np.nonzero(sch.station == j)[0]
        idx = idx[np.argsort(sch.start_s[idx], kind="stable")]
        for a, b in zip(idx[:-1], idx[1:]):
            if sch.start_s[b] < sch.start_s[a] + sch.dur_s[a] + cfg.switch_s:
                bad["站点时间冲突或切换不足"].append(int(b))

    gap = cfg.repeat_gap_h * 3600
    for t, idx in by_target.items():
        idx = sorted(idx, key=lambda i: sch.start_s[i])
        for a, b in zip(idx[:-1], idx[1:]):
            if sch.start_s[b] < sch.start_s[a] + sch.dur_s[a] + gap:
                bad["同目标间隔不足48小时"].append(b)

    invalid = np.zeros(n, bool)
    for v in bad.values():
        invalid[v] = True
    return {
        "legal": not bad,
        "violations": {k: len(v) for k, v in bad.items()},
        "invalid_mask": invalid,
        "per_task": per_task,
        "min_elev_margin": float(np.nanmin(per_task[:, 0]) - cfg.elev_min) if n else None,
        "max_elev_margin": float(cfg.elev_max - np.nanmax(per_task[:, 1])) if n else None,
        "sun_margin": float(cfg.sun_max - np.nanmax(per_task[:, 2])) if n and cfg.use_sun else None,
    }
