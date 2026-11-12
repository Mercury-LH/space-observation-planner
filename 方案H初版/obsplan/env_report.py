"""加分项：环境影响报告——一份计划在"考虑地球天气与太空环境"之后还能完成多少。

说明文档原话："题目没有考虑地球天气和太空天体环境变化，如果可以额外考虑，输出标准结果和考虑之后的结果，可以酌情加分。"
这里对任意一份计划逐任务算三项环境因素，再给出"名义结果"和"考虑环境之后的结果"：

- 月光（太空环境）：观测期间月亮在地平线以上、且目标离月亮不到 moon_sep_deg 度时，天空背景太亮，按观测失败计。
  月亮位置用 DE421 星历（与 Skyfield 校验同一份），目标位置用 SGP4（与窗口计算同一模型），每 step_s 秒采样一次。
- 目标亮度（太空环境）：光学望远镜只能看到足够亮的目标。用漫反射球模型估算目标视星等
  m = m_std + 5·log10(距离 / 1000 km) − 2.5·log10(π·F(φ))，F(φ) = (sin φ + (π − φ)·cos φ) / π，φ 为相位角
  （太阳—目标—测站的夹角）。赛题数据没有目标尺寸，m_std（1000 km、相位角 90° 时的星等）统一取一个可调的典型值，
  暗于 limit_mag 的任务按观测失败计。这一项依赖假设，报告里同时给出几个极限星等下的结果。
- 天气（地球天气）：把计划放到过去 10 年同期每个站的真实逐小时云量里（Open-Meteo 历史再分析，云量 ≤ 30% 算晴），
  观测期间每个小时都晴才算成功，取 10 年平均（environment.evaluate_under_weather）。

"考虑之后的结果"= 去掉被月光、亮度判为失败的任务后再按天气取期望；覆盖和得分都按赛题口径（48 小时间隔）重算。
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .data import Dataset
from .schedule import Schedule, evaluate, score_ref

CLOUD_THRESHOLD = 30.0


def task_geometry(cfg: Config, ds: Dataset, sch: Schedule, step_s: float = 10.0) -> dict:
    """逐任务按 step_s 秒采样：月亮是否在地平线以上、目标离月亮的角距、目标视星等所需的距离与相位角。
    返回每个任务的 最小月距（只看月亮在地平线以上的时刻，否则为 180°）、最大距离、最大相位角。"""
    from sgp4.api import Satrec
    from .visibility import gmst82, julian, moon_ecef, station_ecef, sun_unit_teme, teme_to_ecef
    n = len(sch)
    t0 = np.asarray(sch.start_s, float)
    t1 = t0 + np.asarray(sch.dur_s, float)
    k = np.maximum(np.ceil((t1 - t0) / step_s).astype(int), 1) + 1          # 每个任务的采样点数（含两端）
    owner = np.repeat(np.arange(n), k)
    frac = np.concatenate([np.linspace(0.0, 1.0, m) for m in k])
    sec = t0[owner] + frac * (t1 - t0)[owner]
    jd, fr = julian(cfg, sec)
    theta = gmst82(jd, fr)
    r = np.full((len(sec), 3), np.nan)
    tg = np.asarray(sch.target)[owner]
    order = np.argsort(tg, kind="stable")
    bounds = np.flatnonzero(np.diff(tg[order])) + 1
    for idx in np.split(order, bounds):
        t = int(tg[idx[0]])
        sat = Satrec.twoline2rv(ds.line1[t], ds.line2[t])
        err, rr, _ = sat.sgp4_array(jd[idx], fr[idx])
        rr = np.asarray(rr, float)
        rr[np.asarray(err) != 0] = np.nan
        r[idx] = rr
    r = teme_to_ecef(r, theta)
    pos, up = station_ecef(ds.st_lon, ds.st_lat, ds.st_alt_m)
    st = np.asarray(sch.station)[owner]
    sp, su = pos[st], up[st]
    los = r - sp
    rng = np.linalg.norm(los, axis=1)
    los_u = los / rng[:, None]
    moon = moon_ecef(cfg, sec) - sp
    moon_u = moon / np.linalg.norm(moon, axis=1)[:, None]
    moon_up = np.einsum("nk,nk->n", moon_u, su) > 0
    sep = np.degrees(np.arccos(np.clip(np.einsum("nk,nk->n", los_u, moon_u), -1, 1)))
    sep = np.where(moon_up, sep, 180.0)
    sun = teme_to_ecef(sun_unit_teme(jd, fr), theta)                         # 太阳方向（单位向量，地固系）
    phase = np.degrees(np.arccos(np.clip(np.einsum("nk,nk->n", sun, -los_u), -1, 1)))
    out_sep = np.full(n, 180.0)
    out_rng = np.zeros(n)
    out_phase = np.zeros(n)
    np.minimum.at(out_sep, owner, sep)
    np.maximum.at(out_rng, owner, np.nan_to_num(rng, nan=0.0))
    np.maximum.at(out_phase, owner, np.nan_to_num(phase, nan=0.0))
    return {"moon_sep_min": out_sep, "range_max_km": out_rng, "phase_max_deg": out_phase}


def magnitude(range_km: np.ndarray, phase_deg: np.ndarray, m_std: float) -> np.ndarray:
    """漫反射球的视星等：m_std 是 1000 km、相位角 90° 时的星等。"""
    ph = np.radians(np.clip(phase_deg, 0.0, 179.9))
    f = (np.sin(ph) + (np.pi - ph) * np.cos(ph)) / np.pi
    return m_std + 5.0 * np.log10(np.maximum(range_km, 1.0) / 1000.0) - 2.5 * np.log10(np.maximum(np.pi * f, 1e-6))


def _metrics(cfg: Config, ds: Dataset, sch: Schedule, keep: np.ndarray, cloud: np.ndarray | None) -> dict:
    ev = evaluate(cfg, ds, sch, keep)
    out = {"tasks": int(keep.sum()), "covered": int(ev["covered"]), "score": round(float(ev["score"]), 1),
           "combined": round(float(ev["combined"]), 6)}
    if cloud is not None:
        from .environment import evaluate_under_weather
        sub = Schedule(sch.station[keep], sch.target[keep], sch.start_s[keep], sch.dur_s[keep])
        w = evaluate_under_weather(cfg, ds, sub, cloud, CLOUD_THRESHOLD)
        out["weather_covered_mean"] = round(w["coverage_mean"], 1)
        out["weather_score_mean"] = round(w["score_mean"], 1)
        out["weather_combined_mean"] = round(cfg.w_cov * w["coverage_mean"] / ds.n_targets
                                             + cfg.w_score * w["score_mean"] / score_ref(cfg, ds), 6)
    return out


def report(cfg: Config, ds: Dataset, sch: Schedule, moon_sep_deg: float = 30.0, m_std: float = 6.0,
           limit_mag: float = 15.0, limits=(13.0, 14.0, 15.0, 16.0), weather: bool = True, geom: dict | None = None) -> dict:
    """一份计划的环境影响：每项因素各影响多少任务，名义结果与考虑之后的结果。"""
    g = task_geometry(cfg, ds, sch) if geom is None else geom
    moon_bad = g["moon_sep_min"] < moon_sep_deg
    mag = magnitude(g["range_max_km"], g["phase_max_deg"], m_std)
    dim_bad = mag > limit_mag
    cloud = None
    if weather:
        from .environment import fetch_cloud_history
        cloud = fetch_cloud_history(cfg, ds, verbose=False)
    all_ok = np.ones(len(sch), bool)
    res = {
        "assumptions": {"moon_sep_deg": moon_sep_deg, "m_std": m_std, "limit_mag": limit_mag,
                        "cloud_threshold_pct": CLOUD_THRESHOLD, "weather_years": None if cloud is None else int(cloud.shape[0])},
        "affected_tasks": {"moon": int(moon_bad.sum()), "dim": int(dim_bad.sum()), "moon_or_dim": int((moon_bad | dim_bad).sum())},
        "dim_by_limit": {f"{lm:g}": int((mag > lm).sum()) for lm in limits},
        "nominal": _metrics(cfg, ds, sch, all_ok, cloud),
        "after_moon": _metrics(cfg, ds, sch, ~moon_bad, None),
        "after_moon_dim": _metrics(cfg, ds, sch, ~(moon_bad | dim_bad), cloud),
        "magnitude_pct": {p: round(float(np.percentile(mag, p)), 2) for p in (50, 90, 99)},
    }
    return res
