"""SGP4 传播 + 可见窗口提取。

时间离散：采样点 k = 0..n_cells，对应 start + k*cell_s。任务的起止都落在采样点上，
任务 [a, a+m] 合法 <=> 采样点 a..a+m 全部可见，且相邻采样点之间没有越过 85° 的天顶穿越。
- 仰角下限：同一次过境内仰角是单峰的，两端都 >=10° 则中间也 >=10°；
- 仰角上限：近天顶过境在 30 秒内可能先升过 85° 再落回，对这类采样对做 1 秒加密检查；
- 太阳高度角 30 秒内单调，两端都满足则中间也满足。
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, SatrecArray, jday

from .config import Config
from .data import Dataset

WGS84_A = 6378.137
WGS84_F = 1 / 298.257223563


def julian(cfg: Config, seconds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = cfg.start_dt
    jd0, fr0 = jday(s.year, s.month, s.day, s.hour, s.minute, s.second)
    fr = fr0 + seconds / 86400.0
    whole = np.floor(fr)
    return np.full(seconds.shape, jd0) + whole, fr - whole


def gmst82(jd: np.ndarray, fr: np.ndarray) -> np.ndarray:
    """IAU-82 格林尼治平恒星时（弧度），TEME -> PEF 用的就是它。"""
    tut1 = (jd - 2451545.0 + fr) / 36525.0
    g = (-6.2e-6 * tut1**3 + 0.093104 * tut1**2
         + (876600.0 * 3600 + 8640184.812866) * tut1 + 67310.54841)
    return np.radians((g % 86400.0) / 240.0)


def sun_unit_teme(jd: np.ndarray, fr: np.ndarray) -> np.ndarray:
    """低精度太阳方向（约 0.01°），惯性系近似当作 TEME。"""
    n = jd - 2451545.0 + fr
    L = np.radians((280.460 + 0.9856474 * n) % 360.0)
    g = np.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 4.0e-7 * n)
    return np.stack([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)], -1)


def station_ecef(lon_deg, lat_deg, alt_m):
    lon, lat, h = np.radians(lon_deg), np.radians(lat_deg), np.asarray(alt_m) / 1000.0
    e2 = WGS84_F * (2 - WGS84_F)
    N = WGS84_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    pos = np.stack([(N + h) * np.cos(lat) * np.cos(lon), (N + h) * np.cos(lat) * np.sin(lon),
                    (N * (1 - e2) + h) * np.sin(lat)], -1)
    up = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
    return pos, up


def teme_to_ecef(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * r[..., 0] + s * r[..., 1], -s * r[..., 0] + c * r[..., 1], r[..., 2]], -1)


def elevation_deg(r_ecef: np.ndarray, st_pos: np.ndarray, st_up: np.ndarray) -> np.ndarray:
    d = r_ecef - st_pos
    return np.degrees(np.arcsin(np.clip((d @ st_up) / np.linalg.norm(d, axis=-1), -1, 1)))


def sun_elevation(cfg: Config, ds: Dataset, seconds: np.ndarray) -> np.ndarray:
    """(T, S) 各站太阳高度角。"""
    jd, fr = julian(cfg, seconds)
    sun = teme_to_ecef(sun_unit_teme(jd, fr), gmst82(jd, fr))
    _, up = station_ecef(ds.st_lon, ds.st_lat, ds.st_alt_m)
    return np.degrees(np.arcsin(sun @ up.T))


@dataclass
class Windows:
    station: np.ndarray       # (W,) int16
    target: np.ndarray        # (W,) int32
    k0: np.ndarray            # (W,) int32 首个采样点
    k1: np.ndarray            # (W,) int32 末个采样点（含）
    dark: list                # 每站 [(d0, d1), ...] 采样点区间
    n_samples: int
    sgp4_failed: np.ndarray   # (N,) bool，规划期内 SGP4 报错（如已陨落）

    def save(self, path: Path):
        dk = np.array([(s, a, b) for s, runs in enumerate(self.dark) for a, b in runs], dtype=np.int32).reshape(-1, 3)
        np.savez_compressed(path, station=self.station, target=self.target, k0=self.k0, k1=self.k1,
                            dark=dk, n_samples=self.n_samples, sgp4_failed=self.sgp4_failed)

    @classmethod
    def load(cls, path: Path, n_stations: int) -> "Windows":
        z = np.load(path)
        dark = [[] for _ in range(n_stations)]
        for s, a, b in z["dark"]:
            dark[s].append((int(a), int(b)))
        return cls(z["station"], z["target"], z["k0"], z["k1"], dark, int(z["n_samples"]), z["sgp4_failed"])


def _runs(edge_ok: np.ndarray, min_len: int):
    """edge_ok (n, T-1)：采样点 k 与 k+1 之间连通。返回 (row, k0, k1)，k1-k0 >= min_len。"""
    n = edge_ok.shape[0]
    pad = np.zeros((n, 1), np.int8)
    d = np.diff(np.concatenate([pad, edge_ok.astype(np.int8), pad], 1), axis=1)
    rs, cs = np.nonzero(d == 1)
    re_, ce = np.nonzero(d == -1)
    length = ce - cs
    keep = length >= min_len
    return rs[keep], cs[keep], ce[keep]


def moon_ecef(cfg: Config, seconds: np.ndarray) -> np.ndarray:
    """(T, 3) 月亮在地固系（ITRS）中的地心位置，km。"""
    from skyfield.api import Loader
    from skyfield.framelib import itrs
    load = Loader(cfg.cache_dir, verbose=False)
    ts = load.timescale(builtin=True)
    eph = load("de421.bsp")
    s = cfg.start_dt
    t = ts.utc(s.year, s.month, s.day, s.hour, s.minute, s.second + seconds)
    return eph["earth"].at(t).observe(eph["moon"]).frame_xyz(itrs).km.T


def _chunk_job(args):
    (cfg, lines, idx0, st_pos, st_up, dark, moon) = args
    sats = [Satrec.twoline2rv(a, b) for a, b in lines]
    T = cfg.n_cells + 1
    sec = np.arange(T, dtype=np.float64) * cfg.cell_s
    jd, fr = julian(cfg, sec)
    theta = gmst82(jd, fr)
    e, r, _ = SatrecArray(sats).sgp4(jd, fr)
    # 外推太远时 SGP4 可能不报错却给出离谱位置（实测有卫星被算到上千万公里外），
    # 30 秒采样还会把这种乱跳的位置"混叠"成一条看似正常的过境曲线。
    # 用 TLE 自身的近地点/远地点做合理性检查，不合理的整颗目标判为外推失败。
    a_km = np.array([(398600.4418 / (s.no_kozai / 60.0) ** 2) ** (1 / 3) for s in sats])
    ecc = np.array([s.ecco for s in sats])
    rn = np.linalg.norm(r, axis=-1)
    implausible = (e == 0) & ((rn < 0.9 * a_km[:, None] * (1 - ecc[:, None]))
                              | (rn > 1.1 * a_km[:, None] * (1 + ecc[:, None])))
    failed = (e != 0).any(1) | implausible.any(1)
    ok_t = (e == 0) & ~failed[:, None]
    if cfg.use_sunlit:
        # 与 Skyfield is_sunlit 同一思路：球形地球，日-星连线被地球挡住即为地影
        sun = sun_unit_teme(jd, fr)                           # (T, 3)
        proj = np.einsum("ntk,tk->nt", r, sun)
        perp2 = np.einsum("ntk,ntk->nt", r, r) - proj**2
        ok_t &= (proj > 0) | (perp2 > (WGS84_A + cfg.sunlit_margin_km) ** 2)
    recef = teme_to_ecef(r, theta[None, :])
    lo, hi = cfg.elev_min + cfg.elev_margin, cfg.elev_max - cfg.elev_margin
    edges, checks = [], []   # checks: (row, station, k) 需要 1 秒加密检查的采样对 (k, k+1)
    moon_units = []
    for j in range(len(st_pos)):
        el = elevation_deg(recef, st_pos[j], st_up[j])
        vis = ok_t & (el >= lo) & (el <= hi) & dark[None, :, j]
        if moon is not None:
            mv = moon - st_pos[j]
            mu_ = mv / np.linalg.norm(mv, axis=1, keepdims=True)
            # 月亮"在地平线以上"也留余量：月出/月落前后 moon_margin_deg 内按已升起处理（与 Skyfield 视高度有约 0.002° 差异）
            moon_up = mu_ @ st_up[j] > -np.sin(np.radians(cfg.moon_margin_deg))
            d = recef - st_pos[j]
            dhat = d / np.linalg.norm(d, axis=-1, keepdims=True)
            cos_sep = np.einsum("ntk,tk->nt", dhat, mu_)
            limit_deg = cfg.moon_sep_deg + cfg.moon_margin_deg
            vis &= ~moon_up[None, :] | (cos_sep <= np.cos(np.radians(limit_deg)))
            moon_units.append(mu_)
            sep = np.degrees(np.arccos(np.clip(cos_sep, -1, 1)))
            step = np.degrees(np.arccos(np.clip(np.einsum("ntk,ntk->nt", dhat[:, :-1], dhat[:, 1:]), -1, 1)))
            # 两次采样之间，弧上任一点离端点不超过 step/2：只有端点离禁区不足 step/2 时才可能擦进去
            # 两个采样点之间月亮可能升起，任一端月亮在地平线以上都要检查
            near_moon = (moon_up[:-1] | moon_up[1:])[None, :] & (np.minimum(sep[:, :-1], sep[:, 1:]) - step / 2 < limit_deg)
        else:
            near_moon = None
        edge = vis[:, :-1] & vis[:, 1:]
        if near_moon is not None:                 # 可能擦进月亮禁区的采样对做 1 秒加密检查
            for row, k in zip(*np.nonzero(edge & near_moon)):
                checks.append((row, j, k))
        # 过境峰值：真实最高点一定在采样峰值两侧各 30 秒内，近天顶时可能越过 85°
        peak = np.zeros_like(vis)
        peak[:, 1:-1] = vis[:, 1:-1] & (el[:, 1:-1] >= el[:, :-2]) & (el[:, 1:-1] >= el[:, 2:]) & (el[:, 1:-1] > 30.0)
        for row, k in zip(*np.nonzero(peak)):
            for kk in (k - 1, k):
                if edge[row, kk]:
                    checks.append((row, j, kk))
        edges.append(edge)
    if checks:
        checks = np.unique(np.array(checks), axis=0)
        fine = np.arange(1, cfg.cell_s, dtype=np.float64)
        for row in np.unique(checks[:, 0]):
            cr = checks[checks[:, 0] == row]
            secs = (cr[:, 2:3] * cfg.cell_s + fine[None, :]).ravel()
            fj, ff = julian(cfg, secs)
            ee, rr, _ = sats[row].sgp4_array(fj, ff)
            rec = teme_to_ecef(rr, gmst82(fj, ff)).reshape(len(cr), len(fine), 3)
            ee = ee.reshape(len(cr), len(fine))
            for n, (_, j, k) in enumerate(cr):
                fel = elevation_deg(rec[n], st_pos[j], st_up[j])
                bad = (ee[n] != 0).any() or (fel > hi).any() or (fel < lo).any()
                if not bad and moon is not None:
                    mu_ = moon_units[j][k]
                    if (moon_units[j][k:k + 2] @ st_up[j] > -np.sin(np.radians(cfg.moon_margin_deg))).any():
                        d = rec[n] - st_pos[j]
                        cs = (d @ mu_) / np.linalg.norm(d, axis=1)
                        bad = (cs > np.cos(np.radians(cfg.moon_sep_deg + cfg.moon_margin_deg))).any()
                if bad:
                    edges[j][row, k] = False
    out = []
    for j, edge in enumerate(edges):
        rows, k0, k1 = _runs(edge, cfg.min_cells)
        out.append((np.full(len(rows), j, np.int16), (rows + idx0).astype(np.int32), k0.astype(np.int32), k1.astype(np.int32)))
    st = np.concatenate([o[0] for o in out])
    tg = np.concatenate([o[1] for o in out])
    a = np.concatenate([o[2] for o in out])
    b = np.concatenate([o[3] for o in out])
    return st, tg, a, b, failed


def compute_windows(cfg: Config, ds: Dataset, workers: int | None = None, verbose=True,
                    keep_short: bool = False) -> Windows:
    """keep_short：同一遍计算里顺带保留比最短观测少一格的窗口（按秒精修的"按秒才可见"候选），
    放在返回值的 short 属性里 (station, target, k0, k1)。"""
    t0 = time.time()
    if keep_short:
        from dataclasses import replace
        full_cfg, cfg = cfg, replace(cfg, obs_min_s=cfg.obs_min_s - cfg.cell_s)
    T = cfg.n_cells + 1
    sec = np.arange(T, dtype=np.float64) * cfg.cell_s
    if cfg.use_sun:
        dark = sun_elevation(cfg, ds, sec) <= cfg.sun_max - cfg.sun_margin
    else:
        dark = np.ones((T, ds.n_stations), bool)
    dark_runs = []
    for j in range(ds.n_stations):
        _, a, b = _runs((dark[:-1, j] & dark[1:, j])[None, :], 1)
        dark_runs.append(list(zip(a.tolist(), b.tolist())))
    st_pos, st_up = station_ecef(ds.st_lon, ds.st_lat, ds.st_alt_m)
    moon = moon_ecef(cfg, sec) if cfg.moon_sep_deg > 0 else None
    chunk = 200
    jobs = [(cfg, list(zip(ds.line1[i:i + chunk], ds.line2[i:i + chunk])), i, st_pos, st_up, dark, moon)
            for i in range(0, ds.n_targets, chunk)]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_chunk_job, jobs))
    st = np.concatenate([r[0] for r in res])
    tg = np.concatenate([r[1] for r in res])
    k0 = np.concatenate([r[2] for r in res])
    k1 = np.concatenate([r[3] for r in res])
    failed = np.concatenate([r[4] for r in res])
    short = None
    if keep_short:
        is_short = k1 - k0 < full_cfg.min_cells
        short = (st[is_short], tg[is_short], k0[is_short], k1[is_short])
        st, tg, k0, k1 = st[~is_short], tg[~is_short], k0[~is_short], k1[~is_short]
    order = np.lexsort((k0, st))
    w = Windows(st[order], tg[order], k0[order], k1[order], dark_runs, T, failed)
    w.short = short
    if verbose:
        print(f"[windows] {len(w.k0)} 个窗口, {np.unique(w.target).size} 颗目标可观测, "
              f"SGP4 报错 {failed.sum()} 颗, 用时 {time.time() - t0:.1f}s")
    return w


def get_windows(cfg: Config, ds: Dataset, refresh=False) -> Windows:
    path = Path(cfg.cache_dir) / f"windows_{cfg.scenario_key()}.npz"
    if path.exists() and not refresh:
        return Windows.load(path, ds.n_stations)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = compute_windows(cfg, ds, keep_short=True)
    w.save(path)
    st, tg, k0, k1 = w.short
    np.savez_compressed(path.with_name(f"short_{cfg.scenario_key()}.npz"), station=st, target=tg, k0=k0, k1=k1)
    return w
