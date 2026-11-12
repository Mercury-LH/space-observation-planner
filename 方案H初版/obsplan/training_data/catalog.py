"""Original synthetic SGP4 mean elements, not a catalog of actual satellites.

Round-trip through exported TLEs before propagation. Each seed owns a distinct
catalog; priority/station variants of that catalog must remain in one split.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, WGS72, jday
from sgp4.exporter import export_tle

from obsplan.data import Dataset, tle_checksum_ok
from obsplan.visibility import Windows

FAMILIES = ["LEO", "MEO", "GEO_like"]
# Land-based anchor regions; these are fictitious instruments, not operational sites.
# lon(deg), lat(deg), height(m); jitter <= 0.08deg and height <= 30m.
STATION_ANCHORS = np.array([
    [-111.6, 31.95, 1800], [-70.8, -30.2, 1600], [20.8, -32.4, 1400],
    [149.1, -31.3, 1100], [-3.5, 40.4, 700], [78.6, 17.4, 550],
    [87.6, 43.8, 900], [106.9, 47.9, 1400], [91.0, 29.6, 3600],
    [116.5, 40.0, 100],
], dtype=float)


def catalog_specs() -> list[dict]:
    """Frozen 20/5/5 catalog split, each with four scenes (80/20/20)."""
    out = []
    for split, sizes in [("train", [1000] * 12 + [5000] * 5 + [10000] * 3),
                         ("validation", [1000] * 3 + [5000, 10000]),
                         ("test", [1000] * 3 + [5000, 10000])]:
        for j, n in enumerate(sizes):
            k = len(out)
            # Different dates/seeds, never propagate an unchanged old TLE into a new season.
            day = datetime(2026, 1, 10, tzinfo=timezone.utc) + timedelta(days=(k * 37) % 330)
            out.append(dict(catalog_id=f"cat-{k:03d}", split=split, n_targets=n,
                            seed=2026110700 + k, start_utc=day.isoformat(),
                            distribution="high_leo_shift" if split == "test" and j in (2, 4) else "nominal"))
    return out


def generate_catalog(spec: dict) -> tuple[Dataset, dict[str, np.ndarray]]:
    rng = np.random.default_rng(spec["seed"])
    n = int(spec["n_targets"])
    shifted = spec.get("distribution") == "high_leo_shift"
    family = rng.choice(3, n, p=[0.90, 0.05, 0.05] if shifted else [0.70, 0.20, 0.10])
    alt = np.empty(n)
    ecc = np.empty(n)
    inc = np.empty(n)
    for f, lo, hi, emax in [(0, 550, 1600, 0.012), (1, 8000, 24000, 0.08),
                            (2, 35000, 36500, 0.025)]:
        m = family == f
        alt[m] = rng.uniform(lo, hi, m.sum())
        ecc[m] = rng.uniform(0, emax, m.sum())
        if f == 0:
            inc[m] = rng.uniform(70 if shifted else 20, 110, m.sum())
        elif f == 1:
            inc[m] = rng.uniform(20, 80, m.sum())
        else:
            inc[m] = rng.uniform(0, 20, m.sum())
    semimajor = 6378.135 + alt
    perigee = semimajor * (1 - ecc) - 6378.135
    if (perigee < 450).any():
        raise ValueError("synthetic perigee below the frozen 450 km minimum")
    raan, argp, anomaly = rng.uniform(0, 360, (3, n))
    # Positive weak drag for LEO; zero for higher orbits. Values are modeling assumptions.
    bstar = np.where(family == 0, rng.uniform(1e-6, 3e-5, n), 0.0)
    mean_motion = np.sqrt(398600.8 / semimajor**3) * 60.0  # rad/min (WGS72)
    epoch = datetime.fromisoformat(spec["start_utc"]) + timedelta(days=3)
    jd, fr = jday(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second)
    nums = np.arange(10001, 10001 + n, dtype=np.int64)
    l1, l2 = [], []
    for i in range(n):
        sat = Satrec()
        sat.sgp4init(WGS72, "i", int(nums[i]), jd + fr - 2433281.5,
                     float(bstar[i]), 0.0, 0.0, float(ecc[i]), np.radians(argp[i]),
                     np.radians(inc[i]), np.radians(anomaly[i]), float(mean_motion[i]), np.radians(raan[i]))
        a, b = export_tle(sat)
        if not (tle_checksum_ok(a) and tle_checksum_ok(b)):
            raise ValueError("TLE export checksum failure")
        back = Satrec.twoline2rv(a, b)
        err, r, v = back.sgp4(jd, fr)
        if err or not np.isfinite([*r, *v]).all():
            raise ValueError(f"invalid exported TLE for synthetic target {i}: {err}")
        l1.append(a)
        l2.append(b)
    stations = STATION_ANCHORS + rng.uniform([-0.08, -0.08, -30], [0.08, 0.08, 30], (10, 3))
    priority = rng.choice([2., 3., 5.], n, p=[0.5, 0.3, 0.2])
    labels = [{2.: "低", 3.: "中", 5.: "高"}[v] for v in priority]
    ds = Dataset([f"SYN-{spec['catalog_id']}-{i:05d}" for i in range(n)], nums, l1, l2,
                 priority, labels, stations[:, 0], stations[:, 1], stations[:, 2], [])
    table = dict(target_id=np.arange(n, dtype=np.int32), satnum=nums,
                 family=family.astype(np.int8), semimajor_km=semimajor, eccentricity=ecc,
                 inclination_deg=inc, raan_deg=raan, arg_perigee_deg=argp, mean_anomaly_deg=anomaly,
                 bstar=bstar, mean_motion_rad_min=mean_motion, mean_element_perigee_km=perigee,
                 line1=np.array(l1), line2=np.array(l2), names=np.array(ds.names),
                 base_priority=priority.astype(np.float32), stations=stations)
    return ds, table


def make_variant(ds: Dataset, win: Windows, seed: int, variant: int) -> tuple[Dataset, Windows, dict]:
    rng = np.random.default_rng(seed + 100000 * (variant + 1))
    modes = ["balanced", "high_priority_dense", "scarce_targets_prioritized", "half_station_capacity"]
    if variant not in range(4):
        raise ValueError("variant must be 0..3")
    p = rng.choice([2., 3., 5.], ds.n_targets,
                   p=[0.2, 0.3, 0.5] if variant == 1 else [0.5, 0.3, 0.2])
    if variant == 2:
        counts = np.bincount(win.target, minlength=ds.n_targets)
        # Deterministic rarity rank with random tie break; unobservable targets remain explicit.
        order = np.lexsort((rng.random(ds.n_targets), np.where(counts > 0, counts, 10**9)))
        p[:] = 2.
        p[order[:int(0.5 * len(p))]] = 3.
        p[order[:int(0.2 * len(p))]] = 5.
    stations = np.arange(ds.n_stations) if variant < 3 else np.sort(rng.choice(ds.n_stations, 5, replace=False))
    labels = [{2.: "低", 3.: "中", 5.: "高"}[v] for v in p]
    out = Dataset(ds.names, ds.satnum, ds.line1, ds.line2, p, labels,
                  ds.st_lon[stations], ds.st_lat[stations], ds.st_alt_m[stations], [])
    mask = np.isin(win.station, stations)
    remap = np.full(ds.n_stations, -1, dtype=np.int16)
    remap[stations] = np.arange(len(stations))
    w = Windows(remap[win.station[mask]], win.target[mask], win.k0[mask], win.k1[mask],
                [win.dark[s] for s in stations], win.n_samples, win.sgp4_failed.copy())
    return out, w, dict(variant=variant, variant_name=modes[variant], base_station_indices=stations.tolist())


def write_native_input(path: Path, ds: Dataset) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "轨道.txt").write_text("\n".join(x for row in zip(ds.names, ds.line1, ds.line2) for x in row) + "\n", encoding="utf-8")
    (path / "站址.txt").write_text("\n".join(f"{j + 1}: {a:.10f}, {b:.10f}, {c:.6f}" for j, (a,b,c) in enumerate(zip(ds.st_lon, ds.st_lat, ds.st_alt_m))) + "\n", encoding="utf-8")
    for label in ["低", "中", "高"]:
        (path / f"{label}优先级目标.txt").write_text("\n".join(str(int(s)) for s,l in zip(ds.satnum, ds.priority_label) if l == label) + "\n", encoding="utf-8")


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
