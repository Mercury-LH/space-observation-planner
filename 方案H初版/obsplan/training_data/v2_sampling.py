"""Frozen, physically bounded sampling for 600 independent v0.2 catalogs.

The Gaussian claim applies to the 540 core catalogs' scalar controls. The 60
explicit stress catalogs, categorical orbit families, calendar months and
periodic angles are different distributions and must be reported separately.
No model outcomes, visibility results or teacher scores select these specs.
"""
from __future__ import annotations

import calendar
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from statistics import NormalDist

import numpy as np
from sgp4.api import Satrec, WGS72, jday
from sgp4.exporter import export_tle

from obsplan.data import Dataset, tle_checksum_ok


SAMPLING_VERSION = "synthetic-v0.2-sampling-1"
CONFIG_SEED = 2027110700
CATALOG_SEED_BASE = 202711070000
CORE_COUNTS = {"train": 420, "validation": 60, "test": 60}
STRESS_TYPES = (
    "high_load", "low_load", "leo_polar", "high_orbit", "high_priority", "regional_network",
)
FAMILIES = ["LEO", "MEO", "GEO_like"]
# Each tuple is mean, standard deviation, lower support, upper support.
CORE_CONTROLS = {
    "n_targets": (7000., 3000., 1000., 16000.),
    "n_stations": (10., 2.5, 4., 16.),
    "leo_fraction": (.70, .10, .40, .92),
    "meo_remaining_fraction": (.65, .12, .25, .90),
    "high_priority_fraction": (.25, .10, .03, .55),
    "medium_remaining_fraction": (.40, .10, .15, .75),
    "leo_altitude_mean_km": (1000., 180., 650., 1450.),
    "leo_inclination_mean_deg": (65., 20., 20., 105.),
}
CDF_TOLERANCE = 1e-12

# Fictitious instruments in broad land regions. Heights are design assumptions,
# not surveyed sites, available observing facilities, or a terrain database.
# The first ten regional anchors retain v0.1's interpretation; every catalog
# has new orbit seeds, new station combinations and small bounded perturbations.
# lon(deg), lat(deg), assumed height(m).
STATION_ANCHORS = np.array([
    [-111.6, 31.95, 1800], [-70.8, -30.2, 1600], [20.8, -32.4, 1400],
    [149.1, -31.3, 1100], [-3.5, 40.4, 700], [78.6, 17.4, 550],
    [87.6, 43.8, 900], [106.9, 47.9, 1400], [91.0, 29.6, 3600],
    [116.5, 40.0, 100], [-110.0, 52.0, 700], [-65.0, -31.0, 600],
    [-47.0, -15.0, 800], [17.0, -22.0, 1600], [-6.0, 32.0, 600],
    [36.0, 0.0, 1700], [35.0, 39.0, 1000], [67.0, 48.0, 500],
    [100.0, 38.0, 2800], [120.0, -26.0, 500], [172.0, -43.0, 400],
    [-100.0, 35.0, 700], [-105.0, 24.0, 1500], [-74.0, -13.0, 2500],
], dtype=float)
REGIONAL_ANCHORS = np.array([5, 6, 7, 8, 9, 17, 18], dtype=np.int16)


def _cdf(x: float, params: tuple | list) -> float:
    mean, sd, lower, upper = params
    d = NormalDist(mean, sd)
    a, b = d.cdf(lower), d.cdf(upper)
    return float((d.cdf(float(x)) - a) / (b - a))


def _ppf(u, params: tuple | list) -> np.ndarray:
    """Inverse truncated-normal CDF, never boundary clipping of normal draws."""
    mean, sd, lower, upper = params
    if not (np.isfinite(params).all() and sd > 0 and lower < upper):
        raise ValueError("invalid truncated-normal parameters")
    u = np.asarray(u, dtype=float)
    if not np.isfinite(u).all() or (u < 0).any() or (u > 1).any():
        raise ValueError("quantiles must be finite and in [0, 1]")
    d = NormalDist(mean, sd)
    a, b = d.cdf(lower), d.cdf(upper)
    ps = a + u.ravel() * (b - a)
    if not (0 <= a < b <= 1):
        raise ValueError("truncation exceeds supported numerical tail range")
    # Very remote normal tails can round to CDF 0 or 1. Keep inverse-CDF
    # arguments finite; this guards floating-point endpoints, not normal draws.
    ps = np.clip(ps, np.nextafter(0., 1.), np.nextafter(1., 0.))
    result = np.fromiter((d.inv_cdf(float(p)) for p in ps), dtype=float, count=len(ps))
    result[u.ravel() == 0] = lower
    result[u.ravel() == 1] = upper
    return result.reshape(u.shape)


def _draw(rng: np.random.Generator, params: tuple | list, size: int) -> np.ndarray:
    return _ppf(rng.random(size), params)


def _integer(x: float) -> int:
    return int(np.floor(float(x) + .5))


def size_bin(n: int) -> str:
    return "small" if n < 4000 else ("medium" if n < 10000 else "large")


def specs_sha256(rows: list[dict]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _orbit_parameters(control: dict) -> dict:
    return {
        "LEO": {
            "altitude_km": [control["leo_altitude_mean_km"], 250., 550., 1800.],
            "eccentricity": [.004, .003, .00005, .012],
            "inclination_deg": [control["leo_inclination_mean_deg"], 12., 0., 120.],
            "bstar": [1.2e-5, .8e-5, 1e-7, 5e-5],
        },
        "MEO": {
            "altitude_km": [17000., 4000., 8000., 26000.],
            "eccentricity": [.025, .020, .0001, .08],
            "inclination_deg": [55., 20., 10., 100.],
            "bstar": None,
        },
        "GEO_like": {
            "altitude_km": [35786., 300., 34500., 37000.],
            "eccentricity": [.005, .006, .00001, .025],
            "inclination_deg": [5., 5., 0., 25.],
            "bstar": None,
        },
    }


def _make_spec(index: int, split: str, control: dict, quantiles: dict,
               month: int, stress_type: str | None) -> dict:
    seed = CATALOG_SEED_BASE + index
    rng = np.random.default_rng(np.random.SeedSequence([seed, 11]))
    n, s = _integer(control["n_targets"]), _integer(control["n_stations"])
    network_mode = "global_two_hemispheres"
    if stress_type == "regional_network":
        network_mode = "regional_asia"
        anchors = np.sort(rng.choice(REGIONAL_ANCHORS, s, replace=False))
    else:
        north = np.flatnonzero(STATION_ANCHORS[:, 1] > 0)
        south = np.flatnonzero(STATION_ANCHORS[:, 1] < 0)
        chosen = np.array([rng.choice(north), rng.choice(south)], dtype=np.int16)
        remaining = np.setdiff1d(np.arange(len(STATION_ANCHORS)), chosen)
        anchors = np.sort(np.r_[chosen, rng.choice(remaining, s - 2, replace=False)])
    # Month strata are uniform. Start day is uniform among six-day intervals
    # fully contained in that month; time of day is fixed at midnight UTC.
    day = int(rng.integers(1, calendar.monthrange(2027, month)[1] - 5))
    start = datetime(2027, month, day, tzinfo=timezone.utc)
    leo, meo_rest = control["leo_fraction"], control["meo_remaining_fraction"]
    high, medium_rest = control["high_priority_fraction"], control["medium_remaining_fraction"]
    orbit = _orbit_parameters(control)
    if stress_type == "leo_polar":
        orbit["LEO"]["inclination_deg"] = [95., 7., 75., 110.]
    return dict(
        catalog_id=f"v2-cat-{index:03d}", split=split, seed=seed,
        n_targets=n, n_stations=s, size_bin=size_bin(n),
        start_utc=start.isoformat(), sampling_version=SAMPLING_VERSION,
        distribution="nominal_gaussian" if stress_type is None else "stress",
        sampling_group="core" if stress_type is None else "stress", stress_type=stress_type,
        sampling_latents={k: float(v) for k, v in control.items()},
        sampling_quantiles={k: float(v) for k, v in quantiles.items()},
        family_probabilities=[leo, (1 - leo) * meo_rest, (1 - leo) * (1 - meo_rest)],
        priority_probabilities=[(1 - high) * (1 - medium_rest), (1 - high) * medium_rest, high],
        orbit_parameters=orbit, station_anchor_ids=[int(v) for v in anchors],
        network_mode=network_mode,
        station_jitter={"longitude_deg": [0., .02, -.06, .06],
                        "latitude_deg": [0., .02, -.06, .06],
                        "height_m": [0., 20., -60., 60.]},
    )


def specs() -> list[dict]:
    """Return the frozen 540 core + 60 stress independent catalog specifications."""
    rows = []
    for split_index, (split, n) in enumerate(CORE_COUNTS.items()):
        rng = np.random.default_rng(np.random.SeedSequence([CONFIG_SEED, split_index]))
        quantiles, controls = {}, {}
        for key, params in CORE_CONTROLS.items():
            u = (np.arange(n, dtype=float) + rng.random(n)) / n
            rng.shuffle(u)
            quantiles[key] = u
            controls[key] = _ppf(u, params)
        months = np.tile(np.arange(1, 13), n // 12)
        rng.shuffle(months)
        for j in range(n):
            rows.append(_make_spec(len(rows), split,
                                   {k: values[j] for k, values in controls.items()},
                                   {k: values[j] for k, values in quantiles.items()},
                                   int(months[j]), None))
    for split_index, split in enumerate(("validation", "test")):
        for type_index, stress_type in enumerate(STRESS_TYPES):
            rng = np.random.default_rng(np.random.SeedSequence([CONFIG_SEED, 100 + split_index, type_index]))
            for j in range(5):
                control = {k: float(_draw(rng, p, 1)[0]) for k, p in CORE_CONTROLS.items()}
                if stress_type == "high_load":
                    control["n_targets"] = float(_draw(rng, (13500., 1200., 11000., 16000.), 1)[0])
                    control["n_stations"] = float(_draw(rng, (5., 1., 4., 7.), 1)[0])
                elif stress_type == "low_load":
                    control["n_targets"] = float(_draw(rng, (2000., 600., 1000., 3500.), 1)[0])
                    control["n_stations"] = float(_draw(rng, (14., 1., 12., 16.), 1)[0])
                elif stress_type == "leo_polar":
                    control["leo_fraction"] = float(_draw(rng, (.96, .015, .93, .99), 1)[0])
                    control["leo_inclination_mean_deg"] = 95.
                elif stress_type == "high_orbit":
                    control["leo_fraction"] = float(_draw(rng, (.30, .05, .20, .40), 1)[0])
                    control["meo_remaining_fraction"] = float(_draw(rng, (.45, .10, .20, .70), 1)[0])
                elif stress_type == "high_priority":
                    control["high_priority_fraction"] = float(_draw(rng, (.72, .07, .60, .90), 1)[0])
                else:
                    control["n_targets"] = float(_draw(rng, (9000., 1500., 6000., 12000.), 1)[0])
                    control["n_stations"] = float(_draw(rng, (6., 1., 4., 7.), 1)[0])
                month = 6 if stress_type == "regional_network" else 1 + (type_index * 2 + j * 3 + split_index) % 12
                rows.append(_make_spec(len(rows), split, control, {}, month, stress_type))
    return rows


def sampling_report(rows: list[dict]) -> dict:
    """Audit the frozen design, including ECDF error against truncated normals.

    This is a deterministic distribution-contract check, not an IID KS p-value
    or proof that the synthetic population matches real space-object catalogs.
    """
    errors = []
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        return dict(passed=False, errors=["specs must be a list of objects"])
    try:
        fingerprint = specs_sha256(rows)
    except (ValueError, TypeError):
        return dict(passed=False, errors=["specs must be finite JSON values"])
    if len(rows) != 600:
        errors.append("expected 600 independent catalogs")
    for key in ("catalog_id", "seed"):
        kind = str if key == "catalog_id" else int
        if any(type(r.get(key)) is not kind for r in rows):
            errors.append(f"invalid or missing {key} type")
        elif len({r[key] for r in rows}) != len(rows):
            errors.append(f"duplicate or missing {key}")
    expected_splits = {"train": 420, "validation": 90, "test": 90}
    actual_splits = dict(Counter(r.get("split") for r in rows))
    if actual_splits != expected_splits:
        errors.append("split counts differ from 420/90/90")
    core = [r for r in rows if r.get("distribution") == "nominal_gaussian"]
    stress = [r for r in rows if r.get("distribution") == "stress"]
    if len(core) != 540 or len(stress) != 60:
        errors.append("expected 540 nominal and 60 separately labeled stress catalogs")
    for r in rows:
        label = str(r.get("catalog_id"))
        try:
            n, s = r["n_targets"], r["n_stations"]
            if type(n) is not int or not 1000 <= n <= 16000 or type(s) is not int or not 4 <= s <= 16:
                errors.append(f"{label}: target/station bounds or integer types invalid")
            if r["size_bin"] != size_bin(n):
                errors.append(f"{label}: wrong size bin")
            if n != _integer(r["sampling_latents"]["n_targets"]) or s != _integer(r["sampling_latents"]["n_stations"]):
                errors.append(f"{label}: rounded counts do not match recorded latents")
            if r["sampling_version"] != SAMPLING_VERSION:
                errors.append(f"{label}: sampling version differs")
            if type(r["seed"]) is not int or not CATALOG_SEED_BASE <= r["seed"] < CATALOG_SEED_BASE + 600:
                errors.append(f"{label}: seed outside the new v2 domain")
            start = datetime.fromisoformat(r["start_utc"])
            if start.tzinfo is None or start.utcoffset() != timedelta(0) or start.year != 2027 or (start + timedelta(days=6)).year != 2027:
                errors.append(f"{label}: invalid 2027 UTC planning interval")
            anchors = r["station_anchor_ids"]
            if len(anchors) != s or len(set(anchors)) != s or any(type(a) is not int or not 0 <= a < len(STATION_ANCHORS) for a in anchors):
                errors.append(f"{label}: invalid station anchor selection")
            for key in ("family_probabilities", "priority_probabilities"):
                p = np.asarray(r[key], dtype=float)
                if p.shape != (3,) or not np.isfinite(p).all() or (p <= 0).any() or not np.isclose(p.sum(), 1., rtol=0, atol=1e-12):
                    errors.append(f"{label}: invalid {key}")
            if r["distribution"] == "nominal_gaussian" and (r["sampling_group"] != "core" or r["stress_type"] is not None):
                errors.append(f"{label}: inconsistent nominal labels")
            if r["distribution"] == "stress" and (r["sampling_group"] != "stress" or r["stress_type"] not in STRESS_TYPES or r["split"] == "train"):
                errors.append(f"{label}: inconsistent stress labels")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append(f"{label}: malformed specification ({type(exc).__name__})")
    distribution_checks, coverage = {}, {}
    for split, expected_n in CORE_COUNTS.items():
        subset = [r for r in core if r.get("split") == split]
        if len(subset) != expected_n:
            errors.append(f"{split}: incorrect nominal catalog count")
        distribution_checks[split] = {}
        try:
            month_counts = Counter(datetime.fromisoformat(r["start_utc"]).month for r in subset)
            if month_counts != {month: expected_n // 12 for month in range(1, 13)}:
                errors.append(f"{split}: month strata differ")
            coverage[split] = dict(core_catalogs=len(subset),
                                   size_bins=dict(Counter(r["size_bin"] for r in subset)),
                                   month_counts={str(k): month_counts[k] for k in sorted(month_counts)},
                                   targets_ge_10000=sum(r["n_targets"] >= 10000 for r in subset),
                                   targets_le_3000=sum(r["n_targets"] <= 3000 for r in subset),
                                   total_targets=sum(r["n_targets"] for r in subset))
            if split != "train" and coverage[split]["targets_ge_10000"] < 9:
                errors.append(f"{split}: fewer than nine nominal large catalogs")
            for key, params in CORE_CONTROLS.items():
                values = np.array([r["sampling_latents"][key] for r in subset], dtype=float)
                u = np.array([_cdf(x, params) for x in values])
                saved = np.array([r["sampling_quantiles"][key] for r in subset], dtype=float)
                n = len(u)
                ordered = np.sort(u)
                d = float(max(np.max(np.arange(1, n + 1) / n - ordered),
                              np.max(ordered - np.arange(n) / n)))
                one_per_stratum = np.array_equal(np.sort(np.floor(u * n).astype(int)), np.arange(n))
                passed = bool(np.isfinite(values).all() and (values >= params[2]).all()
                              and (values <= params[3]).all() and np.allclose(u, saved, rtol=0, atol=CDF_TOLERANCE)
                              and one_per_stratum and d <= 1 / n + CDF_TOLERANCE)
                distribution_checks[split][key] = dict(n=n, params=list(params), max_cdf_error=d,
                                                       limit=1 / n + CDF_TOLERANCE,
                                                       one_per_quantile_stratum=bool(one_per_stratum), passed=passed)
                if not passed:
                    errors.append(f"{split}/{key}: truncated-normal sampling contract failed")
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f"{split}: malformed nominal controls ({type(exc).__name__})")
    stress_counts = {}
    for split in ("validation", "test"):
        actual = Counter(r.get("stress_type") for r in stress if r.get("split") == split)
        stress_counts[split] = dict(actual)
        if actual != {name: 5 for name in STRESS_TYPES}:
            errors.append(f"{split}: stress-type counts differ")
    return dict(passed=not errors, errors=errors, sampling_version=SAMPLING_VERSION,
                specs_sha256=fingerprint,
                counts=dict(catalogs=len(rows), core=len(core), stress=len(stress), by_split=actual_splits),
                core_distribution_checks=distribution_checks, coverage=coverage,
                stress_counts=stress_counts,
                interpretation="stratified truncated-normal scalar controls in core only; not IID normality, orbital-population fit or a Gaussian mixture claim for all scenes")


def generate_catalog(spec: dict) -> tuple[Dataset, dict[str, np.ndarray]]:
    """Generate published-TLE-compatible inputs; geometry remains a separate step."""
    if any(type(spec.get(k)) is not int for k in ("n_targets", "n_stations", "seed")):
        raise ValueError("catalog dimensions and seed must be integers")
    n, s, seed = int(spec["n_targets"]), int(spec["n_stations"]), int(spec["seed"])
    if not (1 <= n <= 16000 and 4 <= s <= 16):
        raise ValueError("catalog dimensions outside supported bounds")
    family_p = np.asarray(spec["family_probabilities"], dtype=float)
    priority_p = np.asarray(spec["priority_probabilities"], dtype=float)
    for p in (family_p, priority_p):
        if p.shape != (3,) or not np.isfinite(p).all() or (p <= 0).any() or not np.isclose(p.sum(), 1., rtol=0, atol=1e-12):
            raise ValueError("invalid categorical probabilities")
    orbit_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    station_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    priority_rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    family = orbit_rng.choice(3, n, p=family_p)
    alt, ecc, inc, bstar = (np.empty(n) for _ in range(4))
    for i, name in enumerate(FAMILIES):
        mask, params = family == i, spec["orbit_parameters"][name]
        count = int(mask.sum())
        alt[mask] = _draw(orbit_rng, params["altitude_km"], count)
        ecc[mask] = _draw(orbit_rng, params["eccentricity"], count)
        inc[mask] = _draw(orbit_rng, params["inclination_deg"], count)
        bstar[mask] = 0. if params["bstar"] is None else _draw(orbit_rng, params["bstar"], count)
    semimajor = 6378.135 + alt
    perigee = semimajor * (1 - ecc) - 6378.135
    if not np.isfinite(perigee).all() or (perigee < 450.).any() or (ecc >= 1).any() or (ecc < 0).any():
        raise ValueError("invalid synthetic orbit; no silent target replacement")
    raan, argp, anomaly = orbit_rng.uniform(0., 360., (3, n))
    motion = np.sqrt(398600.8 / semimajor ** 3) * 60.
    start = datetime.fromisoformat(spec["start_utc"])
    if start.tzinfo is None or start.utcoffset() != timedelta(0):
        raise ValueError("start_utc must have UTC timezone")
    epoch = start + timedelta(days=3)
    jd, fr = jday(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second)
    nums = np.arange(10001, 10001 + n, dtype=np.int64)
    line1, line2 = [], []
    for i in range(n):
        sat = Satrec()
        sat.sgp4init(WGS72, "i", int(nums[i]), jd + fr - 2433281.5, float(bstar[i]), 0., 0.,
                     float(ecc[i]), np.radians(argp[i]), np.radians(inc[i]), np.radians(anomaly[i]),
                     float(motion[i]), np.radians(raan[i]))
        a, b = export_tle(sat)
        if not (tle_checksum_ok(a) and tle_checksum_ok(b)):
            raise ValueError("exported TLE checksum failure")
        back = Satrec.twoline2rv(a, b)
        error, position, velocity = back.sgp4(jd, fr)
        if error or not np.isfinite([*position, *velocity]).all():
            raise ValueError(f"exported TLE propagation failed for target {i}: {error}")
        line1.append(a)
        line2.append(b)
    anchors = np.asarray(spec["station_anchor_ids"], dtype=np.int16)
    if anchors.shape != (s,) or len(np.unique(anchors)) != s or (anchors < 0).any() or (anchors >= len(STATION_ANCHORS)).any():
        raise ValueError("invalid station anchors")
    jitter = spec["station_jitter"]
    stations = STATION_ANCHORS[anchors].copy()
    for axis, key in enumerate(("longitude_deg", "latitude_deg", "height_m")):
        stations[:, axis] += _draw(station_rng, jitter[key], s)
    priority = priority_rng.choice([2., 3., 5.], n, p=priority_p)
    labels = [{2.: "低", 3.: "中", 5.: "高"}[v] for v in priority]
    ds = Dataset([f"SYN-{spec['catalog_id']}-{i:05d}" for i in range(n)], nums, line1, line2,
                 priority, labels, stations[:, 0], stations[:, 1], stations[:, 2], [])
    table = dict(target_id=np.arange(n, dtype=np.int32), satnum=nums, family=family.astype(np.int8),
                 semimajor_km=semimajor, eccentricity=ecc, inclination_deg=inc, raan_deg=raan,
                 arg_perigee_deg=argp, mean_anomaly_deg=anomaly, bstar=bstar,
                 mean_motion_rad_min=motion, mean_element_perigee_km=perigee,
                 line1=np.array(line1), line2=np.array(line2), names=np.array(ds.names),
                 base_priority=priority.astype(np.float32), stations=stations,
                 station_anchor_id=anchors)
    return ds, table
