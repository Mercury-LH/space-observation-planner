"""SGP4 visibility with numerical event refinement, bounded memory and caching.

TLE propagation uses WGS72/TEME. Station geodetic coordinates use WGS84.
TEME is rotated to a pseudo Earth-fixed frame using Vallado GMST with UT1=UTC,
without polar motion, atmospheric refraction or an illumination/night filter.
This explicit development approximation is not an IERS-accurate ITRS model.

The scan brackets elevation extrema as well as threshold crossings. It is not a
proof that an arbitrary function has no hidden events between scan samples.
Use half-step convergence audits and independent plan geometry validation.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from typing import Callable

import numpy as np
import sgp4
from sgp4.api import Satrec, SatrecArray, jday, accelerated

from .types import Station, Target, Window

GEOMETRY_VERSION = "events-v1-ut1utc-wgs72-wgs84"
EARTH_ROTATION_RAD_S = 7.29211514670698e-5


def gmst_radians(jd: np.ndarray) -> np.ndarray:
    """Vallado's GMST polynomial, assuming the supplied Julian date is UT1."""
    centuries = (jd - 2451545.0) / 36525.0
    seconds = (
        -6.2e-6 * centuries ** 3 + 0.093104 * centuries ** 2
        + (876600.0 * 3600 + 8640184.812866) * centuries + 67310.54841
    )
    return np.remainder(seconds * (math.pi / 43200.0), 2 * math.pi)


def station_vectors(station: Station) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 ECEF km and geodetic local vertical unit vector."""
    lat, lon = np.radians([station.latitude_deg, station.longitude_deg])
    a, e2 = 6378.137, 6.6943799901413165e-3
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    h = station.altitude_m / 1000
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    position = np.array([
        (n + h) * up[0], (n + h) * up[1], (n * (1 - e2) + h) * up[2],
    ])
    return position, up


def earth_fixed(position, velocity, angle):
    """Rotate TEME positions/velocities into pseudo Earth-fixed coordinates."""
    c, s = np.cos(angle), np.sin(angle)
    result = np.empty_like(position)
    result[..., 0] = c * position[..., 0] + s * position[..., 1]
    result[..., 1] = -s * position[..., 0] + c * position[..., 1]
    result[..., 2] = position[..., 2]
    rate = np.empty_like(velocity)
    rate[..., 0] = c * velocity[..., 0] + s * velocity[..., 1] + EARTH_ROTATION_RAD_S * result[..., 1]
    rate[..., 1] = -s * velocity[..., 0] + c * velocity[..., 1] - EARTH_ROTATION_RAD_S * result[..., 0]
    rate[..., 2] = velocity[..., 2]
    return result, rate


def elevation_sine_and_rate(position, velocity, station_position, up):
    relative = position - station_position
    squared = np.sum(relative * relative, axis=-1)
    distance = np.sqrt(squared)
    vertical = np.sum(relative * up, axis=-1)
    radial_dot = np.sum(relative * velocity, axis=-1)
    value = vertical / distance
    rate = np.sum(velocity * up, axis=-1) / distance - vertical * radial_dot / (squared * distance)
    return value, rate


class Geometry:
    def __init__(self, targets: list[Target], stations: list[Station], start_utc: datetime):
        if start_utc.tzinfo is None or start_utc.utcoffset() is None:
            raise ValueError("start_utc must include an explicit timezone")
        self.start_utc = start_utc.astimezone(timezone.utc)
        self.targets = {target.target_id: target for target in targets}
        self.stations = {station.station_id: station for station in stations}
        if len(self.targets) != len(targets) or len(self.stations) != len(stations):
            raise ValueError("Target and station identifiers must be unique")
        self.satellites = {
            target.target_id: Satrec.twoline2rv(target.line1, target.line2) for target in targets
        }
        self.vectors = {s.station_id: station_vectors(s) for s in stations}
        dt = self.start_utc
        self.jd, self.fraction = jday(
            dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6,
        )

    def dates(self, seconds):
        seconds = np.atleast_1d(np.asarray(seconds, dtype=float))
        if not np.all(np.isfinite(seconds)):
            raise ValueError("Nonfinite propagation time")
        return np.full(seconds.shape, self.jd), self.fraction + seconds / 86400

    def state(self, target_id, seconds):
        jd, fraction = self.dates(seconds)
        error, position, velocity = self.satellites[target_id].sgp4_array(jd, fraction)
        if np.any(error):
            codes, counts = np.unique(error[error != 0], return_counts=True)
            raise ValueError(f"SGP4 error target={target_id}: {dict(zip(codes.tolist(), counts.tolist()))}")
        result = earth_fixed(position, velocity, gmst_radians(jd + fraction))
        if not all(np.all(np.isfinite(array)) for array in result):
            raise ValueError(f"Nonfinite propagated state for {target_id}")
        return result

    def sine_and_rate(self, target_id, station_id, seconds):
        position, velocity = self.state(target_id, seconds)
        values = elevation_sine_and_rate(position, velocity, *self.vectors[station_id])
        if not all(np.all(np.isfinite(array)) for array in values):
            raise ValueError(f"Nonfinite topocentric geometry for {target_id} at {station_id}")
        return values

    def elevation(self, target_id, station_id, seconds):
        values, _ = self.sine_and_rate(target_id, station_id, seconds)
        result = np.degrees(np.arcsin(np.clip(values, -1, 1)))
        return float(result[0]) if np.ndim(seconds) == 0 else result


def _bisect_many(evaluate, left, right, left_value, component, level, tolerance):
    left, right = np.array(left, dtype=float), np.array(right, dtype=float)
    left_value = np.array(left_value, dtype=float) - level
    if not len(left):
        return left
    count = max(0, math.ceil(math.log2(max(float(np.max(right - left)) / tolerance, 1))))
    for _ in range(count):
        mid = (left + right) / 2
        mid_value = np.asarray(evaluate(mid)[component]) - level
        if not np.all(np.isfinite(mid_value)):
            raise ValueError("Nonfinite geometry while refining an event")
        same = np.signbit(left_value) == np.signbit(mid_value)
        left = np.where(same, mid, left)
        left_value = np.where(same, mid_value, left_value)
        right = np.where(same, right, mid)
    return (left + right) / 2


def extract_intervals(
    times: np.ndarray,
    sine: np.ndarray,
    rates: np.ndarray,
    evaluate: Callable,
    lower_deg: float = 10.0,
    upper_deg: float = 85.0,
    tolerance_s: float = 0.01,
) -> list[tuple[float, float]]:
    """Refine all bracketed extrema/thresholds; retain short physical windows.

    Threshold event edges are eroded by one tolerance to guarantee inward
    rounding within the root bracket. Chunk/horizon endpoints are not eroded.
    No candidate starts or observation durations are generated here.
    """
    times, sine, rates = map(np.asarray, (times, sine, rates))
    if len(times) < 2:
        return []
    if not np.all(np.isfinite(sine)) or not np.all(np.isfinite(rates)):
        raise ValueError("extract_intervals requires finite valid-state segments")
    if not 0 < tolerance_s < np.min(np.diff(times)):
        raise ValueError("Event tolerance must be positive and smaller than scan spacing")
    lower, upper = np.sin(np.radians([lower_deg, upper_deg]))
    # A sub-grid excursion across either threshold can be detected even when
    # both elevation endpoints lie on the same side, via a bracketed extremum.
    extremum_indices = np.flatnonzero(rates[:-1] * rates[1:] < 0)
    extrema = _bisect_many(
        evaluate, times[extremum_indices], times[extremum_indices + 1],
        rates[extremum_indices], 1, 0.0, tolerance_s,
    )
    if len(extrema):
        extra_values = np.asarray(evaluate(extrema)[0])
        if not np.all(np.isfinite(extra_values)):
            raise ValueError("Nonfinite geometry at a refined extremum")
        order = np.argsort(np.concatenate([times, extrema]))
        event_times = np.concatenate([times, extrema])[order]
        values = np.concatenate([sine, extra_values])[order]
    else:
        event_times, values = times, sine
    roots = []
    for threshold in (lower, upper):
        signed = values - threshold
        crossings = np.flatnonzero(signed[:-1] * signed[1:] < 0)
        found = _bisect_many(
            evaluate, event_times[crossings], event_times[crossings + 1],
            values[crossings], 0, threshold, tolerance_s,
        )
        roots.extend(found.tolist())
        roots.extend(event_times[np.flatnonzero(signed == 0)].tolist())
    internal = sorted({float(t) for t in roots if times[0] < t < times[-1]})
    bounds = np.array([float(times[0]), *internal, float(times[-1])])
    mids = (bounds[:-1] + bounds[1:]) / 2
    middle_values = np.asarray(evaluate(mids)[0])
    if not np.all(np.isfinite(middle_values)):
        raise ValueError("Nonfinite geometry at a visibility interval midpoint")
    allowed = (middle_values >= lower) & (middle_values <= upper)
    intervals = []
    for index in np.flatnonzero(allowed):
        start, end = bounds[index], bounds[index + 1]
        if index:
            start += tolerance_s
        if index < len(bounds) - 2:
            end -= tolerance_s
        if end > start:
            intervals.append((float(start), float(end)))
    return intervals


def _fingerprint(geometry, config):
    payload = {
        "version": GEOMETRY_VERSION, "sgp4": sgp4.__version__,
        "start": geometry.start_utc.isoformat(), "config": config,
        "targets": [(t.target_id, t.line1, t.line2) for t in geometry.targets.values()],
        "stations": [asdict(s) for s in geometry.stations.values()],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _window_worker(targets, stations, start_utc, config):
    geometry = Geometry(targets, stations, start_utc)
    windows, stats = compute_windows(geometry, **config)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stats["worker_peak_rss_mib"] = rss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    return windows, stats


def compute_windows(
    geometry: Geometry,
    horizon_s: float,
    scan_step_s: float = 30.0,
    target_chunk: int = 32,
    time_chunk_s: float = 21600.0,
    root_tolerance_s: float = 0.01,
    cache_dir: Path | None = None,
    progress: Callable | None = None,
    workers: int = 1,
) -> tuple[list[Window], dict]:
    """Chunked cross-station reuse; cache keyed by all physical inputs/settings.

    Nonzero SGP4 samples fail closed: intervals touching such a sample are
    excluded and counts/affected targets are reported, not silently accepted.
    A failure during event refinement aborts the computation.
    """
    if not math.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError("horizon_s must be finite and positive")
    if (
        not math.isfinite(scan_step_s) or not math.isfinite(time_chunk_s)
        or not 0 < root_tolerance_s < scan_step_s <= 120
        or time_chunk_s < scan_step_s or target_chunk < 1
        or not isinstance(workers, int) or not 1 <= workers <= 32
    ):
        raise ValueError("Invalid scan/chunk/tolerance setting; scan step must be <=120 s")
    started = time.perf_counter()
    config = dict(
        horizon_s=horizon_s, scan_step_s=scan_step_s, target_chunk=target_chunk,
        time_chunk_s=time_chunk_s, root_tolerance_s=root_tolerance_s,
    )
    key = _fingerprint(geometry, config)
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"windows-{key}.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as stored:
                station_ids = stored["station_ids"].tolist()
                windows = [
                    Window(int(row[0]), station_ids[int(row[1])], float(row[2]), float(row[3]))
                    for row in stored["windows"]
                ]
                stats = json.loads(str(stored["stats"]))
                if stats.get("cache_key") != key:
                    raise ValueError("Window cache metadata does not match its fingerprint")
            if any(
                w.target_id not in geometry.targets or w.station_id not in geometry.stations
                or not math.isfinite(w.start) or not math.isfinite(w.end)
                or not 0 <= w.start < w.end <= horizon_s
                for w in windows
            ):
                raise ValueError("Invalid window record in cache")
            stats.update(cache_hit=True, wall_time_s=time.perf_counter() - started)
            return windows, stats
    targets = list(geometry.targets)
    stations = list(geometry.stations.values())
    windows = []
    errors: dict[str, int] = {}
    error_targets = set()
    propagated_samples = 0
    worker_peak_rss_mib = 0.0
    if workers > 1 and len(targets) > target_chunk:
        # Each job owns a bounded target group across the full horizon so it can
        # merge adjacent time slabs. Orbit state grids never cross processes.
        group_size = max(target_chunk, 128)
        groups = [targets[i:i + group_size] for i in range(0, len(targets), group_size)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _window_worker, [geometry.targets[t] for t in group], stations,
                    geometry.start_utc, config,
                )
                for group in groups
            ]
            for completed, future in enumerate(as_completed(futures), 1):
                result, worker_stats = future.result()
                windows.extend(result)
                propagated_samples += worker_stats["propagated_grid_samples"]
                error_targets.update(worker_stats["propagation_error_targets"])
                worker_peak_rss_mib = max(worker_peak_rss_mib, worker_stats["worker_peak_rss_mib"])
                for code, count in worker_stats["propagation_error_samples"].items():
                    errors[code] = errors.get(code, 0) + count
                if progress:
                    progress({"completed_target_groups": completed, "total_target_groups": len(groups)})
        # The ordinary loop is skipped; serialization and cache logic are shared.
        sequential_targets = []
    else:
        sequential_targets = targets
    n_times = math.ceil(horizon_s / time_chunk_s)
    total_chunks = n_times * math.ceil(len(targets) / target_chunk)
    completed = 0
    for first in range(0, len(sequential_targets), target_chunk):
        ids = sequential_targets[first:first + target_chunk]
        satellites = SatrecArray([geometry.satellites[target_id] for target_id in ids])
        grouped = {(target, station.station_id): [] for target in ids for station in stations}
        for slab in range(n_times):
            begin, end = slab * time_chunk_s, min((slab + 1) * time_chunk_s, horizon_s)
            intervals = math.ceil((end - begin) / scan_step_s)
            times = np.linspace(begin, end, intervals + 1)
            jd, fraction = geometry.dates(times)
            codes, position, velocity = satellites.sgp4(jd, fraction)
            propagated_samples += codes.size
            position, velocity = earth_fixed(position, velocity, gmst_radians(jd + fraction))
            valid = (codes == 0) & np.all(np.isfinite(position), axis=-1) & np.all(np.isfinite(velocity), axis=-1)
            if not np.all(valid):
                for row in np.flatnonzero(np.any(~valid, axis=1)):
                    error_targets.add(ids[row])
                unique, counts = np.unique(codes[~valid], return_counts=True)
                for code, count in zip(unique, counts):
                    key_code = str(code) if code else "nonfinite"
                    errors[key_code] = errors.get(key_code, 0) + int(count)
            for station in stations:
                sine, rates = elevation_sine_and_rate(
                    position, velocity, *geometry.vectors[station.station_id],
                )
                for row, target_id in enumerate(ids):
                    def evaluate(t, target_id=target_id, station_id=station.station_id):
                        return geometry.sine_and_rate(target_id, station_id, t)
                    # Contiguous valid samples; a one-sample fragment cannot
                    # certify any interval and therefore contributes no window.
                    masks = valid[row]
                    edges = np.diff(np.r_[False, masks, False].astype(int))
                    for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                        if right - left < 2:
                            continue
                        pieces = extract_intervals(
                            times[left:right], sine[row, left:right], rates[row, left:right],
                            evaluate, station.min_elevation_deg, station.max_elevation_deg,
                            root_tolerance_s,
                        )
                        output = grouped[(target_id, station.station_id)]
                        for start, finish in pieces:
                            if output and abs(output[-1][1] - start) < 1e-8:
                                output[-1] = (output[-1][0], finish)
                            else:
                                output.append((start, finish))
            completed += 1
            if progress:
                progress({"completed_chunks": completed, "total_chunks": total_chunks})
        for (target_id, station_id), pieces in grouped.items():
            windows.extend(Window(target_id, station_id, a, b) for a, b in pieces)
    windows.sort(key=lambda w: (w.station_id, w.start, w.target_id, w.end))
    stats = {
        **config, "geometry_version": GEOMETRY_VERSION, "cache_key": key,
        "cache_hit": False, "sgp4_accelerated": accelerated,
        "generation_workers": workers, "maximum_worker_peak_rss_mib": worker_peak_rss_mib,
        "targets": len(targets), "stations": len(stations), "windows": len(windows),
        "propagated_grid_samples": propagated_samples,
        "propagation_error_samples": errors, "propagation_error_targets": sorted(error_targets),
        "window_targets": len({w.target_id for w in windows}),
        "schedulable_120s_targets": len({w.target_id for w in windows if w.end - w.start >= 120}),
        "wall_time_s": time.perf_counter() - started,
        "scope": "refined numerical events; no formal continuous-domain completeness proof",
        "coordinate_model": "SGP4 WGS72; WGS84 geodetic stations; GMST UT1=UTC; no polar motion/refraction",
        "night_filter": None,
        "event_inward_margin_s": root_tolerance_s,
        "error_policy": "exclude intervals touching invalid scan states; abort invalid refined events",
    }
    if cache_path is not None:
        station_ids = [s.station_id for s in stations]
        station_indices = {s: i for i, s in enumerate(station_ids)}
        rows = np.asarray(
            [(w.target_id, station_indices[w.station_id], w.start, w.end) for w in windows],
            dtype=float,
        ).reshape(-1, 4)
        # Different processes may request the same key concurrently. Unique
        # temporary files plus atomic replace avoid colliding writers.
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.savez_compressed(stream, windows=rows, station_ids=np.asarray(station_ids), stats=json.dumps(stats))
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)
    return windows, stats
