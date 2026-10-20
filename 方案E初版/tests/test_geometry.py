from datetime import datetime, timezone

import numpy as np
import pytest
from sgp4.propagation import gstime

from space_planner.geometry import (
    Geometry, compute_windows, extract_intervals, gmst_radians, station_vectors,
)
from space_planner.types import Station, Target


# Public example from python-sgp4's package documentation; no private TLE.
ISS = Target(
    25544,
    "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991",
    "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482",
    5, datetime(2019, 12, 9, tzinfo=timezone.utc),
)


def parabola(peak, center, curvature):
    def evaluate(t):
        t = np.asarray(t)
        angle = peak - curvature * (t - center) ** 2
        derivative = -2 * curvature * (t - center)
        return np.sin(np.radians(angle)), np.cos(np.radians(angle)) * np.radians(derivative)
    return evaluate


def test_gmst_and_wgs84_station():
    dates = np.array([2451545.0, 2458826.5, 2461256.0])
    assert np.max(np.abs(gmst_radians(dates) - np.array([gstime(jd) for jd in dates]))) < 1e-10
    position, up = station_vectors(Station("equator", 0, 0, 100))
    assert position == pytest.approx([6378.237, 0, 0])
    assert up == pytest.approx([1, 0, 0])


def test_subscan_pass_recovered_through_extremum():
    times = np.array([0.0, 30.0])
    evaluate = parabola(11, 15, 0.02)
    values, rates = evaluate(times)
    assert np.all(values < np.sin(np.radians(10)))
    intervals = extract_intervals(times, values, rates, evaluate)
    assert len(intervals) == 1
    expected = np.sqrt(1 / 0.02)
    assert intervals[0][0] == pytest.approx(15 - expected, abs=0.02)
    assert intervals[0][1] == pytest.approx(15 + expected, abs=0.02)


def test_upper_elevation_cap_splits_hidden_peak():
    times = np.array([0.0, 30.0])
    evaluate = parabola(88, 15, 0.02)
    values, rates = evaluate(times)
    assert np.all(values < np.sin(np.radians(85)))
    intervals = extract_intervals(times, values, rates, evaluate)
    assert len(intervals) == 2
    assert intervals[0][1] < 15 < intervals[1][0]
    for start, end in intervals:
        values, _ = evaluate(np.linspace(start, end, 100))
        assert np.all(values <= np.sin(np.radians(85)))


def test_no_short_window_pruning_and_horizon_clipping():
    times = np.array([0.0, 10.0, 20.0])
    evaluate = parabola(10.02, 10, 0.02)
    intervals = extract_intervals(times, *evaluate(times), evaluate)
    assert len(intervals) == 1
    assert 1.9 < intervals[0][1] - intervals[0][0] < 2.1
    constant = lambda t: (np.full(np.shape(t), 0.5), np.zeros(np.shape(t)))
    assert extract_intervals(times, *constant(times), constant) == [(0, 20)]


def test_gridpoint_threshold_and_invalid_input():
    times = np.array([0.0, 10.0, 20.0])
    def evaluate(t):
        return np.sin(np.radians(t)), np.cos(np.radians(t)) * np.pi / 180
    intervals = extract_intervals(times, *evaluate(times), evaluate)
    assert intervals == [(10.01, 20.0)]
    with pytest.raises(ValueError, match="finite"):
        extract_intervals(times, np.array([0, np.nan, 1]), np.ones(3), evaluate)
    with pytest.raises(ValueError, match="timezone"):
        Geometry([ISS], [], datetime(2019, 12, 9))


def test_refined_nonfinite_values_fail_closed():
    times = np.array([0.0, 30.0])
    def broken(t):
        return np.full(np.shape(t), np.nan), np.zeros(np.shape(t))
    with pytest.raises(ValueError, match="midpoint"):
        extract_intervals(times, np.array([0.5, 0.5]), np.zeros(2), broken)
    def broken_extremum(t):
        return np.full(np.shape(t), np.nan), 15 - np.asarray(t)
    with pytest.raises(ValueError, match="extremum"):
        extract_intervals(times, np.array([0.5, 0.5]), np.array([15, -15]), broken_extremum)


def test_chunk_merge_cache_key_and_scan_convergence(tmp_path):
    station = Station("S1", 40.0, -75.0)
    geometry = Geometry([ISS], [station], ISS.epoch)
    kwargs = dict(horizon_s=86400, scan_step_s=30, target_chunk=1, time_chunk_s=21600)
    first, stats = compute_windows(geometry, **kwargs, cache_dir=tmp_path)
    cached, warm = compute_windows(geometry, **kwargs, cache_dir=tmp_path)
    fine, fine_stats = compute_windows(
        geometry, horizon_s=86400, scan_step_s=15, target_chunk=1, time_chunk_s=86400,
    )
    assert first and first == cached
    assert not stats["cache_hit"] and warm["cache_hit"]
    assert len(first) == len(fine)
    assert max(abs(a.start - b.start) for a, b in zip(first, fine)) < 0.02
    assert max(abs(a.end - b.end) for a, b in zip(first, fine)) < 0.02
    assert stats["cache_key"] != fine_stats["cache_key"]
    shifted = Geometry([ISS], [Station("S1", 41.0, -75.0)], ISS.epoch)
    _, changed = compute_windows(shifted, **kwargs, cache_dir=tmp_path)
    assert not changed["cache_hit"] and changed["cache_key"] != stats["cache_key"]


def test_independent_skyfield_coordinates():
    from skyfield.api import EarthSatellite, load, wgs84
    ts = load.timescale(builtin=True)
    station = Station("S1", 40, -75, 123)
    geometry = Geometry([ISS], [station], ISS.epoch)
    seconds = np.arange(0, 86400, 120.0)
    t = ts.utc(2019, 12, 9, 0, 0, seconds)
    satellite = EarthSatellite(ISS.line1, ISS.line2, ts=ts)
    location = wgs84.latlon(40, -75, elevation_m=123)
    altitude, _, _ = (satellite - location).at(t).altaz()
    ours = geometry.elevation(25544, "S1", seconds)
    # Skyfield includes Earth orientation/time scale corrections omitted here.
    # This tolerance verifies frame/sign/units, not operational orbit accuracy.
    assert np.max(np.abs(altitude.degrees - ours)) < 0.03
