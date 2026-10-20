"""Failure-path and process-equivalence checks using public/synthetic data."""

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from space_planner import geometry as module
from space_planner.geometry import Geometry, compute_windows
from space_planner.types import Station, Target


# Public python-sgp4 documentation example; no private competition input.
ISS = Target(
    25544,
    "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991",
    "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482",
    5, datetime(2019, 12, 9, tzinfo=timezone.utc),
)


@pytest.mark.parametrize("failure", ["error_code", "nonfinite_position", "nonfinite_velocity"])
def test_scalar_state_rejects_sgp4_and_nonfinite_failures(failure):
    geometry = Geometry([ISS], [Station("s", 0, 0)], ISS.epoch)

    class FailedSatellite:
        def sgp4_array(self, jd, fraction):
            codes = np.zeros(len(jd), dtype=np.uint8)
            position = np.full((len(jd), 3), 7000.0)
            velocity = np.zeros((len(jd), 3))
            if failure == "error_code":
                codes[0] = 6
            elif failure == "nonfinite_position":
                position[0, 0] = np.nan
            else:
                velocity[0, 0] = np.inf
            return codes, position, velocity

    geometry.satellites[ISS.target_id] = FailedSatellite()
    match = "SGP4 error" if failure == "error_code" else "Nonfinite"
    with np.errstate(invalid="ignore"):
        with pytest.raises(ValueError, match=match):
            geometry.state(ISS.target_id, [0, 30])


def _install_constant_scan(monkeypatch, geometry, failure):
    """Isolate validity segmentation from the separately tested frame rotation."""
    class ScanArray:
        def __init__(self, satellites):
            self.count = len(satellites)

        def sgp4(self, jd, fraction):
            times = (fraction - geometry.fraction) * 86400
            shape = (self.count, len(jd))
            codes = np.zeros(shape, dtype=np.uint8)
            position = np.broadcast_to(
                np.array([6878.137, 866.025403784, 0.0]), (*shape, 3),
            ).copy()
            velocity = np.zeros_like(position)
            bad = np.isclose(times, 60)
            if failure == "error_code":
                codes[:, bad] = 6
            elif failure == "nonfinite":
                position[:, bad, 0] = np.nan
            return codes, position, velocity

    monkeypatch.setattr(module, "SatrecArray", ScanArray)
    monkeypatch.setattr(module, "earth_fixed", lambda p, v, _: (p, v))
    monkeypatch.setattr(
        geometry, "sine_and_rate",
        lambda target, station, t: (
            np.full(np.shape(t), 0.5), np.zeros(np.shape(t)),
        ),
    )


@pytest.mark.parametrize("failure,code", [("error_code", "6"), ("nonfinite", "nonfinite")])
def test_invalid_scan_state_excludes_both_adjacent_intervals(monkeypatch, failure, code):
    geometry = Geometry([ISS], [Station("s", 0, 0)], ISS.epoch)
    _install_constant_scan(monkeypatch, geometry, failure)
    windows, stats = compute_windows(
        geometry, horizon_s=120, scan_step_s=30, time_chunk_s=120,
    )
    assert [(window.start, window.end) for window in windows] == [(0, 30), (90, 120)]
    assert stats["propagation_error_targets"] == [ISS.target_id]
    assert stats["propagation_error_samples"] == {code: 1}
    assert not any(window.start < 60 < window.end for window in windows)


def test_invalid_state_at_time_chunk_boundary_does_not_rejoin(monkeypatch):
    geometry = Geometry([ISS], [Station("s", 0, 0)], ISS.epoch)
    _install_constant_scan(monkeypatch, geometry, "error_code")
    windows, stats = compute_windows(
        geometry, horizon_s=120, scan_step_s=30, time_chunk_s=60,
    )
    assert [(window.start, window.end) for window in windows] == [(0, 30), (90, 120)]
    # The shared boundary is propagated in both chunks and both errors count.
    assert stats["propagation_error_samples"] == {"6": 2}


def test_refinement_propagation_error_aborts_instead_of_caching(monkeypatch, tmp_path):
    geometry = Geometry([ISS], [Station("s", 0, 0)], ISS.epoch)
    _install_constant_scan(monkeypatch, geometry, "none")

    def fail(*args):
        raise ValueError("SGP4 error during refinement")

    monkeypatch.setattr(geometry, "sine_and_rate", fail)
    with pytest.raises(ValueError, match="during refinement"):
        compute_windows(geometry, horizon_s=120, cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def _renumber(line, target_id):
    first = line[:2] + f"{target_id:05d}" + line[7:68]
    checksum = sum(int(c) if c.isdigit() else int(c == "-") for c in first) % 10
    return first + str(checksum)


def test_parallel_and_sequential_window_generation_match():
    # >128 targets ensures two process jobs, not a single job with two workers.
    # Aliases of the public orbit form a deterministic synthetic population.
    targets = [
        replace(
            ISS, target_id=10000 + index,
            line1=_renumber(ISS.line1, 10000 + index),
            line2=_renumber(ISS.line2, 10000 + index),
        )
        for index in range(129)
    ]
    geometry = Geometry(targets, [Station("s", 40, -75)], ISS.epoch)
    options = dict(horizon_s=86400, target_chunk=32, time_chunk_s=21600)
    sequential, first = compute_windows(geometry, workers=1, **options)
    parallel, second = compute_windows(geometry, workers=2, **options)
    assert sequential
    assert sequential == parallel
    assert first["cache_key"] == second["cache_key"]
    assert first["propagated_grid_samples"] == second["propagated_grid_samples"]
    assert first["propagation_error_samples"] == second["propagation_error_samples"] == {}
    assert second["generation_workers"] == 2
    assert second["maximum_worker_peak_rss_mib"] > 0


def test_geometry_version_changes_cache_key_but_priority_does_not(monkeypatch):
    station = Station("s", 40, -75)
    first = Geometry([ISS], [station], ISS.epoch)
    different_priority = Geometry([replace(ISS, priority=2)], [station], ISS.epoch)
    config = {"horizon_s": 120, "scan_step_s": 30}
    original_key = module._fingerprint(first, config)
    assert original_key == module._fingerprint(different_priority, config)
    monkeypatch.setattr(module, "GEOMETRY_VERSION", module.GEOMETRY_VERSION + "-test-change")
    assert original_key != module._fingerprint(first, config)
