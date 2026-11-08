"""Exact Skyfield cache equivalence, scene binding and safe fallback checks."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from obsplan.config import Config
from obsplan.data import load
from obsplan.schedule import Schedule
from obsplan import validator
from obsplan.validator import SkyfieldGeometryCache
from obsplan.validator_cache import _near_shadow_boundary


SCHEME = Path(__file__).resolve().parents[1]
PREVIEW = SCHEME / "datasets/synthetic-v0.1/preview/scenes/cat-000-v0"


@pytest.fixture
def case():
    metadata = json.loads((PREVIEW / "scene.json").read_text())
    cfg = Config(**metadata["config"]).with_(data_dir=str(PREVIEW / "input"))
    ds = load(cfg)
    with np.load(PREVIEW / "plan.npz", allow_pickle=False) as z:
        sch = Schedule(*(z[k][:32].copy() for k in ("station", "target", "start_s", "dur_s")))
    return cfg, ds, sch


@pytest.fixture
def real_case(case):
    cfg, ds, sch = case
    candidates = [SCHEME / "cache/de421.bsp",
                  SCHEME / "outputs/synthetic-v0.1-full/.cache/de421.bsp"]
    ephemeris = next((p for p in candidates if p.is_file()), None)
    if ephemeris is None:
        pytest.skip("Real Skyfield equivalence needs a locally cached de421.bsp")
    return cfg.with_(cache_dir=str(ephemeris.parent)), ds, sch


def assert_same(reference, cached, *, exact=False):
    for key in ("legal", "violations", "geometry_checked"):
        assert cached[key] == reference[key]
    np.testing.assert_array_equal(cached["invalid_mask"], reference["invalid_mask"])
    if exact:
        np.testing.assert_array_equal(cached["per_task"], reference["per_task"])
    else:
        np.testing.assert_allclose(cached["per_task"], reference["per_task"], rtol=0, atol=1e-9,
                                   equal_nan=True)
    for key in ("min_elev_margin", "max_elev_margin", "sun_margin"):
        if reference[key] is None:
            assert cached[key] is None
        else:
            assert cached[key] == pytest.approx(reference[key], rel=0, abs=1e-9)


def test_cache_reuses_exact_points_and_bounds_chunks(case):
    cfg, ds, _ = case
    cache = SkyfieldGeometryCache(cfg, ds, chunk_size=3)
    calls = []
    def times(seconds):
        calls.append(np.array(seconds))
        return SimpleNamespace(gast=seconds * 2,
                               M=np.broadcast_to(seconds, (3, 3, len(seconds))).copy())
    assert cache.prepare([np.array([0., 1., 2., 3.]), np.array([1., 3., 5.])], times)
    assert cache.stats()["computed_points"] == 5
    assert max(map(len, calls)) == 3
    assert cache.prepare([np.array([3., 5., 8.])], times)
    stats = cache.stats()
    assert stats["computed_points"] == 6 and stats["reused_points"] == 2
    out = cache.times(np.array([8., 0., 5.]), times)
    np.testing.assert_array_equal(out.gast, [16., 0., 10.])
    np.testing.assert_array_equal(out.M[0, 0], [8., 0., 5.])
    with pytest.raises(ValueError, match="not prepared"):
        cache.times(np.array([99.]), times)


@pytest.mark.parametrize("field,value", [("chunk_size", 0), ("chunk_size", True),
                                         ("max_points", -1), ("max_points", 1.5)])
def test_invalid_cache_budget_rejected(case, field, value):
    cfg, ds, _ = case
    with pytest.raises(ValueError):
        SkyfieldGeometryCache(cfg, ds, **{field: value})


@pytest.mark.parametrize("change", ["date", "station", "tle", "priority"])
def test_context_rejects_changed_configuration_or_dataset(case, change):
    cfg, ds, sch = case
    cache = SkyfieldGeometryCache(cfg, ds)
    changed = deepcopy(ds)
    if change == "date":
        cfg = cfg.with_(start_utc=(cfg.start_dt + timedelta(days=1)).isoformat())
    elif change == "station":
        changed.st_lat[0] += .01
    elif change == "tle":
        changed.line2[0] = changed.line2[0][:-1] + "X"
    else:
        changed.priority[0] += 1
    with pytest.raises(ValueError, match="different cfg or dataset"):
        validator.validate(cfg, changed, sch, geometry_cache=cache)


@pytest.mark.parametrize("mutation", ["id", "nan", "conflict"])
def test_structural_failures_do_not_prepare_or_load_geometry(case, monkeypatch, mutation):
    cfg, ds, sch = case
    cache = SkyfieldGeometryCache(cfg, ds)
    def forbidden(*args):
        pytest.fail("invalid plans must not load astronomy")
    monkeypatch.setattr(validator, "_sky", forbidden)
    if mutation == "id":
        sch.target[0] = ds.n_targets
    elif mutation == "nan":
        sch.start_s = sch.start_s.astype(float)
        sch.start_s[0] = np.nan
    else:
        sch.station[1] = sch.station[0]
        sch.start_s[1] = sch.start_s[0]
    result = validator.validate(cfg, ds, sch, geometry_cache=cache)
    assert not result["legal"] and not result["geometry_checked"]
    assert cache.stats()["computed_points"] == cache.stats()["optimized_calls"] == 0


@pytest.mark.parametrize("offset,station_shift", [(0, 0), (37, .3)])
def test_real_skyfield_equivalence_across_dates_and_sites(real_case, offset, station_shift):
    cfg, ds, sch = real_case
    cfg = cfg.with_(start_utc=(cfg.start_dt + timedelta(days=offset)).isoformat())
    ds.st_lat += station_shift
    ds.st_lon += station_shift
    reference = validator.validate(cfg, ds, sch)
    cache = SkyfieldGeometryCache(cfg, ds, chunk_size=257)
    cached = validator.validate(cfg, ds, sch, geometry_cache=cache)
    assert_same(reference, cached)
    assert cache.stats()["optimized_calls"] == 1


def test_grid_and_final_share_cache_but_reference_stays_untouched(real_case):
    cfg, ds, final = real_case
    with np.load(PREVIEW / "grid_plan.npz", allow_pickle=False) as z:
        grid = Schedule(*(z[k][:32].copy() for k in ("station", "target", "start_s", "dur_s")))
    cache = SkyfieldGeometryCache(cfg, ds)
    for sch in (grid, final):
        expected = validator.validate(cfg, ds, sch)
        assert_same(expected, validator.validate(cfg, ds, sch, geometry_cache=cache))
        assert_same(expected, validator.validate(cfg, ds, sch), exact=True)
    assert cache.stats()["reused_points"] > 0
    assert cache.stats()["optimized_calls"] == 2


@pytest.mark.parametrize("mode,reason", [
    ("fractional", "noninteger_or_subsecond_reference_backend"),
    ("subsecond", "noninteger_or_subsecond_reference_backend"),
    ("moon", "moon_reference_backend"),
    ("version", "unsupported_skyfield_version"),
    ("budget", "point_budget_reference_backend"),
])
def test_unsupported_conditions_use_identical_reference(real_case, monkeypatch, mode, reason):
    import skyfield
    cfg, ds, sch = real_case
    sch = Schedule(*(getattr(sch, k)[:1].astype(float if k in ("start_s", "dur_s") else int)
                     for k in ("station", "target", "start_s", "dur_s")))
    step = 1.
    if mode == "fractional":
        sch.start_s += .2
        sch.dur_s += .3
    elif mode == "subsecond":
        step = .5
    elif mode == "moon":
        cfg = cfg.with_(moon_sep_deg=30)
    elif mode == "version":
        monkeypatch.setattr(skyfield, "__version__", "unverified-version")
    cache = SkyfieldGeometryCache(cfg, ds, max_points=1 if mode == "budget" else 1_000_000)
    expected = validator.validate(cfg, ds, sch, step_s=step)
    cached = validator.validate(cfg, ds, sch, step_s=step, geometry_cache=cache)
    assert_same(expected, cached, exact=True)
    assert cache.stats()["fallback_reasons"] == {reason: 1}


@pytest.mark.parametrize("boundary", ["elev_min", "elev_max", "sun_max", "shadow", "radius"])
def test_numerical_boundaries_fall_back_to_original_grouping(real_case, monkeypatch, boundary):
    cfg, ds, sch = real_case
    sch = Schedule(*(getattr(sch, k)[:1].copy() for k in ("station", "target", "start_s", "dur_s")))
    baseline = validator.validate(cfg, ds, sch)
    if boundary == "shadow":
        monkeypatch.setattr(validator, "_near_shadow_boundary", lambda *args: True)
    elif boundary == "radius":
        from skyfield.api import EarthSatellite
        ts, _ = validator._sky(str(Path(cfg.cache_dir)))
        t0 = cfg.start_dt
        seconds = validator._sample_times(float(sch.start_s[0]), float(sch.dur_s[0]), 1.)
        times = ts.utc(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second + seconds)
        target = int(sch.target[0])
        geo = EarthSatellite(ds.line1[target], ds.line2[target], ts=ts).at(times)
        lower = np.linalg.norm(geo.position.km, axis=0).min()
        mean_motion = np.sqrt(398600.4418 / (lower / .9)**3) * 60
        # Change only the independent plausibility bound, leaving Skyfield's
        # real propagated orbit unchanged, to exercise its exact boundary.
        monkeypatch.setattr(validator, "Satrec", SimpleNamespace(twoline2rv=lambda *args:
                            SimpleNamespace(no_kozai=mean_motion, ecco=0.)))
    else:
        column = {"elev_min": 0, "elev_max": 1, "sun_max": 2}[boundary]
        cfg = cfg.with_(**{boundary: float(baseline["per_task"][0, column]) + 1e-10})
    cache = SkyfieldGeometryCache(cfg, ds)
    expected = validator.validate(cfg, ds, sch)
    actual = validator.validate(cfg, ds, sch, geometry_cache=cache)
    assert_same(expected, actual, exact=True)
    assert cache.stats()["fallback_reasons"] == {"geometry_boundary_reference_backend": 1}


@pytest.mark.parametrize("distance,expected", [(-.5, True), (.5, True), (-100., False), (100., False)])
def test_shadow_tangent_guard_uses_physical_distance(distance, expected):
    from skyfield.constants import ERAD
    radius = 7_000_000.
    closest = ERAD + distance
    u = np.array([-np.sqrt(1 - (closest / radius)**2), closest / radius, 0.])[:, None]
    position = np.array([radius, 0., 0.])[:, None]
    sun = u * 149_597_870_700. + position
    assert _near_shadow_boundary(position, sun) is expected


def test_shadow_guard_fails_safe_on_nonfinite_geometry():
    assert _near_shadow_boundary(np.full((3, 1), np.nan), np.zeros((3, 1)))


def test_shadow_tangent_behind_sun_ray_does_not_trigger():
    from skyfield.constants import ERAD
    radius = 7_000_000.
    u = np.array([np.sqrt(1 - (ERAD / radius)**2), ERAD / radius, 0.])[:, None]
    position = np.array([radius, 0., 0.])[:, None]
    sun = u * 149_597_870_700. + position
    assert not _near_shadow_boundary(position, sun)
