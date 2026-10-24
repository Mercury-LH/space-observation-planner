"""Untrusted solver outputs must fail closed before geometry or indexing.

Synthetic geometry checks the independent validator's orchestration and sample
boundaries. It does not replace the existing real Skyfield integration test.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from obsplan.config import Config
from obsplan.schedule import Schedule
from obsplan import validator


def plan(**changes):
    values = dict(station=np.array([0]), target=np.array([0]),
                  start_s=np.array([0.0]), dur_s=np.array([120.0]))
    values.update(changes)
    return Schedule(**values)


@pytest.fixture
def catalog():
    return SimpleNamespace(n_stations=3, n_targets=3,
                           st_lat=np.zeros(3), st_lon=np.zeros(3),
                           st_alt_m=np.zeros(3), line1=["tle1"] * 3,
                           line2=["tle2"] * 3)


@pytest.fixture
def no_geometry(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Malformed schedules must not load ephemerides")
    monkeypatch.setattr(validator, "_sky", forbidden)


@pytest.mark.parametrize("changes", [
    {"station": np.array([-1])}, {"station": np.array([3])},
    {"target": np.array([-1])}, {"target": np.array([3])},
    {"target": np.array([2**64 - 1], dtype=np.uint64)},
    {"station": np.array([0.5])}, {"target": np.array([0.0])},
    {"station": np.array([True])}, {"target": np.array([np.nan])},
    {"station": np.array(0)}, {"target": np.array([[0]])},
    {"start_s": np.array([[0.0]])}, {"dur_s": np.array([])},
    {"station": [[0], [0, 1]]},
    {"start_s": np.array(["0"])}, {"dur_s": np.array([120 + 0j])},
    {"dur_s": np.array([True])},
    {"start_s": np.array([-0.01])},
    {"start_s": np.array([6 * 86400 - 119.99])},
    {"start_s": np.array([np.iinfo(np.int64).max]),
     "dur_s": np.array([np.iinfo(np.int64).max])},
    {"dur_s": np.array([-120])}, {"dur_s": np.array([0])},
    {"dur_s": np.array([119.999])},
] + [{field: np.array([value])}
     for field in ("start_s", "dur_s")
     for value in (np.nan, np.inf, -np.inf)])
def test_malformed_schedule_is_reported_without_geometry(catalog, no_geometry, changes):
    result = validator.validate(Config(), catalog, plan(**changes))
    assert result["legal"] is False
    assert result["violations"]
    assert result["geometry_checked"] is False
    assert result["min_elev_margin"] is None
    assert result["max_elev_margin"] is None
    assert result["sun_margin"] is None


def test_missing_schedule_arrays_are_reported(catalog, no_geometry):
    result = validator.validate(Config(), catalog, SimpleNamespace(station=[0]))
    assert result["legal"] is False
    assert result["invalid_mask"].size == 0


def test_nested_station_conflicts_are_all_flagged(catalog, no_geometry):
    schedule = plan(station=np.array([0, 0, 0]), target=np.array([0, 1, 2]),
                    start_s=np.array([0, 200, 400]), dur_s=np.array([1000, 120, 120]))
    result = validator.validate(Config(), catalog, schedule)
    assert result["violations"]["站点时间冲突或切换不足"] == 2
    np.testing.assert_array_equal(result["invalid_mask"], [False, True, True])


def test_nested_target_intervals_use_furthest_previous_end(catalog, no_geometry):
    # The last row clears row 1's 48-hour gap, but still conflicts with row 0.
    schedule = plan(station=np.array([0, 1, 2]), target=np.array([0, 0, 0]),
                    start_s=np.array([0, 172920, 345960]),
                    dur_s=np.array([350000, 120, 120]))
    result = validator.validate(Config(), catalog, schedule)
    assert result["violations"]["同目标间隔不足48小时"] == 2
    np.testing.assert_array_equal(result["invalid_mask"], [False, True, True])


@pytest.fixture
def synthetic_geometry(monkeypatch):
    """Fake only astronomy outputs; exercise the validator's full control flow."""
    from skyfield import api

    state = SimpleNamespace(alt=lambda times: np.full(len(times), 45.0),
                            sun=lambda times: np.full(len(times), -10.0),
                            radius=7000.0, sunlit=True, sampled=[])

    class Topocentric:
        def at(self, times):
            state.sampled.append(np.array(times))
            return SimpleNamespace(altaz=lambda: (
                SimpleNamespace(degrees=state.alt(times)), None, None))

    class Satellite:
        def __init__(self, *args, **kwargs):
            pass

        def at(self, times):
            position = np.zeros((3, len(times)))
            position[0] = state.radius
            return SimpleNamespace(position=SimpleNamespace(km=position),
                                   is_sunlit=lambda eph: np.full(len(times), state.sunlit))

        def __sub__(self, site):
            return Topocentric()

    class Earth:
        def __add__(self, site):
            return self

        def at(self, times):
            apparent = SimpleNamespace(altaz=lambda: (
                SimpleNamespace(degrees=state.sun(times)), None, None))
            return SimpleNamespace(observe=lambda body: SimpleNamespace(apparent=lambda: apparent))

    ts = SimpleNamespace(utc=lambda year, month, day, hour, minute, secs: np.asarray(secs))
    eph = dict(earth=Earth(), sun=object(), moon=object())
    monkeypatch.setattr(validator, "_sky", lambda cache: (ts, eph))
    monkeypatch.setattr(api, "EarthSatellite", Satellite)
    monkeypatch.setattr(api, "wgs84", SimpleNamespace(latlon=lambda *args, **kwargs: object()))
    monkeypatch.setattr(validator, "Satrec", SimpleNamespace(twoline2rv=lambda *args:
        SimpleNamespace(no_kozai=np.sqrt(398600.4418 / 7000**3) * 60, ecco=0.0)))
    return state


@pytest.mark.parametrize("duration", [120, 240, 300])
def test_valid_duration_has_no_invented_240_second_hard_limit(catalog, synthetic_geometry, duration):
    result = validator.validate(Config(), catalog, plan(dur_s=np.array([duration])))
    assert result["legal"] is True
    assert result["geometry_checked"] is True
    assert result["min_elev_margin"] == 35
    assert result["max_elev_margin"] == 40
    assert result["sun_margin"] == 4
    assert not result["invalid_mask"].any()


def test_off_grid_endpoint_is_checked(catalog, synthetic_geometry):
    synthetic_geometry.alt = lambda times: np.where(
        np.isclose(times, 120.5, rtol=0, atol=1e-9), 85.1, 45.0)
    result = validator.validate(Config(), catalog,
                                plan(start_s=np.array([0.2]), dur_s=np.array([120.3])))
    assert result["legal"] is False
    assert result["violations"]["仰角高于上限"] == 1
    samples = synthetic_geometry.sampled[0]
    assert samples[0] == 0.2
    assert samples[-1] == 120.5
    assert np.diff(samples).max() <= 1.0 + 1e-12


@pytest.mark.parametrize("channel,violation", [
    ("alt", "仰角计算非有限"), ("sun", "太阳几何计算非有限")])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_geometry_is_illegal(catalog, synthetic_geometry, channel, violation, value):
    setattr(synthetic_geometry, channel, lambda times: np.full(len(times), value))
    result = validator.validate(Config(), catalog, plan())
    assert result["legal"] is False
    assert result["violations"][violation] == 1
    assert result["invalid_mask"][0]


@pytest.mark.parametrize("radius", [np.nan, np.inf, 500.0, 100000.0])
def test_failed_or_implausible_propagation_is_illegal(catalog, synthetic_geometry, radius):
    synthetic_geometry.radius = radius
    result = validator.validate(Config(), catalog, plan())
    assert result["legal"] is False
    assert result["violations"]["轨道传播失败或不可信"] == 1


def test_exact_switch_repeat_and_horizon_boundaries_remain_legal(catalog, synthetic_geometry):
    schedule = plan(station=np.array([0, 0, 1, 2]), target=np.array([0, 1, 0, 2]),
                    start_s=np.array([0, 150, 120 + 48 * 3600, 6 * 86400 - 120]),
                    dur_s=np.full(4, 120))
    assert validator.validate(Config(), catalog, schedule)["legal"] is True


@pytest.mark.parametrize("dtype", [int, float])
def test_empty_schedule_is_legal_without_ephemerides(catalog, no_geometry, dtype):
    schedule = plan(**{field: np.array([], dtype=dtype)
                       for field in ("station", "target", "start_s", "dur_s")})
    result = validator.validate(Config(), catalog, schedule)
    assert result["legal"] is True
    assert result["invalid_mask"].shape == (0,)
    assert result["per_task"].shape == (0, 3)


@pytest.mark.parametrize("step", [0, -1, np.nan, np.inf, 2])
def test_validation_cannot_silently_use_coarse_or_invalid_sampling(catalog, no_geometry, step):
    with pytest.raises(ValueError, match="step_s"):
        validator.validate(Config(), catalog, plan(), step_s=step)
