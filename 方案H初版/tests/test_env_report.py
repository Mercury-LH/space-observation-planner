"""加分项环境报告：视星等模型的基本性质；月距与几何量和 Skyfield 独立计算一致。"""
import numpy as np
import pytest

from obsplan.env_report import magnitude, task_geometry
from obsplan.schedule import Schedule


def test_magnitude_model():
    m = magnitude(np.array([1000.0, 10000.0, 1000.0, 1000.0]), np.array([90.0, 90.0, 0.0, 150.0]), m_std=6.0)
    assert m[0] == pytest.approx(6.0)                      # 定义：1000 km、相位角 90° 时等于 m_std
    assert m[1] == pytest.approx(11.0)                     # 距离 10 倍暗 5 等
    assert m[2] < m[0] < m[3]                              # 满相（相位角 0°）最亮，相位角越大越暗


@pytest.mark.slow
def test_geometry_matches_skyfield(cfg, ds):
    """几个任务的月距、距离与 Skyfield 独立算出的一致（误差远小于 30° 的判定阈值）。"""
    from skyfield.api import EarthSatellite, wgs84
    from obsplan.validator import _ephemeris_dir, _sky
    rng = np.random.default_rng(1)
    n = 24
    tg = rng.integers(0, ds.n_targets, n)
    st = rng.integers(0, ds.n_stations, n)
    t0 = rng.uniform(0, 5 * 86400, n).round()
    sch = Schedule(st, tg, t0, np.full(n, 120.0))
    g = task_geometry(cfg, ds, sch, step_s=60.0)
    ts, eph = _sky(_ephemeris_dir(cfg))
    checked = 0
    for i in range(n):
        sat = EarthSatellite(ds.line1[tg[i]], ds.line2[tg[i]], ts=ts)
        site = wgs84.latlon(ds.st_lat[st[i]], ds.st_lon[st[i]], elevation_m=ds.st_alt_m[st[i]])
        secs = np.array([t0[i], t0[i] + 60.0, t0[i] + 120.0])
        t = ts.utc(cfg.start_dt.year, cfg.start_dt.month, cfg.start_dt.day, 0, 0, secs)
        obs = (eph["earth"] + site).at(t)
        moon = obs.observe(eph["moon"]).apparent()
        malt = moon.altaz()[0].degrees
        topo = (sat - site).at(t)
        sep = topo.separation_from(moon).degrees if hasattr(topo, "separation_from") else None
        dist = topo.distance().km
        assert g["range_max_km"][i] == pytest.approx(dist.max(), rel=2e-3)
        if sep is not None and (malt > 1).all():
            assert g["moon_sep_min"][i] == pytest.approx(sep.min(), abs=0.5)
            checked += 1
    assert checked >= 3                                    # 至少几个任务时月亮在地平线以上，月距确实被比较过
