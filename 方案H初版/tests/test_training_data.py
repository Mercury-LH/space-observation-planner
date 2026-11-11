"""合成场景生成器：v0.2 与数据集冻结的参数完全一致；v0.3 补充场景的设计约束。"""
import numpy as np

from obsplan import scenes
from obsplan.training_data import v2_sampling as v2, v3_sampling as v3

V2_FROZEN_SHA256 = "52f2b3af32c95aa8769dfb24cb7691ae586db33008ea515c80c63a4cab04d4a3"   # datasets/synthetic-v0.2/catalog-specs.json


def _gc_deg(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    c = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def test_v2_specs_match_the_frozen_dataset():
    rows = v2.specs()
    assert v2.specs_sha256(rows) == V2_FROZEN_SHA256
    assert v2.sampling_report(rows)["passed"]


def test_v3_specs_design():
    rows = v3.specs()
    assert [sum(r["split"] == s for r in rows) for s in v3.COUNTS] == list(v3.COUNTS.values())
    ids, seeds = {r["catalog_id"] for r in rows}, {r["seed"] for r in rows}
    assert len(ids) == len(seeds) == len(rows)
    assert not seeds & {r["seed"] for r in v2.specs()}
    A = v2.STATION_ANCHORS
    for r in rows:
        a = np.array(r["station_anchor_ids"])
        assert len(a) == r["n_stations"] == len(set(a.tolist()))
        assert not set(a.tolist()) & set(v3.EXCLUDED_ANCHORS)
        assert ((A[a, 1] >= 46) & (A[a, 1] > 0)).any()          # 至少一个北半球高纬度站
        assert (A[a, 1] < 0).any() and (A[a, 1] > 0).any()
    meo = np.array([r["family_probabilities"][1] for r in rows])
    assert 0.04 < np.median(meo) < 0.07 and meo.max() < 0.16          # 中轨比例中位约 5%（v0.2 中位 19%）
    assert len({r["start_utc"][5:7] for r in rows}) == 12


def test_v3_anchors_are_not_contest_stations(ds):
    used = np.unique(np.concatenate([r["station_anchor_ids"] for r in v3.specs()]))
    A = v2.STATION_ANCHORS[used]
    d = _gc_deg(A[:, 0][:, None], A[:, 1][:, None], ds.st_lon[None], ds.st_lat[None])
    assert d.min() > 2.5          # 约 280 km 以上；v0.2 的 10 号锚点离赛题站不到 1°，v0.3 已排除


def test_v3_catalog_has_shell_and_low_leo():
    spec = dict(v3.specs()[0], n_targets=600)
    ds, table = v2.generate_catalog(spec)
    leo = table["family"] == 0
    alt = table["semimajor_km"][leo] - 6378.135
    shell = spec["LEO_mixture"]["shell_altitude_km"]
    in_shell = (alt >= shell[2]) & (alt <= shell[3])
    assert in_shell.mean() > 0.6 * spec["LEO_mixture"]["shell_fraction"]
    assert np.median(alt) < 800
    assert (table["mean_element_perigee_km"] >= spec["min_perigee_km"]).all()
    assert len(ds.names) == 600 and not ds.issues


def test_scene_selection_by_dataset():
    assert all(c.startswith("v3-") for c in scenes.select("train", 5, dataset="v3"))
    assert len(scenes.select("test", dataset="v3")) == 36
    assert len(scenes.select("test", dataset="v2")) == 90
    assert scenes.spec_of("v3-cat-000")["sampling_version"] == v3.SAMPLING_VERSION
