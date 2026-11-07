"""Sampling-contract and native orbit tests, not surrogate schedule validation."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
import json

import numpy as np
import pytest
from sgp4.api import Satrec, jday

from obsplan.config import Config
from obsplan.data import load, tle_checksum_ok
from obsplan.training_data.catalog import catalog_specs as old_specs, write_native_input
from obsplan.training_data.v2_sampling import (
    CATALOG_SEED_BASE, CORE_CONTROLS, FAMILIES, REGIONAL_ANCHORS, STATION_ANCHORS,
    STRESS_TYPES, _cdf, _ppf, generate_catalog, sampling_report, size_bin, specs, specs_sha256,
)


@pytest.fixture(scope="module")
def design():
    return specs()


def test_spec_design_is_reproducible_json_safe_and_separated_from_v1(design):
    assert design == specs()
    assert design == json.loads(json.dumps(design, allow_nan=False))
    assert len(design) == len({r["seed"] for r in design}) == len({r["catalog_id"] for r in design}) == 600
    assert Counter(r["split"] for r in design) == {"train": 420, "validation": 90, "test": 90}
    assert Counter(r["distribution"] for r in design) == {"nominal_gaussian": 540, "stress": 60}
    assert {r["seed"] for r in design}.isdisjoint(r["seed"] for r in old_specs())
    assert {r["catalog_id"] for r in design}.isdisjoint(r["catalog_id"] for r in old_specs())
    assert all(r["seed"] == CATALOG_SEED_BASE + i for i, r in enumerate(design))
    for r in design:
        start = datetime.fromisoformat(r["start_utc"])
        assert start.year == (start + timedelta(days=6)).year == 2027
        assert start.month == (start + timedelta(days=6)).month
        assert start.utcoffset() == timedelta(0)


def test_core_distribution_contract_checks_latents_and_month_strata(design):
    report = sampling_report(design)
    assert report["passed"], report["errors"]
    assert report["specs_sha256"] == specs_sha256(design)
    for split, count in (("train", 420), ("validation", 60), ("test", 60)):
        assert report["coverage"][split]["month_counts"] == {str(m): count // 12 for m in range(1, 13)}
        for key, check in report["core_distribution_checks"][split].items():
            assert check["passed"] and check["one_per_quantile_stratum"]
            assert check["n"] == count
            assert check["max_cdf_error"] <= check["limit"] <= 1 / count + 1.1e-12
            assert check["params"] == list(CORE_CONTROLS[key])
    # Frozen strata produce meaningful same-distribution large holdouts.
    assert report["coverage"]["validation"]["targets_ge_10000"] == 10
    assert report["coverage"]["test"]["targets_ge_10000"] == 10


@pytest.mark.parametrize("mutation,fragment", [
    (lambda r: r[0].update(seed=r[1]["seed"]), "duplicate"),
    (lambda r: r[0].update(split="test"), "split counts"),
    (lambda r: r[0].update(n_targets=r[0]["n_targets"] + 1), "rounded counts"),
    (lambda r: r[0]["sampling_latents"].update(leo_fraction=.9), "sampling contract"),
    (lambda r: r[0].update(station_anchor_ids=[0] * r[0]["n_stations"]), "station anchor"),
    (lambda r: r[0].update(priority_probabilities=[.5, .4, .4]), "probabilities"),
    (lambda r: r[0].update(start_utc="2027-12-31T00:00:00+00:00"), "planning interval"),
    (lambda r: r[-1].update(stress_type="undocumented"), "stress"),
])
def test_sampling_audit_rejects_semantic_changes(design, mutation, fragment):
    changed = deepcopy(design)
    mutation(changed)
    report = sampling_report(changed)
    assert not report["passed"]
    assert any(fragment in error for error in report["errors"])


def test_sampling_audit_rejects_nonfinite_and_malformed_inputs(design):
    changed = deepcopy(design)
    changed[0]["sampling_latents"]["n_targets"] = float("nan")
    assert not sampling_report(changed)["passed"]
    assert not sampling_report([])["passed"]
    assert not sampling_report([None])["passed"]


def test_stress_groups_are_preassigned_separate_from_core(design):
    stress = [r for r in design if r["distribution"] == "stress"]
    for split in ("validation", "test"):
        assert Counter(r["stress_type"] for r in stress if r["split"] == split) == {name: 5 for name in STRESS_TYPES}
    for r in stress:
        assert r["sampling_quantiles"] == {}
        control = r["sampling_latents"]
        if r["stress_type"] == "high_load":
            assert r["n_targets"] >= 11000 and r["n_stations"] <= 7
        elif r["stress_type"] == "low_load":
            assert r["n_targets"] <= 3500 and r["n_stations"] >= 12
        elif r["stress_type"] == "leo_polar":
            assert control["leo_fraction"] >= .93
            assert r["orbit_parameters"]["LEO"]["inclination_deg"] == [95., 7., 75., 110.]
        elif r["stress_type"] == "high_priority":
            assert r["priority_probabilities"][2] >= .60
        elif r["stress_type"] == "regional_network":
            assert set(r["station_anchor_ids"]) <= set(REGIONAL_ANCHORS)
            assert datetime.fromisoformat(r["start_utc"]).month == 6
    for r in design:
        if r["network_mode"] == "global_two_hemispheres":
            latitude = STATION_ANCHORS[r["station_anchor_ids"], 1]
            assert (latitude < 0).any() and (latitude > 0).any()


def test_inverse_cdf_handles_remote_tails_without_clipping_normal_samples():
    params = (20., 12., 0., 120.)  # upper normal CDF rounds to one
    u = np.array([0., .00001, .1, .5, .9, .99999, 1.])
    values = _ppf(u, params)
    assert values[0] == 0 and values[-1] == 120
    assert (np.diff(values) > 0).all()
    np.testing.assert_allclose([_cdf(x, params) for x in values], u, atol=1e-14, rtol=0)
    with pytest.raises(ValueError):
        _ppf([-.1], params)
    with pytest.raises(ValueError):
        _ppf([.5], (20., 0., 0., 120.))


@pytest.mark.parametrize("index", [0, 419, 540, 545, 550, 555, 560, 565])
def test_generated_tles_native_roundtrip_and_six_day_propagation(design, index, tmp_path):
    spec = {**design[index], "n_targets": 48}
    ds, table = generate_catalog(spec)
    write_native_input(tmp_path / "input", ds)
    published = load(Config(data_dir=str(tmp_path / "input")))
    assert not published.issues
    assert published.line1 == ds.line1 and published.line2 == ds.line2
    np.testing.assert_array_equal(published.priority, table["base_priority"])
    assert table["station_anchor_id"].tolist() == spec["station_anchor_ids"]
    np.testing.assert_allclose(np.c_[published.st_lon, published.st_lat, published.st_alt_m], table["stations"],
                               rtol=0, atol=5.1e-7)
    assert table["mean_element_perigee_km"].min() >= 450
    start = datetime.fromisoformat(spec["start_utc"])
    epoch = start + timedelta(days=3)
    for i, (a, b) in enumerate(zip(published.line1, published.line2)):
        assert len(a) == len(b) == 69 and tle_checksum_ok(a) and tle_checksum_ok(b)
        sat = Satrec.twoline2rv(a, b)
        assert sat.epochyr == 27
        assert sat.epochdays == pytest.approx(epoch.timetuple().tm_yday)
        assert sat.ecco == pytest.approx(table["eccentricity"][i], abs=5.1e-8)
        assert np.degrees(sat.inclo) == pytest.approx(table["inclination_deg"][i], abs=5.1e-5)
        params = spec["orbit_parameters"][FAMILIES[int(table["family"][i])]]
        for key, values in (("altitude_km", table["semimajor_km"] - 6378.135),
                            ("eccentricity", table["eccentricity"]),
                            ("inclination_deg", table["inclination_deg"])):
            assert params[key][2] <= values[i] <= params[key][3]
        for day in (0, 3, 6):
            dt = start + timedelta(days=day)
            jd, fr = jday(dt.year, dt.month, dt.day, 0, 0, 0)
            error, position, velocity = sat.sgp4(jd, fr)
            assert not error and np.isfinite([*position, *velocity]).all()


def test_catalog_generation_is_reproducible_with_independent_station_rng(design):
    small = {**design[0], "n_targets": 32}
    a, ta = generate_catalog(small)
    b, tb = generate_catalog(small)
    assert a.line1 == b.line1 and a.line2 == b.line2
    for key in ta:
        np.testing.assert_array_equal(ta[key], tb[key])
    _, larger = generate_catalog({**small, "n_targets": 64})
    np.testing.assert_array_equal(ta["stations"], larger["stations"])
    c, tc = generate_catalog({**small, "seed": small["seed"] + 1})
    assert a.line2 != c.line2
    assert not np.array_equal(ta["stations"], tc["stations"])


@pytest.mark.parametrize("field,value", [("n_targets", 0), ("n_targets", 16001), ("n_targets", 12.5),
                                        ("n_stations", 3), ("n_stations", 17), ("seed", True)])
def test_catalog_rejects_invalid_dimensions(design, field, value):
    spec = {**design[0], "n_targets": 8, field: value}
    with pytest.raises(ValueError):
        generate_catalog(spec)


def test_size_bins_are_not_mistaken_for_exact_legacy_sizes():
    assert [size_bin(n) for n in (1000, 3999, 4000, 9999, 10000, 16000)] == [
        "small", "small", "medium", "medium", "large", "large"]
