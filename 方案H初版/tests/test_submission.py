from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from obsplan.config import Config
from obsplan.schedule import Schedule, counted_mask
from obsplan.submission import END, START, json_bytes, parse_json, records, score, to_schedule, utc


@pytest.fixture
def catalog():
    return SimpleNamespace(satnum=np.array([28446, 25544, 33591]), priority=np.array([5., 3., 2.]),
                           n_stations=10, n_targets=3)


def row(n=28446, s=1, a=100, duration=240):
    return {"noradId": n, "siteCode": f"site_{s}", "startTime": utc(START + timedelta(seconds=a)),
            "endTime": utc(START + timedelta(seconds=a + duration))}


def report(catalog, rows, forecasts=None):
    return score(rows, catalog, forecasts if forecasts is not None else rows, source={"kind": "test"})


@pytest.mark.parametrize("duration,factor", [(119.999999, 0), (120, .6), (180, .8), (240, 1), (300, 1)])
def test_intersection_duration_and_saturation(catalog, duration, factor):
    result = report(catalog, [row(duration=300)], [row(duration=duration)])
    task = result["tasks"][0]
    if duration < 120:
        assert task["status"] == "hard" and task["score"] == 0
    elif duration < 295:
        assert task["status"] == "unresolved"  # overrun >5s is explicitly unspecified
        assert result["summary"]["final_score"] is None
    else:
        assert task["duration_factor"] == factor
    exact = report(catalog, [row(duration=duration)])
    assert exact["tasks"][0]["duration_factor"] == pytest.approx(factor)


@pytest.mark.parametrize("deviation,status", [(0, "normal"), (1, "normal"), (1.000001, "soft"),
                                             (5, "soft"), (5.000001, "unresolved")])
def test_one_and_five_second_boundaries(catalog, deviation, status):
    result = report(catalog, [row()], [row(a=100 + deviation, duration=240 - deviation)])
    task = result["tasks"][0]
    assert task["status"] == status
    assert task["effective_duration_s"] == pytest.approx(240 - deviation)
    if status != "unresolved":
        assert task["score"] == pytest.approx(5 * (.6 + .4 * (120 - deviation) / 120) * (.5 if status == "soft" else 1))


def test_inside_long_window_is_not_endpoint_mismatch(catalog):
    result = report(catalog, [row(a=200)], [row(a=0, duration=1000)])
    assert result["summary"]["clean"] and result["summary"]["final_score"] == 5


@pytest.mark.parametrize("gap,hard", [(29.999999, 1), (30, 0)])
def test_site_gap(catalog, gap, hard):
    rows = [row(duration=120), row(n=25544, a=220 + gap, duration=120)]
    assert report(catalog, rows)["summary"]["hard_errors"] == hard


@pytest.mark.parametrize("gap,hard", [(172799.999999, 1), (172800, 0)])
def test_satellite_gap(catalog, gap, hard):
    rows = [row(duration=120), row(s=2, a=220 + gap, duration=120)]
    assert report(catalog, rows)["summary"]["hard_errors"] == hard


def test_invalid_observation_stays_in_adjacency_chain(catalog):
    rows = [row(a=0, duration=120), row(s=2, a=86400, duration=60),
            row(s=3, a=172920, duration=120), row(s=4, a=345840, duration=120)]
    result = report(catalog, rows)
    assert [t["status"] for t in result["tasks"]] == ["normal", "hard", "hard", "normal"]
    sch = to_schedule(rows, Config(), catalog)
    assert counted_mask(Config(), sch, np.array([True, False, True, True])).tolist() == [True, False, False, True]


def test_document_example_and_no_rounding(catalog):
    rows = [row(duration=210), row(n=99999, s=2), row(n=25544, s=3, a=200, duration=210)]
    forecasts = [row(duration=210), row(n=25544, s=3, a=203, duration=207)]
    result = report(catalog, rows, forecasts)
    assert result["summary"] == {"tasks": 3, "covered": 2, "hard_errors": 1, "soft_errors": 1,
                                 "unresolved": 0, "base_score": pytest.approx(5.835),
                                 "penalty_factor": 0, "final_score": 0, "clean": False}


def test_no_overlap_empty_and_multiple_reasons_count_once(catalog):
    assert report(catalog, [], [])["summary"]["final_score"] == 0
    result = report(catalog, [row(n=99999), row(n=99999, a=101)], [])
    assert result["summary"]["hard_errors"] == 2
    assert result["summary"]["covered"] == 0
    assert result["summary"]["penalty_factor"] == 0
    assert len(result["tasks"][1]["reasons"]) > 1


def test_never_sums_disjoint_windows(catalog):
    result = report(catalog, [row(a=0, duration=240)], [row(a=0, duration=100), row(a=140, duration=100)])
    assert result["tasks"][0]["effective_duration_s"] == 100
    assert result["summary"]["hard_errors"] == 1


@pytest.mark.parametrize("patch", [{"noradId": True}, {"noradId": 28446.0}, {"noradId": "28446"},
                                    {"noradId": 8882}, {"noradId": 100045}, {"siteCode": "site_01"},
                                    {"startTime": "2026-07-31T00:00:00+00:00"}, {"extra": 1}])
def test_strict_wire_format(catalog, patch):
    value = {**row(), **patch}
    with pytest.raises(ValueError):
        to_schedule([value], Config(), catalog)
    result = report(catalog, [value], [])
    assert not result["format_valid"] and result["summary"]["hard_errors"] == 1


@pytest.mark.parametrize("text", ['{}', '[NaN]', '[{"noradId":28446,"noradId":25544}]', '[// comment\n]'])
def test_standard_json_only(text):
    with pytest.raises(ValueError):
        parse_json(text)


def test_numeric_site_sort_roundtrip_and_last_second(catalog):
    last = int((END - START).total_seconds())
    sch = Schedule(np.array([9, 1, 0]), np.array([0, 1, 2]), np.array([last - 120, 300, 0]), np.full(3, 120))
    rows = records(Config(), catalog, sch)
    assert [r["siteCode"] for r in rows] == ["site_1", "site_2", "site_10"]
    assert rows[-1]["endTime"] == "2026-08-06T23:59:59Z"
    assert len(to_schedule(parse_json(json_bytes(rows)), Config(), catalog)) == 3
    sch.start_s[0] += 1
    with pytest.raises(ValueError, match="范围"):
        records(Config(), catalog, sch)
    with pytest.raises(ValueError, match="排列"):
        to_schedule(rows[::-1], Config(), catalog)
    assert not report(catalog, rows[::-1], rows)["format_valid"]


def test_export_never_relabels_or_silently_drops(catalog):
    catalog.satnum[0] = 100045
    sch = Schedule(np.array([0]), np.array([0]), np.array([0]), np.array([120]))
    with pytest.raises(ValueError, match="五位"):
        records(Config(), catalog, sch)


def test_seven_day_problem_does_not_claim_three_observation_upper_bound():
    from obsplan.bounds import _dual_setup
    cfg = Config()
    assert cfg.start_dt == START and cfg.days == 7
    pb = SimpleNamespace(gap=cfg.gap_cells, m_min=cfg.min_cells, n_samples=cfg.n_cells + 1)
    with pytest.raises(NotImplementedError, match="4 次"):
        _dual_setup(pb)
