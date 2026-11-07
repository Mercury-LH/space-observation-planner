"""Check physical input round-trips, grouped splits and label provenance.

Only SGP4 propagation in this file is real geometry. Repair tests use explicit
validator reports to exercise rejection paths without claiming Skyfield checks.
"""
from collections import Counter
from datetime import datetime, timedelta
import json

import numpy as np
import pytest
from sgp4.api import Satrec, jday

from obsplan.config import Config
from obsplan.data import load, tle_checksum_ok
from obsplan.schedule import Schedule
from obsplan.training_data import build as b
from obsplan.training_data.catalog import (
    catalog_specs, generate_catalog, make_variant, write_native_input,
)
from obsplan.visibility import Windows


@pytest.fixture
def small_catalog():
    spec = {**catalog_specs()[0], "n_targets": 48}
    ds, table = generate_catalog(spec)
    return spec, ds, table


def test_exported_tle_catalog_is_deterministic_and_native_input_roundtrips(small_catalog, tmp_path):
    spec, ds, table = small_catalog
    again, same_table = generate_catalog(spec)
    assert again.line1 == ds.line1 and again.line2 == ds.line2
    for key in table:
        np.testing.assert_array_equal(table[key], same_table[key])
    different, _ = generate_catalog({**spec, "seed": spec["seed"] + 1})
    assert all(a != b for a, b in zip(ds.line2, different.line2))
    write_native_input(tmp_path / "input", ds)
    published = load(Config(data_dir=str(tmp_path / "input")))
    assert not published.issues
    assert published.line1 == ds.line1 and published.line2 == ds.line2
    assert published.names == ds.names
    assert published.priority_label == ds.priority_label
    np.testing.assert_array_equal(published.satnum, ds.satnum)
    np.testing.assert_array_equal(published.priority, ds.priority)
    np.testing.assert_allclose(published.st_lon, ds.st_lon, rtol=0, atol=5.1e-11)
    np.testing.assert_allclose(published.st_lat, ds.st_lat, rtol=0, atol=5.1e-11)
    np.testing.assert_allclose(published.st_alt_m, ds.st_alt_m, rtol=0, atol=5.1e-7)
    assert len(np.unique(published.satnum)) == ds.n_targets
    start = datetime.fromisoformat(spec["start_utc"])
    for i, (line1, line2) in enumerate(zip(published.line1, published.line2)):
        assert len(line1) == len(line2) == 69
        assert tle_checksum_ok(line1) and tle_checksum_ok(line2)
        sat = Satrec.twoline2rv(line1, line2)
        # Verify exported units/rounding against the original SGP4 mean elements.
        assert sat.ecco == pytest.approx(table["eccentricity"][i], abs=5.1e-8)
        assert np.degrees(sat.inclo) == pytest.approx(table["inclination_deg"][i], abs=5.1e-5)
        assert sat.no_kozai == pytest.approx(table["mean_motion_rad_min"][i], abs=2.3e-11)
        assert table["mean_element_perigee_km"][i] >= 450
        for day in (0, 3, 6):
            dt = start + timedelta(days=day)
            jd, fr = jday(dt.year, dt.month, dt.day, 0, 0, 0)
            error, position, velocity = sat.sgp4(jd, fr)
            assert error == 0
            assert np.isfinite([*position, *velocity]).all()
            radius = np.linalg.norm(position)
            a = (398600.4418 / (sat.no_kozai / 60.0) ** 2) ** (1 / 3)
            assert 0.9 * a * (1 - sat.ecco) <= radius <= 1.1 * a * (1 + sat.ecco)


def test_frozen_120_scene_spec_has_disjoint_catalogs_and_expected_ood():
    specs = catalog_specs()
    assert specs == catalog_specs()
    assert len(specs) == len({s["catalog_id"] for s in specs}) == len({s["seed"] for s in specs}) == 30
    assert Counter(s["split"] for s in specs) == {"train": 20, "validation": 5, "test": 5}
    assert sum(s["n_targets"] for s in specs) == 103_000
    assert Counter(s["n_targets"] for s in specs if s["split"] == "train") == {1000: 12, 5000: 5, 10000: 3}
    scenes = [(s["catalog_id"], f"{s['catalog_id']}-v{v}", s["split"]) for s in specs for v in range(4)]
    assert len(scenes) == len({scene_id for _, scene_id, _ in scenes}) == 120
    assert Counter(split for _, _, split in scenes) == {"train": 80, "validation": 20, "test": 20}
    for catalog in {row[0] for row in scenes}:
        assert len({split for cat, _, split in scenes if cat == catalog}) == 1
    shifted = {s["catalog_id"] for s in specs if s["distribution"] == "high_leo_shift"}
    assert shifted == {"cat-027", "cat-029"}
    assert all(s["split"] == "test" for s in specs if s["catalog_id"] in shifted)
    assert len({s["start_utc"] for s in specs}) == 30


def synthetic_windows(n_targets=48):
    rows = []
    for target, count in enumerate((2, 8, 12, 24, 50)):
        for j in range(count):
            station = j % 10
            start = (j % 6) * 2880 + target * 100 + j * 2
            rows.append((station, target, start, start + 4 + j % 5))
    rows.sort(key=lambda row: (row[0], row[2]))
    a = np.array(rows, np.int32)
    return Windows(a[:, 0].astype(np.int16), a[:, 1], a[:, 2], a[:, 3],
                   [[(0, 17280)] for _ in range(10)], 17281, np.zeros(n_targets, bool))


@pytest.mark.parametrize("variant", range(4))
def test_pool_budget_and_provenance_survive_variant_station_remap(small_catalog, variant):
    spec, ds, _ = small_catalog
    full = synthetic_windows(ds.n_targets)
    originals = {key: getattr(full, key).copy() for key in ("station", "target", "k0", "k1", "sgp4_failed")}
    sds, variant_windows, meta = make_variant(ds, full, spec["seed"], variant)
    selected_stations = np.array(meta["base_station_indices"])
    eligible_original = np.flatnonzero(np.isin(full.station, selected_stations))
    pool, local_ids = b.pool_windows(variant_windows, spec["seed"] + variant * 1000)
    repeated, repeated_ids = b.pool_windows(variant_windows, spec["seed"] + variant * 1000)
    source_ids = eligible_original[local_ids]
    np.testing.assert_array_equal(local_ids, repeated_ids)
    assert len(source_ids) == len(np.unique(source_ids))
    assert np.all(np.diff(source_ids) > 0)
    assert local_ids.dtype == np.int32
    assert np.bincount(pool.target, minlength=ds.n_targets).max() <= b.WINDOW_BUDGET == 12
    before_counts = np.bincount(variant_windows.target, minlength=ds.n_targets)
    np.testing.assert_array_equal(np.bincount(pool.target, minlength=ds.n_targets), np.minimum(before_counts, 12))
    for field in ("target", "k0", "k1"):
        np.testing.assert_array_equal(getattr(pool, field), getattr(full, field)[source_ids])
        np.testing.assert_array_equal(getattr(pool, field), getattr(repeated, field))
    np.testing.assert_array_equal(pool.station, np.searchsorted(selected_stations, full.station[source_ids]))
    assert sds.n_stations == (5 if variant == 3 else 10)
    assert sds.line1 == ds.line1 and sds.line2 == ds.line2
    for field, original in originals.items():
        np.testing.assert_array_equal(getattr(full, field), original)
    # Pool choice has no priority inputs; altering priorities must not alter its source rows.
    sds.priority[:] = 5
    _, after_priority_change = b.pool_windows(variant_windows, spec["seed"] + variant * 1000)
    np.testing.assert_array_equal(local_ids, after_priority_change)


def test_pool_takes_each_day_station_stratum_before_repeating():
    rows = [(station, 0, day * 2880 + repetition * 100, day * 2880 + repetition * 100 + 8)
            for day in range(6) for station in range(2) for repetition in range(2)]
    a = np.array(rows, np.int32)
    win = Windows(a[:, 0].astype(np.int16), a[:, 1], a[:, 2], a[:, 3],
                  [[(0, 17280)]] * 2, 17281, np.zeros(1, bool))
    pool, source = b.pool_windows(win, seed=1234)
    assert len(source) == 12
    assert len({(int(k) // 2880, int(s)) for k, s in zip(pool.k0, pool.station)}) == 12


def fake_report(n, invalid=(), geometry=True):
    mask = np.zeros(n, bool)
    mask[list(invalid)] = True
    return dict(legal=not mask.any(), geometry_checked=geometry,
                violations={"explicit_test_violation": len(invalid)} if invalid else {},
                invalid_mask=mask, per_task=np.tile([20., 30., -20.], (n, 1)),
                min_elev_margin=10., max_elev_margin=55., sun_margin=14.)


def test_grid_repair_keeps_original_evidence_and_realigns_first_remaining_action(tmp_path, monkeypatch):
    original = Schedule(np.zeros(4, np.int64), np.array([2, 0, 2, 1]),
                        np.array([0, 300, 172950, 173250]), np.array([120, 240, 120, 180]))
    calls = []

    def validate_stub(cfg, ds, plan, step_s):
        assert step_s == 1.0
        calls.append(plan)
        return fake_report(len(plan), (0, 3)) if len(calls) == 1 else fake_report(len(plan))

    monkeypatch.setattr(b, "validate", validate_stub)
    repaired, report, keep = b.checked_plan(Config(), None, original, tmp_path, "grid")
    assert keep.tolist() == [False, True, True, False]
    assert len(calls) == 2
    assert report["removed_tasks"] == 2 and report["original_tasks"] == 4
    assert report["legal"] and report["geometry_checked"] and report["step_s"] == 1.0
    assert len(report["failed_attempts"]) == 1
    with np.load(tmp_path / "grid_before_repair.npz", allow_pickle=False) as old:
        np.testing.assert_array_equal(old["target"], original.target)
        np.testing.assert_array_equal(old["start_s"], original.start_s)
    with np.load(tmp_path / "grid_before_repair_geometry.npz", allow_pickle=False) as old:
        np.testing.assert_array_equal(old["invalid_mask"], ~keep)
    assert json.loads((tmp_path / "grid_before_repair_validation.json").read_text())["legal"] is False
    result = {"grid_plan": repaired, "grid_task_candidate": np.array([2, 0, 2, 1], np.int32),
              "candidate_window": np.array([0, 1, 2, 3], np.int32)}
    b.realign_grid_labels(result, keep)
    assert result["grid_task_candidate"].tolist() == [0, 2]
    assert result["grid_selected"].tolist() == [True, False, True, False]
    assert result["grid_selected_count"].tolist() == [1, 0, 1, 0]
    assert result["grid_selection_rank"].tolist() == [0, -1, 1, -1]
    assert result["grid_start_s"].tolist() == [300, -1, 172950, -1]
    assert result["grid_duration_s"].tolist() == [240, -1, 120, -1]
    assert np.array_equal(original.start_s, [0, 300, 172950, 173250])


def test_structural_failure_stops_without_deleting_tasks(tmp_path, monkeypatch):
    plan = Schedule(np.zeros(2, np.int64), np.arange(2), np.zeros(2, np.int64), np.full(2, 120))
    calls = []

    def invalid_structure(cfg, ds, sch, step_s):
        calls.append(sch)
        return fake_report(len(sch), (0, 1), geometry=False)

    monkeypatch.setattr(b, "validate", invalid_structure)
    with pytest.raises(ValueError, match="structural failure"):
        b.checked_plan(Config(), None, plan, tmp_path, "grid")
    assert len(calls) == 1
    assert (tmp_path / "grid_before_repair.npz").exists()
    assert not (tmp_path / "grid_geometry.npz").exists()


def test_inventory_paths_are_relative_and_payload_hashes_match(tmp_path):
    payloads = {"catalogs/cat-000/targets.npz": b"catalog", "scenes/a/input/轨道.txt": b"tle",
                "scenes/a/failure.json": b'{"retained":true}'}
    for relative, data in payloads.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache/de421.bsp").write_bytes(b"not a release payload")
    (tmp_path / "manifest.json").write_text("{}")
    inventory = b.file_manifest(tmp_path)
    assert {row["path"] for row in inventory} == set(payloads)
    for row in inventory:
        assert not row["path"].startswith("/") and ".." not in row["path"].split("/")
        assert str(tmp_path) not in json.dumps(row)
        assert row["bytes"] == len(payloads[row["path"]])
        assert row["sha256"] == b.sha256(tmp_path / row["path"])


def test_scene_keeps_grid_and_final_provenance_when_different_rows_are_rejected(small_catalog, tmp_path, monkeypatch):
    import numba
    spec, ds, table = small_catalog
    cat = tmp_path / "catalogs" / spec["catalog_id"]
    cat.mkdir(parents=True)
    np.savez_compressed(cat / "targets.npz", **table)
    synthetic_windows(ds.n_targets).save(cat / "windows.npz")
    original = Schedule(np.zeros(3, np.int64), np.array([2, 0, 1]),
                        np.array([0, 300, 600]), np.array([150, 240, 180]))
    final = Schedule(original.station.copy(), original.target.copy(),
                     original.start_s + 1, original.dur_s - 1)

    def teacher_stub(pb):
        assert numba.get_num_threads() == 1
        return dict(features=np.ones((4, 20), np.float32),
                    candidate_window=np.arange(4, dtype=np.int32),
                    candidate_duration_cells=np.array([8, 6, 5, 4], np.int16),
                    rule_score_uncovered=np.ones(4), rule_score_covered=np.ones(4),
                    grid_task_candidate=np.array([2, 0, 1], np.int32),
                    grid_selected=np.array([True, True, True, False]),
                    grid_selected_count=np.array([1, 1, 1, 0], np.int32),
                    grid_selection_rank=np.array([1, 2, 0, -1], np.int32),
                    grid_start_s=np.array([300, 600, 0, -1]),
                    grid_duration_s=np.array([240, 180, 150, -1]),
                    grid_plan=original, final_plan=final, metadata={}, timings={})

    def checked_stub(cfg, ds, plan, path, name):
        keep = np.array([False, True, True]) if name == "grid" else np.array([True, False, True])
        retained = Schedule(plan.station[keep], plan.target[keep], plan.start_s[keep], plan.dur_s[keep])
        return retained, dict(legal=True, geometry_checked=True, removed_tasks=1, wall_s=0.), keep

    monkeypatch.setattr(b, "build_teacher", teacher_stub)
    monkeypatch.setattr(b, "checked_plan", checked_stub)
    previous_threads = numba.get_num_threads()
    outcome = b.build_scene(tmp_path, spec, 0)
    assert numba.get_num_threads() == previous_threads
    assert outcome["status"] == "complete"
    path = tmp_path / "scenes" / f"{spec['catalog_id']}-v0"
    with np.load(path / "grid_plan.npz", allow_pickle=False) as grid:
        assert grid["candidate_id"].tolist() == [0, 1]
        assert grid["source_grid_row"].tolist() == [1, 2]
        assert grid["start_s"].tolist() == [300, 600]
    with np.load(path / "plan.npz", allow_pickle=False) as out:
        assert out["candidate_id"].tolist() == [2, 1]
        assert out["source_grid_row"].tolist() == [0, 2]
        assert out["start_s"].tolist() == [1, 601]
    with np.load(path / "actions.npz", allow_pickle=False) as actions:
        assert actions["grid_task_candidate"].tolist() == [0, 1]
        assert actions["grid_selected"].tolist() == [True, True, False, False]
    metadata = (path / "scene.json").read_text()
    assert str(tmp_path) not in metadata and "/Users/" not in metadata
    assert outcome["scene"]["scene_numba_threads"] == 1
    assert not (tmp_path / "events.jsonl").exists()  # parent owns shared files
    assert not (tmp_path / "manifest.json").exists()


def test_parallel_collector_records_crash_and_still_collects_every_submitted_variant(tmp_path, monkeypatch):
    submitted, collected = [], []

    class Future:
        def __init__(self, variant):
            self.variant = variant

        def result(self):
            collected.append(self.variant)
            if self.variant == 1:
                raise RuntimeError(f"explicit worker crash at {tmp_path}/scene")
            return {"status": "complete", "scene": {"scene_id": f"cat-test-v{self.variant}"}}

    class Executor:
        def __init__(self, *, max_workers, mp_context):
            assert max_workers == 4 and mp_context.get_start_method() == "spawn"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, root, spec, variant):
            assert function is b.build_scene
            submitted.append(variant)
            return Future(variant)

    monkeypatch.setattr(b, "ProcessPoolExecutor", Executor)
    results = list(b._scene_results(tmp_path, {"catalog_id": "cat-test"}, list(range(4)), 4))
    assert submitted == collected == [0, 1, 2, 3]
    assert [r["status"] for r in results] == ["complete", "failed", "complete", "complete"]
    failure = results[1]["failure"]
    assert str(tmp_path) not in failure["error"]
    assert "<dataset>" in failure["error"]
    assert (tmp_path / "scenes/cat-test-v1/orchestration_failure.json").exists()


def test_manifest_freezes_generation_code_dependencies_and_parallel_identity(tmp_path, monkeypatch):
    # No scenarios: exercise the bookkeeping boundary without solving or loading ephemerides.
    monkeypatch.setattr(b, "catalog_specs", lambda: [])
    ephemeris = tmp_path / "ephemeris.bsp"
    ephemeris.write_bytes(b"explicit metadata-only ephemeris fixture")
    root = tmp_path / "corpus"
    b.build(root, ephemeris, scene_workers=4)
    manifest = json.loads((root / "manifest.json").read_text())
    assert {"numba", "ortools"} <= manifest["dependencies"].keys()
    assert manifest["identity"]["scene_workers"] == 4
    assert manifest["identity"]["scene_numba_threads"] == 1
    paths = {row["path"] for row in manifest["generation_code_files"]}
    assert "obsplan/training_data/build.py" in paths
    assert "obsplan/lagrange.py" in paths
    assert not any(path.endswith(("loader.py", "audit.py", "package.py", "download.py")) for path in paths)
    assert str(tmp_path) not in json.dumps(manifest)
    b.build(root, ephemeris, scene_workers=4, resume=True)
    with pytest.raises(ValueError, match="resume protocol/config/code/dependency mismatch"):
        b.build(root, ephemeris, scene_workers=2, resume=True)
    real_sha256 = b.sha256
    monkeypatch.setattr(b, "sha256", lambda p: "0" * 64 if p.name == "build.py" else real_sha256(p))
    with pytest.raises(ValueError, match="resume protocol/config/code/dependency mismatch"):
        b.build(root, ephemeris, scene_workers=4, resume=True)
