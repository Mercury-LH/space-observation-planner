from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pytest

from space_planner import cli
from space_planner.data import Dataset
from space_planner.types import Observation, Rules, Station, Target, Window


def _write_config(tmp_path: Path, **extra) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "data_root": "../开运联合",
        "start_utc": "2026-08-04T00:00:00Z",
        "horizon_s": 600,
        **extra,
    }))
    return path


def _synthetic_dataset():
    epoch = datetime(2026, 8, 4, tzinfo=timezone.utc)
    return Dataset(
        [Target(i, "", "", 5 if i == 1 else 2, epoch) for i in (1, 2)],
        [Station("1", 0, 0)],
        {"valid": True, "files": {"synthetic": {"sha256": "0" * 64}}},
    )


def _mock_geometry(monkeypatch, elevation=30):
    from space_planner import geometry

    class SyntheticGeometry:
        def __init__(self, targets, stations, start):
            self.stations = stations

        def elevation(self, target, station, times):
            return np.full_like(np.asarray(times, dtype=float), elevation)

    monkeypatch.setattr(cli, "load_dataset", lambda _: _synthetic_dataset())
    monkeypatch.setattr(geometry, "Geometry", SyntheticGeometry)
    monkeypatch.setattr(
        geometry, "compute_windows",
        lambda geometry, horizon, **kwargs: (
            [Window(1, "1", 0, horizon), Window(2, "1", 0, horizon)],
            {"synthetic_test": True},
        ),
    )


def test_config_requires_explicit_start_and_timezone(tmp_path):
    path = _write_config(tmp_path)
    data = json.loads(path.read_text())
    del data["start_utc"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="start_utc"):
        cli.load_config(path)
    assert cli.load_config(path, require_start=False)["horizon_s"] == 600
    with pytest.raises(ValueError, match="explicit UTC offset"):
        cli.load_config(_write_config(tmp_path, start_utc="2026-08-04T00:00:00"))


@pytest.mark.parametrize("extra", [
    {"days": 1}, {"target_ids": [1], "target_limit": 1}, {"target_limit": 0},
    {"target_limit": True}, {"seeds": [0, 0]}, {"alns_iterations": -1},
    {"alns_time_limit_s": -1}, {"station_ids": []}, {"typo_parameter": 2},
    {"windows": {"typo_parameter": 2}},
])
def test_invalid_config_is_rejected_before_expensive_work(tmp_path, extra):
    with pytest.raises(ValueError):
        cli.load_config(_write_config(tmp_path, **extra))


def test_relative_paths_and_offsets_are_normalized(tmp_path):
    config = cli.load_config(_write_config(
        tmp_path, start_utc="2026-08-04T08:00:00+08:00",
        windows={"cache_dir": "outputs/window_cache"},
    ))
    assert Path(config["data_root"]) == cli.PROJECT_ROOT.parent / "开运联合"
    assert config["start_utc"] == "2026-08-04T00:00:00+00:00"
    assert config["windows"]["cache_dir"] == str(cli.PROJECT_ROOT / "outputs/window_cache")


def test_selection_is_reproducible_and_unknown_ids_fail():
    dataset = _synthetic_dataset()
    targets, stations = cli.select_dataset(dataset, {"target_limit": 1, "selection_seed": 7})
    reordered = Dataset(dataset.targets[::-1], dataset.stations, dataset.audit)
    assert cli.select_dataset(reordered, {"target_limit": 1, "selection_seed": 7})[0] == targets
    assert stations == dataset.stations
    with pytest.raises(ValueError, match="Unknown target"):
        cli.select_dataset(dataset, {"target_ids": [100]})
    with pytest.raises(ValueError, match="Unknown station"):
        cli.select_dataset(dataset, {"station_ids": ["99"]})


def test_metrics_report_additive_and_unique_scores_separately():
    plan = [Observation(1, "1", 0, 240), Observation(1, "1", 300, 540)]
    rules = Rules(max_observations_per_target=2)
    result = cli.report_metrics(plan, {1: 5, 2: 2}, rules, [Window(1, "1", 0, 600)])
    assert result["observation_score"] == 10
    assert result["unique_best_observation_score"] == 5
    assert result["coverage_fraction"] == 0.5
    assert result["available_coverage_fraction"] == 1
    assert result["optimistic_observation_score_upper_bound"] is None
    assert result["optimistic_unique_best_score_upper_bound"] == 5


@pytest.mark.parametrize("score_mode,upper", [("step", 4.2), ("linear", 4.6)])
def test_optimistic_bounds_use_window_duration_and_configured_scoring(score_mode, upper):
    rules = Rules(score_mode=score_mode)
    result = cli.report_metrics([], {1: 5, 2: 2, 3: 3}, rules, [
        Window(1, "1", 0, 120), Window(2, "1", 0, 180), Window(3, "1", 0, 119),
    ])
    assert result["optimistic_coverage_upper_bound"] == 2
    assert result["optimistic_observation_score_upper_bound"] == pytest.approx(upper)
    assert "not a continuous-geometry" in result["bound_scope"]


def test_run_saves_metadata_plans_and_independent_geometry_validation(tmp_path, monkeypatch):
    _mock_geometry(monkeypatch)
    config = cli.load_config(_write_config(tmp_path, seeds=[0, 1], alns_iterations=3))
    output = tmp_path / "result"
    result = cli.execute("benchmark", config, output)
    assert result["all_plans_valid"]
    assert len(result["algorithms"]) == 3
    assert all(item["validation"]["geometry_checked"] for item in result["algorithms"])
    assert all(item["metrics"]["coverage_count"] == 2 for item in result["algorithms"])
    assert (output / "plans/alns_seed_0.json").is_file()
    greedy = json.loads((output / "plans/greedy.json").read_text())
    diagnostics = greedy["diagnostics"]
    assert diagnostics["requested_strategy_count"] == 5
    assert diagnostics["completed_strategy_count"] == 5
    assert len(diagnostics["strategy_runs"]) == 5
    assert all(item["elapsed_s"] >= 0 for item in diagnostics["strategy_runs"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["versions"]["python"]
    assert manifest["source_code_sha256"]["src/space_planner/cli.py"]
    assert manifest["input_files"]["synthetic"]["sha256"] == "0" * 64
    windows = json.loads((output / "windows.json").read_text())
    assert windows["horizon_s"] == 600
    with pytest.raises(ValueError, match="not empty"):
        cli.execute("audit", config, output)


def test_geometry_failure_is_preserved_and_skip_is_explicit(tmp_path, monkeypatch):
    _mock_geometry(monkeypatch, elevation=90)
    config = cli.load_config(_write_config(tmp_path, seeds=[0], alns_iterations=0))
    failed = cli.execute("run", config, tmp_path / "failed")
    assert not failed["all_plans_valid"]
    manifest = json.loads((tmp_path / "failed/manifest.json").read_text())
    assert manifest["status"] == "validation_failed"
    skipped = cli.execute("run", config, tmp_path / "skipped", skip_geometry=True)
    assert skipped["all_plans_valid"]
    assert not skipped["geometry_checked"]
    assert not skipped["algorithms"][0]["validation"]["geometry_checked"]


def test_audit_does_not_need_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_dataset", lambda _: _synthetic_dataset())
    config = {"data_root": str(tmp_path), "rules": asdict(Rules())}
    result = cli.execute("audit", config, tmp_path / "audit")
    assert result["data_valid"]
    assert not (tmp_path / "audit/windows.json").exists()


def test_validation_numeric_strings_are_normalized_before_execution(tmp_path, monkeypatch):
    _mock_geometry(monkeypatch)
    config = cli.load_config(_write_config(
        tmp_path, seeds=[0], alns_iterations=0,
        validation={
            "time_tolerance_s": "1e-6",
            "elevation_tolerance_deg": "1e-5",
            "geometry_step_s": "5",
        },
    ))
    assert config["validation"] == {
        "time_tolerance_s": 1e-6, "elevation_tolerance_deg": 1e-5, "geometry_step_s": 5.0,
    }
    assert all(isinstance(value, float) for value in config["validation"].values())
    result = cli.execute("run", config, tmp_path / "normalized")
    assert result["all_plans_valid"]


def test_zero_validation_tolerances_are_allowed(tmp_path):
    config = cli.load_config(_write_config(
        tmp_path, validation={"time_tolerance_s": 0, "elevation_tolerance_deg": "0"},
    ))
    assert config["validation"] == {"time_tolerance_s": 0.0, "elevation_tolerance_deg": 0.0}


@pytest.mark.parametrize("validation", [
    {"time_tolerance_s": -1}, {"elevation_tolerance_deg": float("inf")},
    {"time_tolerance_s": "nan"}, {"geometry_step_s": "0"},
    {"geometry_step_s": True}, {"elevation_tolerance_deg": None},
])
def test_invalid_validation_numbers_are_rejected_early(tmp_path, validation):
    with pytest.raises(ValueError):
        cli.load_config(_write_config(tmp_path, validation=validation))
