"""Export must preserve the evaluated schedule and reject incomplete results."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from space_planner import famou_evaluator as adapter
from space_planner.types import Rules, Station, Window


spec = importlib.util.spec_from_file_location(
    "famou_export_plan", Path(__file__).resolve().parents[1] / "scripts/famou_export_plan.py",
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ConstantGeometry:
    def __init__(self, angle=45):
        self.angle = angle
        self.stations = {"A": Station("A", 0, 0)}

    def elevation(self, target_id, station_id, seconds):
        return np.full_like(np.asarray(seconds, dtype=float), self.angle)


@pytest.fixture
def setup_export(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text('{"test_fixture":true}')
    (run / "windows.json").write_text('{"test_fixture":true}')
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def solve(instance): return []")
    case = adapter.EvaluationCase(
        [Window(1, "A", 0, 600)], {1: 5}, Rules(), 600, ConstantGeometry(),
    )
    monkeypatch.setattr(adapter, "_load_cases", lambda: [case])
    return run, candidate, case


def test_export_uses_one_candidate_run_and_matches_exact_validated_plan(setup_export, tmp_path, monkeypatch):
    run, candidate, _ = setup_export
    output = tmp_path / "export"
    output.mkdir()  # An existing empty destination is accepted.
    plan = [{"target_id": 1, "station_id": "A", "start": 0.0, "end": 240.0}]
    calls = []

    def solve(source, instance, timeout):
        calls.append(source)
        return plan

    monkeypatch.setattr(adapter, "_run_candidate", solve)
    monkeypatch.setenv("FAMOU_RUN_DIRS", "preserve-this-setting")
    metrics = exporter.export_candidate(candidate, run, output)
    assert len(calls) == 1
    assert metrics["validity"] == metrics["geometry_checked"] == 1
    assert metrics["coverage_count"] == 1
    assert metrics["observation_score"] == 5
    report = json.loads((output / "plans/famou_candidate.json").read_text())
    assert report["plan"] == plan
    assert report["validation"]["valid"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["candidate_sha256"] == hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert manifest["input_manifest"] == {"test_fixture": True}
    assert (output / "candidate.py").read_bytes() == candidate.read_bytes()
    assert exporter.os.environ["FAMOU_RUN_DIRS"] == "preserve-this-setting"


@pytest.mark.parametrize("failure", ["structural", "geometry", "malformed", "timeout"])
def test_rejected_candidate_leaves_no_partial_export(setup_export, tmp_path, monkeypatch, failure):
    run, candidate, case = setup_export
    output = tmp_path / "export"
    output.mkdir()
    plan = [{"target_id": 1, "station_id": "A", "start": 0.0, "end": 240.0}]
    if failure == "structural":
        plan[0]["end"] = 119.0
    if failure == "geometry":
        case.geometry.angle = 90.0
    if failure == "malformed":
        plan = {"fake_score": 100}

    def solve(*_):
        if failure == "timeout":
            raise adapter.CandidateFailure(2, "timeout")
        return plan

    monkeypatch.setattr(adapter, "_run_candidate", solve)
    with pytest.raises(adapter.CandidateFailure):
        exporter.export_candidate(candidate, run, output)
    assert list(output.iterdir()) == []


def test_existing_artifact_is_not_overwritten_or_executed(setup_export, tmp_path, monkeypatch):
    run, candidate, _ = setup_export
    output = tmp_path / "export"
    output.mkdir()
    prior = output / "prior.txt"
    prior.write_text("preserve")

    def forbidden(*_):
        raise AssertionError("Candidate should not execute")

    monkeypatch.setattr(adapter, "_run_candidate", forbidden)
    with pytest.raises(ValueError, match="empty"):
        exporter.export_candidate(candidate, run, output)
    assert prior.read_text() == "preserve"
