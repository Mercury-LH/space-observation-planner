"""Synthetic contract checks; these tests never build the competition windows."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from obsplan import benchmark as b
from obsplan import benchmark_worker as worker
from obsplan.config import Config
from obsplan.schedule import Schedule


def result(name="fast", quality=0.6, seconds=120.0, **overrides):
    row = {"suite_id": "new-experiment", "run_id": name, "preset": name,
           "comparison_key": "same-config-input-code", "status": "valid", "rankable": True,
           "fresh_solve": True, "validation": {"legal": True, "step_s": 1.0, "geometry_checked": True},
           "metrics": {"combined": quality, "covered": 3, "score": 10.0},
           "timing_s": {"preprocessing": 1.0, "solver": seconds, "validation": 2.0, "total": seconds + 3}}
    row.update(overrides)
    return row


def test_time_weight_is_fixed_scale_and_quality_not_clipped():
    assert b.utility(0.6, 120, 0.2, 120) == pytest.approx(58.0)
    assert b.utility(1.2, 0, 0, 120) == pytest.approx(120.0)
    assert b.utility(0.6, 120, 0.2, 120) > b.utility(0.6, 240, 0.2, 120)
    for args in [(float("nan"), 1, 0.2, 120), (1, -1, 0.2, 120), (1, 1, 1.1, 120), (1, 1, .2, 0)]:
        with pytest.raises(ValueError):
            b.utility(*args)
    assert "fast" in b.render_report(b.aggregate([result()], weight=0))


@pytest.mark.parametrize("overrides", [
    {"status": "invalid"}, {"status": "timeout"}, {"status": "error"},
    {"status": "unvalidated"}, {"rankable": False}, {"fresh_solve": False},
    {"worker_exit": "worker_error"},
    {"validation": {"legal": False, "step_s": 1.0}},
    {"validation": {"legal": True, "step_s": 30.0}},
    {"validation": {"legal": True, "step_s": 1.0, "geometry_checked": False}},
])
def test_failed_or_unvalidated_cannot_rank(overrides):
    report = b.aggregate([result(**overrides)])
    assert not report["methods"][0]["rankable"]
    assert all(not names for names in report["rankings"].values())
    assert "utility_by_weight" not in report["methods"][0]


def test_one_failed_repeat_excludes_method_and_mixed_evidence_rejected():
    rows = [result(), result(run_id="second", rankable=False, status="timeout")]
    assert not b.aggregate(rows)["methods"][0]["rankable"]
    with pytest.raises(ValueError, match="different"):
        b.aggregate([result(), result("other", comparison_key="different-data")])
    assert len(b.aggregate([result(), result("rerun", suite_id="rerun")])["methods"]) == 2
    with pytest.raises(ValueError, match="parameters"):
        b.aggregate([result(solver_spec_hash="a"), result(run_id="new", solver_spec_hash="b")])


def test_same_preset_cannot_merge_distinct_plugin_entrypoints():
    rows = [result(entrypoint="team_a:solve", solver_spec_hash="same",
                   provenance={"plugin_source_hash": "a"}),
            result(run_id="rerun", suite_id="another-suite", entrypoint="team_b:solve",
                   solver_spec_hash="same", provenance={"plugin_source_hash": "b"})]
    with pytest.raises(ValueError, match="entry points"):
        b.aggregate(rows)


def test_sensitivity_pareto_and_median_of_per_run_scores():
    rows = [result("quality", .7, 1000), result("fast", .65, 10), result("dominated", .6, 200)]
    report = b.aggregate(rows)
    assert report["rankings"]["0.0"][0] == "quality"
    assert report["rankings"]["0.2"][0] == "fast"
    assert {m["preset"] for m in report["methods"] if m["pareto"]} == {"quality", "fast"}
    rows = [result(seconds=t, run_id=str(t)) for t in (0, 120, 1200, 12000)]
    r = b.aggregate(rows)["methods"][0]
    expected = np.median([b.utility(.6, t, weight=.2) for t in (0, 120, 1200, 12000)])
    assert r["utility_by_weight"]["0.2"] == expected


def test_public_config_and_unsupported_config():
    cfg = Config(data_dir="/private/data", cache_dir="/private/cache")
    b.check_config(cfg)
    assert "/private" not in json.dumps(b.public_config(cfg))
    for changed in [cfg.with_(cell_s=10), cfg.with_(w_cov=.4), cfg.with_(score_ref=100), cfg.with_(days=0)]:
        with pytest.raises(ValueError):
            b.check_config(changed)
    assert worker._public_spec({"api_key": "secret", "path": "/Users/private", "seed": 2}) == {
        "api_key": "<redacted>", "path": "<path-or-url>", "seed": 2}


def test_plugin_reference_cannot_be_a_shell_command():
    assert b.resolve_callable("obsplan.benchmark:utility") is b.utility
    for ref in ["echo x;rm -rf x", "a:b()", "$(cat secret):x", "a:b.c"]:
        with pytest.raises(ValueError):
            b.resolve_callable(ref)


def test_schedule_protocol_guards_indices_finite_shape_and_integral_seconds():
    ds = SimpleNamespace(n_stations=2, n_targets=3)
    sch = Schedule(np.array([0]), np.array([1]), np.array([1.0]), np.array([120.0]))
    clean = worker.check_schedule(sch, ds)
    assert clean.start_s.dtype == np.int64
    for name, value in [("target", [-1]), ("station", [2]), ("dur_s", [np.nan]),
                        ("start_s", [1.5]), ("target", [[1]]), ("start_s", [])]:
        bad = deepcopy(sch)
        setattr(bad, name, np.array(value))
        with pytest.raises(ValueError):
            worker.check_schedule(bad, ds)


def test_report_refuses_swapped_old_result(tmp_path):
    b.write_json(tmp_path / "manifest.json", {"suite_id": "new", "time_weight": .2,
        "time_scale_s": 120, "weights": [0, .2], "runs": [{"run_id": "a"}]})
    (tmp_path / "a").mkdir()
    b.write_json(tmp_path / "a" / "result.json", result(run_id="a", suite_id="old"))
    with pytest.raises(ValueError, match="fresh experiment"):
        b.report_directory(tmp_path)


def test_report_combines_verified_reruns_but_not_changed_plugins(tmp_path):
    for suite in ("first", "second"):
        folder = tmp_path / suite
        (folder / "a").mkdir(parents=True)
        b.write_json(folder / "manifest.json", {"suite_id": suite, "time_weight": .2,
            "time_scale_s": 120, "weights": [0, .2], "runs": [{"run_id": "a"}]})
        b.write_json(folder / "a" / "result.json", result(run_id="a", suite_id=suite))
    report = b.report_directory(tmp_path / "first", include=[tmp_path / "second"])
    assert report["methods"][0]["runs"] == 2
    with pytest.raises(ValueError, match="more than once"):
        b.report_directory(tmp_path / "first", include=[tmp_path / "first"])
    with pytest.raises(ValueError, match="plugin code"):
        b.aggregate([result(entrypoint="custom:solve", provenance={"plugin_source_hash": "a"}),
                     result(entrypoint="custom:solve", provenance={"plugin_source_hash": "b"})])


def test_timeout_persists_unrankable_record_and_refuses_overwrite(tmp_path, monkeypatch):
    class TimeoutProcess:
        pid = 999999
        def wait(self, timeout=None):
            if timeout != 3:
                raise subprocess.TimeoutExpired("worker", timeout)
            return -15
    monkeypatch.setattr(b.subprocess, "Popen", lambda *a, **kw: TimeoutProcess())
    monkeypatch.setattr(b.os, "killpg", lambda *a: None)
    request = {"suite_id": "s", "run_id": "r", "preset": "p", "repeat": 0, "seed": 1, "workers": 1}
    directory = tmp_path / "run"
    got = b.execute_run(request, directory, .01)
    assert got["status"] == "timeout" and not got["rankable"]
    assert got["timing_s"]["total"] >= 0
    assert (directory / "result.json").exists()
    with pytest.raises(FileExistsError):
        b.execute_run(request, directory, .01)


@pytest.mark.parametrize("saved_status", ["valid", "unvalidated", "invalid", "error", "environment_changed"])
def test_nonzero_exit_cannot_rank_a_saved_valid_checkpoint(tmp_path, monkeypatch, saved_status):
    directory = tmp_path / "run"

    class FailedAfterSave:
        def wait(self, timeout=None):
            b.write_json(directory / "result.json", result(
                "p", suite_id="s", run_id="r", status=saved_status,
                rankable=saved_status == "valid"))
            return 1

    monkeypatch.setattr(b.subprocess, "Popen", lambda *args, **kwargs: FailedAfterSave())
    request = {"suite_id": "s", "run_id": "r", "preset": "p", "repeat": 0, "seed": 1, "workers": 1}
    got = b.execute_run(request, directory, 1)
    expected = "worker_error" if saved_status in ("valid", "unvalidated") else saved_status
    assert got["status"] == expected
    assert got["worker_exit"] == "worker_error"
    assert got["rankable"] is False
    assert json.loads((directory / "result.json").read_text())["rankable"] is False
    assert not b.aggregate([got])["methods"][0]["rankable"]


def test_worker_contract_and_no_validate_never_ranks(tmp_path, monkeypatch):
    from obsplan import data, visibility, problem, validator, refine
    from obsplan.data import Dataset
    ds = Dataset(names=["private target"], satnum=np.array([1]), line1=["fake1"], line2=["fake2"],
                 priority=np.array([5.0]), priority_label=["高"], st_lon=np.array([0.]),
                 st_lat=np.array([0.]), st_alt_m=np.array([0.]), issues=[])
    monkeypatch.setattr(data, "load", lambda cfg: ds)
    monkeypatch.setattr(visibility, "get_windows", lambda cfg, ds: SimpleNamespace(sgp4_failed=np.array([False])))
    monkeypatch.setattr(problem.Problem, "build", lambda *args: SimpleNamespace())
    monkeypatch.setattr(refine, "near_windows", lambda *args: {})
    calls = []
    def validate(*args, **kwargs):
        calls.append(kwargs)
        return {"legal": True, "violations": {}, "geometry_checked": True}
    monkeypatch.setattr(validator, "validate", validate)
    sch = Schedule(np.array([0]), np.array([0]), np.array([0]), np.array([120]))
    monkeypatch.setattr(worker, "resolve_callable", lambda ref: lambda *a, **kw: (sch, {
        "stage_timings_s": {"source": .1}, "famou": {"invoked": False}}))
    for enabled in (False, True):
        directory = tmp_path / str(enabled)
        directory.mkdir()
        request = {"suite_id": "s", "run_id": str(enabled), "preset": "p", "repeat": 0,
                   "seed": 2026, "workers": 1, "entrypoint": "obsplan.portfolio:solve", "spec": {},
                   "config": asdict(Config(cache_dir=str(tmp_path / "cache"))), "validate": enabled}
        b.write_json(directory / "request.json", request)
        b.write_json(directory / "result.json", {"suite_id": "s", "run_id": str(enabled), "preset": "p"})
        assert worker.work(directory / "request.json") == 0
        saved = json.loads((directory / "result.json").read_text())
        assert saved["status"] == ("valid" if enabled else "unvalidated")
        assert saved["rankable"] is enabled
        assert saved["metrics"]["covered"] == 1
        assert "private target" not in json.dumps(saved)
        assert str(tmp_path) not in json.dumps(saved)
        assert (directory / "plan.csv").exists()
    assert calls == [{"step_s": 1.0}]
