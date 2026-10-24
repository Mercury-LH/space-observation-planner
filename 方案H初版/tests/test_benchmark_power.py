"""Power-profile guard uses synthetic pmset outputs; no scheduling experiment."""
from dataclasses import asdict
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from obsplan import benchmark as b
from obsplan import benchmark_worker as worker
from obsplan.config import Config
from obsplan.schedule import Schedule

SETTINGS = """Battery Power:
 lowpowermode         0
 sleep                1
AC Power:
 lowpowermode         1
 sleep                1
"""
AC = {"source": "AC", "low_power_mode": 1}
BATTERY = {"source": "Battery", "low_power_mode": 0}
UNKNOWN = {"source": "unknown", "low_power_mode": "unknown"}


def battery_output(source="AC Power", charge=20):
    return f"Now drawing from '{source}'\n -InternalBattery-0 {charge}%; charging; 20:00 remaining\n"


def test_active_source_selects_its_own_setting_and_ignores_charge():
    assert b._parse_power_profile(battery_output(), SETTINGS) == AC
    assert b._parse_power_profile(battery_output(charge=99), SETTINGS) == AC
    assert b._parse_power_profile(battery_output("Battery Power"), SETTINGS) == BATTERY
    assert b._parse_power_profile(battery_output(), "AC Power:\n sleep 1\n") == {
        "source": "AC", "low_power_mode": "unknown"}
    assert b._parse_power_profile("unexpected output", SETTINGS) == UNKNOWN


def test_supply_change_during_read_is_unknown(monkeypatch):
    monkeypatch.setattr(b.platform, "system", lambda: "Darwin")
    responses = iter([battery_output(), SETTINGS, battery_output("Battery Power")])
    monkeypatch.setattr(b.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=next(responses)))
    assert b.read_power_profile() == UNKNOWN


@pytest.mark.parametrize("failure", [FileNotFoundError(), subprocess.TimeoutExpired("pmset", 2), None])
def test_unavailable_power_command_never_means_off(monkeypatch, failure):
    monkeypatch.setattr(b.platform, "system", lambda: "Darwin")
    def run(*args, **kwargs):
        if failure is not None:
            raise failure
        return SimpleNamespace(returncode=1, stdout="")
    monkeypatch.setattr(b.subprocess, "run", run)
    assert b.read_power_profile() == UNKNOWN


def test_non_mac_does_not_run_pmset(monkeypatch):
    monkeypatch.setattr(b.platform, "system", lambda: "Linux")
    monkeypatch.setattr(b.subprocess, "run", lambda *a, **kw: pytest.fail("pmset must not run"))
    assert b.read_power_profile() == UNKNOWN


def test_power_profile_is_part_of_environment_fingerprint(monkeypatch):
    monkeypatch.setattr(b.platform, "system", lambda: "Linux")
    monkeypatch.setattr(b, "read_power_profile", lambda: AC)
    first = b.environment_identity()
    monkeypatch.setattr(b, "read_power_profile", lambda: BATTERY)
    second = b.environment_identity()
    assert first["power_profile"] == AC
    assert b.digest(first) != b.digest(second)


@pytest.mark.parametrize("profiles,changed", [
    ((AC, AC, AC), False),
    ((AC, BATTERY, AC), True),  # A later return does not erase the observed change.
    ((AC, AC, BATTERY), True),
    ((AC, {"source": "AC", "low_power_mode": 0}, AC), True),
    ((AC, UNKNOWN, AC), True),
    ((UNKNOWN, UNKNOWN, UNKNOWN), False),
])
def test_worker_preserves_result_but_blocks_changed_environment(tmp_path, monkeypatch, profiles, changed):
    from obsplan import data, visibility, problem, validator, refine
    from obsplan.data import Dataset
    ds = Dataset(names=["private target"], satnum=np.array([1]), line1=["fake1"], line2=["fake2"],
                 priority=np.array([5.0]), priority_label=["高"], st_lon=np.array([0.]),
                 st_lat=np.array([0.]), st_alt_m=np.array([0.]), issues=[])
    monkeypatch.setattr(data, "load", lambda cfg: ds)
    monkeypatch.setattr(visibility, "get_windows", lambda *a: SimpleNamespace(sgp4_failed=np.array([False])))
    monkeypatch.setattr(problem.Problem, "build", lambda *a: SimpleNamespace())
    monkeypatch.setattr(refine, "near_windows", lambda *a: {})
    monkeypatch.setattr(validator, "validate", lambda *a, **kw: {
        "legal": True, "violations": {}, "geometry_checked": True})
    monkeypatch.setattr(worker, "source_identity", lambda *a: {
        "source_hash": "frozen", "source_files": {}, "plugin_source_hash": None, "revision": "test"})
    monkeypatch.setattr(worker, "environment_identity", lambda: {
        "dependencies": {}, "power_profile": profiles[0]})
    later_profiles = iter(profiles[1:])
    monkeypatch.setattr(worker, "read_power_profile", lambda: next(later_profiles))
    sch = Schedule(np.array([0]), np.array([0]), np.array([0]), np.array([120]))
    monkeypatch.setattr(worker, "resolve_callable", lambda ref: lambda *a, **kw: (sch, {}))
    request = {"suite_id": "s", "run_id": "r", "preset": "p", "repeat": 0,
               "seed": 2026, "workers": 1, "entrypoint": "obsplan.portfolio:solve", "spec": {},
               "config": asdict(Config(cache_dir=str(tmp_path / "cache"))), "validate": True}
    b.write_json(tmp_path / "request.json", request)
    b.write_json(tmp_path / "result.json", {"suite_id": "s", "run_id": "r", "preset": "p"})
    assert worker.work(tmp_path / "request.json") == (1 if changed else 0)
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["status"] == ("environment_changed" if changed else "valid")
    assert saved["rankable"] is (not changed)
    assert saved["environment_changed"] is changed
    assert saved["power_profile_checks"] == dict(zip(("start", "after_solver", "after_validation"), profiles))
    assert saved["validation"]["legal"] is True
    assert saved["metrics"]["covered"] == 1
    assert (tmp_path / "plan.csv").exists()
    assert b.aggregate([saved])["methods"][0]["rankable"] is (not changed)


def test_default_weight_is_three_sevenths_and_explicit_weight_is_preserved(tmp_path, monkeypatch):
    assert b.DEFAULT_TIME_WEIGHT == 3 / 7
    assert b.utility(.6, 120) == b.utility(.6, 120, weight=3 / 7)
    assert b.aggregate([])["default_weight"] == 3 / 7
    assert b.aggregate([], weight=.2)["default_weight"] == .2
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "execute_run", lambda *a: {"status": "unvalidated", "timing_s": {"total": 0}})
    monkeypatch.setattr(b, "report_directory", lambda folder: json.loads((folder / "manifest.json").read_text()))
    suite = tmp_path / "suite.json"
    b.write_json(suite, {"presets": ["hplus_fast"]})
    args = SimpleNamespace(suite=str(suite), output=str(tmp_path / "outputs" / "run"),
                           config=None, presets=None, repeats=None, workers=1, timeout_s=None, no_validate=True)
    assert b.run_suite(args)["time_weight"] == 3 / 7
