"""Independent adapter contract, ranking, geometry, and execution-boundary checks."""
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from space_planner import famou_evaluator as adapter
from space_planner.types import Rules, Station, Window


class ConstantGeometry:
    def __init__(self, elevation=45.0):
        self.angle = elevation
        self.stations = {"A": Station("A", 0, 0)}

    def elevation(self, target_id, station_id, seconds):
        return np.full_like(np.asarray(seconds, dtype=float), self.angle)


@pytest.fixture
def case():
    return adapter.EvaluationCase(
        [Window(target, "A", 0, 600) for target in (1, 2, 3)],
        {1: 5, 2: 3, 3: 2}, Rules(), 600, ConstantGeometry(),
    )


@pytest.fixture
def configured(monkeypatch, case):
    monkeypatch.setattr(adapter, "_load_cases", lambda: [case])
    monkeypatch.setenv("FAMOU_CANDIDATE_TIMEOUT_S", "5")
    return case


@pytest.fixture
def candidate(tmp_path):
    def write(body):
        path = tmp_path / "candidate.py"
        path.write_text(body)
        return str(path)
    return write


def observation(target=1, start=0, end=120):
    return {"target_id": target, "station_id": "A", "start": start, "end": end}


MAC_SANDBOX = pytest.mark.skipif(
    sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file(),
    reason="Requires the supported macOS sandbox",
)


@MAC_SANDBOX
def test_two_candidates_have_correct_distinguishable_scores(configured, candidate):
    empty = adapter.evaluate(candidate("def solve(instance): return []"))
    strong = adapter.evaluate(candidate("def solve(instance): return " + repr([observation()])))
    assert empty["validity"] == strong["validity"] == 1
    assert strong["geometry_checked"] == 1
    assert strong["coverage_count"] == 1
    assert strong["observation_score"] == 3
    assert strong["combined_score"] > empty["combined_score"] == 0
    assert all(type(value) is float and math.isfinite(value) for value in strong.values())


@MAC_SANDBOX
def test_coverage_strictly_precedes_secondary_score(configured, candidate):
    one_full = adapter.evaluate(candidate(
        "def solve(instance): return " + repr([observation(1, 0, 240)])
    ))
    two_low = adapter.evaluate(candidate(
        "def solve(instance): return " + repr([observation(2), observation(3, 150, 270)])
    ))
    assert one_full["observation_score"] > two_low["observation_score"]
    assert two_low["combined_score"] > one_full["combined_score"]


@MAC_SANDBOX
@pytest.mark.parametrize("plan", [
    [observation(1), observation(2, 100, 220)],
    [observation(1, 500, 620)],
    [observation(1), observation(1, 150, 270)],
    [observation(1, 0, 119)],
])
def test_invalid_schedules_cannot_score(configured, candidate, plan):
    result = adapter.evaluate(candidate("def solve(instance): return " + repr(plan)))
    assert result["validity"] == result["combined_score"] == 0
    assert result["error_code"] == 4


@MAC_SANDBOX
def test_geometry_is_checked_independently(monkeypatch, case, candidate):
    case.geometry.angle = 90.0
    monkeypatch.setattr(adapter, "_load_cases", lambda: [case])
    result = adapter.evaluate(candidate("def solve(instance): return " + repr([observation()])))
    assert result["validity"] == result["combined_score"] == 0
    assert result["error_code"] == 5


@MAC_SANDBOX
@pytest.mark.parametrize("body", [
    "def solve(instance): return {'plan': []}",
    "def solve(instance): return [{'target_id': True, 'station_id': 'A', 'start': 0, 'end': 120}]",
    "def solve(instance): return [{'target_id': 1, 'station_id': 'A', 'start': '0', 'end': 120}]",
    "def solve(instance): return [{'target_id': 1, 'station_id': 'A', 'start': 0, 'end': float('nan')}]",
    "def solve(instance): return [{'target_id': 1, 'station_id': 'A', 'start': 0, 'end': 120, 'score': 99999}]",
    "def solve(instance): print('extra output'); return []",
])
def test_malformed_or_nonfinite_result_rejected(configured, candidate, body):
    result = adapter.evaluate(candidate(body))
    assert result["validity"] == result["combined_score"] == 0
    assert result["error_code"] in (1, 3)


@MAC_SANDBOX
def test_timeout_is_zero_score(configured, candidate, monkeypatch):
    monkeypatch.setenv("FAMOU_CANDIDATE_TIMEOUT_S", "0.1")
    result = adapter.evaluate(candidate("def solve(instance):\n    while True: pass\n"))
    assert result["validity"] == result["combined_score"] == 0
    assert result["error_code"] == 2
    assert result["elapsed_s"] < 3


@MAC_SANDBOX
def test_secrets_files_mutation_network_and_children_are_blocked(
    configured, candidate, monkeypatch, tmp_path,
):
    secret = tmp_path / "outside_secret.txt"
    secret.write_text("private")
    monkeypatch.setenv("FAMOU_TEST_SECRET", "not-for-candidate")
    body = """
import os, socket, subprocess
def solve(instance):
    assert 'FAMOU_TEST_SECRET' not in os.environ
    for mode in ('r', 'w'):
        try:
            with open(SECRET_PATH, mode) as stream:
                if mode == 'r': stream.read()
                else: stream.write('changed')
        except PermissionError:
            pass
        else:
            raise AssertionError('outside file access was allowed')
    try:
        connection = socket.socket()
        connection.connect(('127.0.0.1', 9))
    except PermissionError:
        pass
    else:
        raise AssertionError('network access was allowed')
    try:
        subprocess.run(['/usr/bin/true'], check=True)
    except PermissionError:
        pass
    else:
        raise AssertionError('child execution was allowed')
    return []
""".replace("SECRET_PATH", repr(str(secret)))
    result = adapter.evaluate(candidate(body))
    assert result["validity"] == 1
    assert secret.read_text() == "private"


@MAC_SANDBOX
def test_scratch_write_is_rejected(configured, candidate):
    result = adapter.evaluate(candidate(
        "def solve(instance):\n"
        "    try: open('unlimited-new-file', 'w')\n"
        "    except PermissionError: return []\n"
        "    raise AssertionError('scratch write was allowed')\n"
    ))
    assert result["validity"] == 1


@MAC_SANDBOX
def test_seed_candidate_is_deterministic_valid_and_better_than_empty(configured):
    seed = adapter.PROJECT_ROOT / "integrations" / "famou" / "init.py"
    first, second = adapter.evaluate(str(seed)), adapter.evaluate(str(seed))
    assert first["validity"] == second["validity"] == 1
    assert first["coverage_count"] == 3
    assert first["combined_score"] == second["combined_score"] > 0


def test_unsupported_sandbox_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter.sys, "platform", "unsupported")
    with pytest.raises(adapter.CandidateFailure) as error:
        adapter._sandbox_command(tmp_path, tmp_path / "candidate.py", 1)
    assert error.value.code == 6


def test_invalid_timeout_is_infrastructure_error(configured, monkeypatch):
    monkeypatch.setenv("FAMOU_CANDIDATE_TIMEOUT_S", "nan")
    with pytest.raises(ValueError, match="TIMEOUT"):
        adapter.evaluate("unused.py")


def test_strict_schema_rejects_huge_infinite_and_boolean_times(case):
    for value in (float("inf"), True, "0"):
        plan = [observation()]
        plan[0]["start"] = value
        with pytest.raises(adapter.CandidateFailure):
            adapter._observations(plan, case)
