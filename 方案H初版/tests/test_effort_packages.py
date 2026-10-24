"""Folder entry points keep the common solver, evidence and validation contract."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from obsplan.effort import EFFORT_ORDER, get_effort, spec_for_effort

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "计算档位"
spec = importlib.util.spec_from_file_location("effort_package_runner", PACKAGES / "runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("level", EFFORT_ORDER)
def test_every_folder_runs_show_config_from_another_directory(level, tmp_path):
    result = subprocess.run([sys.executable, str(PACKAGES / level / "run.py"),
                             "--show-config", "--workers", "1", "--seed", "17"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    suite = json.loads(result.stdout)
    name = get_effort(level).benchmark_name
    assert suite["effort_level"] == level
    assert suite["presets"] == [name]
    assert suite["solvers"][name]["spec"] == spec_for_effort(level, seed=17, workers=1)
    assert suite["config"]["seed"] == 17
    assert (PACKAGES / level / "README.md").is_file()
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "SCHEME_DIR", tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid arguments or --show-config must not start benchmark")
    monkeypatch.setattr(runner.benchmark, "main", forbidden)
    return tmp_path


def test_show_config_does_not_create_output_or_start_benchmark(isolated, capsys):
    assert runner.main("high", ["--show-config", "--output", "outputs/not-created"]) == 0
    suite = json.loads(capsys.readouterr().out)
    assert suite["effort_level"] == "high"
    assert not (isolated / "outputs").exists()


@pytest.mark.parametrize("exit_code", [0, 2])
def test_dispatch_preserves_exit_status_validation_and_full_suite(isolated, monkeypatch, exit_code):
    called = {}
    def benchmark_main(argv):
        called["argv"] = argv
        called["suite_path"] = Path(argv[argv.index("--suite") + 1])
        called["suite"] = json.loads(called["suite_path"].read_text())
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir(parents=True)
        return exit_code
    monkeypatch.setattr(runner.benchmark, "main", benchmark_main)
    assert runner.main("max", ["--output", "outputs/max-test", "--workers", "1",
                                "--repeats", "2"]) == exit_code
    output = isolated / "outputs/max-test"
    assert called["argv"][0] == "run"
    assert called["argv"][-1] == str(output)
    assert "--no-validate" not in called["argv"]
    assert called["suite"]["repeats"] == 2
    assert called["suite"]["solvers"]["hplus_300_full"]["entrypoint"] == "obsplan.portfolio:solve"
    assert json.loads((output / "source-suite.json").read_text()) == called["suite"]
    assert not called["suite_path"].exists()  # only the temporary assembly file is removed


@pytest.mark.parametrize("argv", [
    [], ["--output", "outside"], ["--output", "outputs/../../escape"],
    ["--show-config", "--workers", "0"], ["--show-config", "--seed", "-1"],
    ["--show-config", "--seed", str(2**31 - 1), "--repeats", "2"],
    ["--show-config", "--repeats", "0"], ["--show-config", "--timeout-s", "nan"],
    ["--show-config", "--timeout-s", "0"], ["--show-config", "--time-weight", "1.1"],
    ["--show-config", "--time-scale-s", "inf"], ["--no-validate"], ["--presets", "fusion_baseline"],
])
def test_invalid_or_bypass_arguments_fail_before_execution(isolated, argv):
    with pytest.raises(SystemExit) as caught:
        runner.main("instant", argv)
    assert caught.value.code == 2


def test_existing_output_is_not_overwritten(isolated):
    output = isolated / "outputs/previous"
    output.mkdir(parents=True)
    proof = output / "summary.json"
    proof.write_text('{"original":true}')
    with pytest.raises(SystemExit):
        runner.main("instant", ["--output", "outputs/previous"])
    assert proof.read_text() == '{"original":true}'


def test_output_symlink_escape_is_rejected(isolated, tmp_path):
    (isolated / "outputs").mkdir()
    (isolated / "outputs/escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(SystemExit):
        runner.main("instant", ["--output", "outputs/escape/new-run"])
