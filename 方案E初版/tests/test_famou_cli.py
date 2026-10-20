"""The upstream CLI's exit code alone is not an API/evaluation success signal."""
import importlib.util
import io
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "famou_ctl.py"
    spec = importlib.util.spec_from_file_location("famou_ctl_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_credentials_are_local_and_environment_takes_precedence(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setenv("FAMOU_API_KEY", "environment-key")
    (tmp_path / ".env.famou").write_text(
        "FAMOU_API_KEY=local-key\nFAMOU_API_URL=https://pro-service.famou.com\n"
    )
    assert module.load_environment()["FAMOU_API_KEY"] == "environment-key"


def test_rejects_unexpected_local_credential_setting(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / ".env.famou").write_text("PYTHONPATH=unexpected\n")
    with pytest.raises(ValueError):
        module.load_environment()


@pytest.mark.parametrize("response,arguments,expected", [
    ('{"success":false,"msg":"authentication failed"}\n', ["account", "info", "--json"], 1),
    ('{"success":true,"data":{"result":{"validity":0}}}\n', ["test", "--json"], 1),
    ('{"success":true,"data":{"result":{"validity":1}}}\n', ["test", "--json"], 0),
    ('{"success":true,"data":[]}\n', ["project", "list", "--json"], 0),
])
def test_wrapper_detects_reported_failures(tmp_path, monkeypatch, response, arguments, expected):
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    binary = tmp_path / ".venv-famou/bin/famou-ctl"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(module.sys, "argv", ["famou_ctl.py", *arguments])
    monkeypatch.setattr(module, "load_environment", lambda: {})

    class FakeProcess:
        stdout = io.StringIO(response)

        def wait(self):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    assert module.main() == expected


def test_wrapper_redacts_key_without_command_line_credentials(tmp_path, monkeypatch, capsys):
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    binary = tmp_path / ".venv-famou/bin/famou-ctl"
    binary.parent.mkdir(parents=True)
    binary.touch()
    fake_key = "bce-v3/synthetic-account/synthetic-secret"
    monkeypatch.setattr(module.sys, "argv", ["famou_ctl.py", "account", "info"])
    monkeypatch.setattr(module, "load_environment", lambda: {"FAMOU_API_KEY": fake_key})

    class FakeProcess:
        stdout = io.StringIO("upstream error: " + fake_key + "\n")

        def wait(self):
            return 1

    def launch(command, **kwargs):
        assert fake_key not in command
        assert kwargs["env"]["FAMOU_API_KEY"] == fake_key
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    assert module.main() == 1
    output = capsys.readouterr().out
    assert fake_key not in output
    assert "[REDACTED]" in output
