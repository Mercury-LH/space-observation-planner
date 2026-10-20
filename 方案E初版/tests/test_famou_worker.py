"""Policy tests do not import or contact the optional cloud SDK."""
import asyncio
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "famou_worker_policy", Path(__file__).resolve().parents[1] / "scripts/famou_worker.py",
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class BaseProxy:
    def __init__(self, status="RUNNING", cloud_type="hybrid"):
        self._running = True
        self._config = SimpleNamespace(experiment_id="exp-test")
        self._client = SimpleNamespace(experiment_info=lambda _: SimpleNamespace(
            data=SimpleNamespace(status=status, cloud_type=cloud_type),
        ))
        self.execution_calls = []
        self.pull_calls = []

    async def _process_task(self, program_path, dependencies, timeout):
        self.execution_calls.append((program_path, dependencies, timeout))
        return "SUCCESS", {"metrics": {"validity": 1}}

    async def _pull_programs(self, experiment_id, max_count):
        self.pull_calls.append((experiment_id, max_count))
        return ["a task"]


class Proxy(worker.SafeWorkerMixin, BaseProxy):
    pass


@pytest.mark.parametrize("deps", [["requests"], ["--extra-index-url"], ["https://example.invalid/pkg.tar.gz"]])
def test_dependencies_are_rejected_before_sdk_install_or_evaluation(deps):
    proxy = Proxy()
    status, result = asyncio.run(proxy._process_task("candidate.py", deps, 30))
    assert status == "FAILED"
    assert result["metrics"]["validity"] == 0
    assert proxy.execution_calls == []
    assert proxy.rejected_dependency_tasks == 1
    assert "example.invalid" not in result["metrics"]["error_info"]


def test_stdlib_candidate_passes_to_sdk_once():
    proxy = Proxy()
    status, _ = asyncio.run(proxy._process_task("candidate.py", [], 30))
    assert status == "SUCCESS"
    assert proxy.execution_calls == [("candidate.py", [], 30)]


@pytest.mark.parametrize("status", ["COMPLETED", "FAILED", "CANCELLED", "PAUSED"])
def test_terminal_or_paused_status_stops_new_assignments(status):
    proxy = Proxy(status=status)
    asyncio.run(proxy._refresh_cloud_status())
    assert not proxy._running
    assert proxy.stop_reason == status
    assert asyncio.run(proxy._pull_programs("exp-test", 1)) == []
    assert proxy.pull_calls == []


@pytest.mark.parametrize("status", ["PAUSING"])
def test_transitional_status_waits_without_asking_for_tasks(status):
    proxy = Proxy(status=status)
    asyncio.run(proxy._refresh_cloud_status())
    assert proxy._running
    assert asyncio.run(proxy._pull_programs("exp-test", 1)) == []
    assert proxy.pull_calls == []


def test_running_status_requests_one_task():
    proxy = Proxy()
    asyncio.run(proxy._refresh_cloud_status())
    assert asyncio.run(proxy._pull_programs("exp-test", 1)) == ["a task"]
    assert proxy.pull_calls == [("exp-test", 1)]


@pytest.mark.parametrize("status", ["DRAFT", "CREATED"])
def test_initial_states_register_worker_and_accept_initial_evaluation(status):
    proxy = Proxy(status=status)
    asyncio.run(proxy._refresh_cloud_status())
    assert asyncio.run(proxy._pull_programs("exp-test", 1)) == ["a task"]
    assert proxy.pull_calls == [("exp-test", 1)]


def test_unknown_status_does_not_continue_execution():
    proxy = Proxy(status="UNEXPECTED")
    with pytest.raises(ValueError, match="Unrecognized"):
        asyncio.run(proxy._refresh_cloud_status())
    assert not proxy._running


def test_public_cloud_is_not_accepted():
    proxy = Proxy(cloud_type="public")
    with pytest.raises(ValueError, match="not a hybrid"):
        asyncio.run(proxy._refresh_cloud_status())


def test_sdk_exit_is_drained_not_represented_as_completed():
    class ExitingBase(BaseProxy):
        async def _pull_programs(self, experiment_id, max_count):
            raise SystemExit(0)

    class ExitingProxy(worker.SafeWorkerMixin, ExitingBase):
        pass

    proxy = ExitingProxy()
    assert asyncio.run(proxy._pull_programs("exp-test", 1)) == []
    assert not proxy._running
    assert proxy.stop_reason == "SERVER_NOT_RUNNING"
    assert proxy.cloud_status != "COMPLETED"


@pytest.mark.parametrize("status", ["DRAFT", "CREATED"])
def test_initial_state_without_queue_retries_instead_of_exiting(status):
    class ExitingBase(BaseProxy):
        async def _pull_programs(self, experiment_id, max_count):
            raise SystemExit(0)

    class ExitingProxy(worker.SafeWorkerMixin, ExitingBase):
        pass

    proxy = ExitingProxy(status=status)
    asyncio.run(proxy._refresh_cloud_status())
    assert asyncio.run(proxy._pull_programs("exp-test", 1)) == []
    assert proxy._running
    assert proxy.stop_reason is None


def test_worker_cleanup_flushes_pending_results_after_stop():
    events = []

    class CleanupProxy:
        _heartbeat_task = None
        evaluations_info = {"pending": {}}

        async def start(self):
            events.append("start")
            raise RuntimeError("simulated worker failure")

        def stop(self):
            events.append("stop")

        async def _push_heartbeat_or_result(self):
            events.append("flush")

    with pytest.raises(RuntimeError, match="simulated"):
        asyncio.run(worker.run_worker(CleanupProxy()))
    assert events == ["start", "stop", "flush"]


def test_worker_output_redacts_credentials_in_console_and_log():
    stream, log = io.StringIO(), io.StringIO()
    sink = worker.RedactingTee(stream, log, "fake-private-value")
    sink.write("failure: fake-private-value and bce-v3/example/test\n")
    assert stream.getvalue() == log.getvalue()
    assert "fake-private-value" not in stream.getvalue()
    assert "bce-v3/" not in stream.getvalue()
