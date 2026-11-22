#!/usr/bin/env python3
"""（沿用方案E初版的实现）Run one local Famou evaluator without automatic dependency installation.

Use the pinned SDK environment:
  .venv-famou/bin/python scripts/famou_worker.py --experiment-id EXP_ID

The SDK is imported only by main(), so policy tests do not need the optional SDK.
This worker controls lifecycle and dependencies; candidate isolation is provided
by integrations/famou/evaluator.py.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
STOP_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "PAUSED"})
# DRAFT is returned by live CreateExperiment and documented in the v2.5
# handbook, although the SDK's ExperimentStatus constants omit it. Keep
# assigning in initial states: assign also registers an available evaluator.
BOOTSTRAP_STATUSES = frozenset({"DRAFT", "CREATED"})
WAIT_STATUSES = frozenset({"PAUSING"})


class SafeWorkerMixin:
    """Add dependency policy and bounded status polling to SDK EvaluatorProxy."""

    def __init__(self, *args, status_interval=30.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.status_interval = status_interval
        self.cloud_status = None
        self.stop_reason = None
        self.rejected_dependency_tasks = 0
        self.status_errors = 0

    async def _process_task(self, program_path, dependencies, timeout):
        # The SDK otherwise invokes unrestricted `pip install` in this process
        # before our sandboxed evaluator runs. Reject even installed packages:
        # the experiment contract is Python standard library only.
        if dependencies:
            self.rejected_dependency_tasks += 1
            return "FAILED", {
                "eval_time": 0.0,
                "metrics": {
                    "combined_score": 0.0,
                    "validity": 0,
                    "error_info": (
                        "Candidate rejected: dependencies must be empty. "
                        "This worker never installs cloud-specified packages."
                    ),
                },
            }
        return await super()._process_task(program_path, [], timeout)

    async def _refresh_cloud_status(self):
        response = self._client.experiment_info(self._config.experiment_id)
        if response.data.cloud_type != "hybrid":
            raise ValueError("The selected experiment is not a hybrid experiment")
        status = response.data.status
        self.cloud_status = status
        self.status_errors = 0
        if status in STOP_STATUSES:
            self.stop_reason = status
            self._running = False
            print(f"[INFO] Cloud experiment status: {status}; draining local tasks.", flush=True)
        elif status not in WAIT_STATUSES | BOOTSTRAP_STATUSES and status != "RUNNING":
            self.stop_reason = "UNKNOWN_CLOUD_STATUS"
            self._running = False
            raise ValueError("Unrecognized cloud experiment status")

    async def _monitor_cloud_status(self):
        while self._running:
            await asyncio.sleep(self.status_interval)
            if not self._running:
                break
            try:
                await self._refresh_cloud_status()
            except Exception as exc:
                self.status_errors += 1
                print(f"[WARNING] Cloud status check failed ({self.status_errors}/3): {exc}", flush=True)
                if self.status_errors >= 3:
                    self.stop_reason = "STATUS_CHECK_FAILED"
                    self._running = False

    async def _pull_programs(self, experiment_id, max_count):
        if not self._running or self.cloud_status in WAIT_STATUSES:
            return []
        try:
            return await super()._pull_programs(experiment_id, max_count)
        except SystemExit:
            # SDK 2.5 uses exit(0) on ExperimentNotRunning. Drain/finally flush
            # metrics instead of letting it interrupt cleanup, and do not label
            # that response as successful completion.
            if self.cloud_status in BOOTSTRAP_STATUSES:
                # A newly created experiment may not have a queue ready yet.
                # The independent status monitor still detects failure/finish.
                return []
            self.stop_reason = "SERVER_NOT_RUNNING"
            self._running = False
            return []

    async def _main_loop(self):
        await self._refresh_cloud_status()
        monitor = asyncio.create_task(self._monitor_cloud_status())
        try:
            await super()._main_loop()
        finally:
            monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor


async def run_worker(proxy):
    """Use the SDK's explicit start/stop API; its __aexit__ is not asynchronous."""
    try:
        await proxy.start()
    finally:
        # Base _main_loop drains active tasks first. stop() cancels heartbeat;
        # wait for its final flush, then retry any remaining results explicitly.
        proxy.stop()
        heartbeat = proxy._heartbeat_task
        if heartbeat is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
        if proxy.evaluations_info:
            await proxy._push_heartbeat_or_result()


class RedactingTee:
    def __init__(self, stream, logfile, key):
        self.stream = stream
        self.logfile = logfile
        self.key = key
        self.pattern = re.compile(r"bce-v\d+/[^\s\"'<>]+")

    def write(self, value):
        if self.key:
            value = value.replace(self.key, "[REDACTED]")
        value = self.pattern.sub("[REDACTED]", value)
        self.stream.write(value)
        self.logfile.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.stream.flush()
        self.logfile.flush()

    def isatty(self):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--evaluator-path", type=Path, default=ROOT / "integrations/famou/evaluator.py")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--status-interval", type=float, default=30.0)
    parser.add_argument("--max-concurrent", type=int, default=1)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", args.experiment_id):
        parser.error("experiment-id must contain only letters, numbers, underscore or hyphen")
    if not 1 <= args.poll_interval <= 60 or not 1 <= args.status_interval <= 60:
        parser.error("poll and status intervals must be between 1 and 60 seconds")
    evaluator_path = args.evaluator_path.resolve(strict=True)
    if not evaluator_path.is_file():
        parser.error("evaluator-path must be a file")

    # Import the existing loader without ever printing the local credential.
    from famou_ctl import load_environment
    env = load_environment()
    key = env.get("FAMOU_API_KEY", "")
    if not key:
        parser.error("FAMOU_API_KEY is missing; configure the ignored .env.famou file")
    os.environ.update(env)
    work_dir = ROOT / "outputs/famou_worker" / args.experiment_id
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    start = time.monotonic()
    proxy = None
    code = 1
    with (work_dir / "worker.log").open("a", encoding="utf-8") as log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = RedactingTee(stdout, log, key)
        sys.stderr = RedactingTee(stderr, log, key)
        try:
            from famou_sdk.cli.utils import get_client_from_env
            from famou_sdk.evaluator_proxy import EvaluatorProxy, ProxyConfig

            class SafeEvaluatorProxy(SafeWorkerMixin, EvaluatorProxy):
                pass

            client = get_client_from_env()
            config = ProxyConfig(
                experiment_id=args.experiment_id,
                max_concurrent=args.max_concurrent,
                poll_interval=args.poll_interval,
            )
            proxy = SafeEvaluatorProxy(
                evaluator_file=str(evaluator_path), client=client, config=config,
                status_interval=args.status_interval,
            )
            asyncio.run(run_worker(proxy))
            # Re-read final state: normal worker exit alone is not completion.
            response = client.experiment_info(args.experiment_id)
            proxy.cloud_status = response.data.status
            print(f"[INFO] Worker stopped; cloud status is {proxy.cloud_status}.", flush=True)
            code = 0 if proxy.cloud_status in {"COMPLETED", "PAUSED", "CANCELLED"} else 1
            if proxy.evaluations_info:
                print("[ERROR] Some local results have not been acknowledged by the server.", flush=True)
                code = 1
        except KeyboardInterrupt:
            code = 130
        except Exception as exc:
            print(f"[ERROR] Worker failed: {exc}", file=sys.stderr, flush=True)
            code = 1
        finally:
            status = {
                "experiment_id": args.experiment_id,
                "cloud_status": getattr(proxy, "cloud_status", None),
                "stop_reason": getattr(proxy, "stop_reason", None),
                "rejected_dependency_tasks": getattr(proxy, "rejected_dependency_tasks", 0),
                "unacknowledged_results": len(getattr(proxy, "evaluations_info", {})),
                "elapsed_s": time.monotonic() - start,
                "exit_code": code,
                "evaluator_path": str(evaluator_path),
            }
            (work_dir / "worker_status.json").write_text(
                json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            sys.stdout, sys.stderr = stdout, stderr
    return code


if __name__ == "__main__":
    raise SystemExit(main())
