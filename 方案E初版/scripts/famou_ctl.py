#!/usr/bin/env python3
"""Run the pinned Famou CLI with a local, ignored credential file.

From the repository root: python3 scripts/famou_ctl.py account info --json
Credentials are environment variables, never command-line arguments.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_environment() -> dict[str, str]:
    env = os.environ.copy()
    credential_file = ROOT / ".env.famou"
    if credential_file.exists():
        for line in credential_file.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            name, separator, value = line.partition("=")
            if not separator or name not in {"FAMOU_API_URL", "FAMOU_API_KEY"}:
                raise ValueError("Unexpected setting in .env.famou")
            env.setdefault(name, value.strip())
    env.setdefault("FAMOU_API_URL", "https://pro-service.famou.com")
    env.setdefault("FAMOU_API_TIMEOUT", "30")
    return env


def main() -> int:
    executable = ROOT / ".venv-famou/bin/famou-ctl"
    if not executable.exists():
        print("Install the Python 3.12 Famou environment first; see integrations/famou/README.md.",
              file=sys.stderr)
        return 2
    env = load_environment()
    key = env.get("FAMOU_API_KEY", "")
    # Protect logs if an upstream exception happens to contain a credential.
    redact = re.compile(r"bce-v\d+/[^\s\"'<>]+")
    process = subprocess.Popen(
        [str(executable), *sys.argv[1:]], cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    last_json = None
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if key:
                line = line.replace(key, "[REDACTED]")
            line = redact.sub("[REDACTED]", line)
            print(line, end="", flush=True)
            try:
                value = json.loads(line)
                if isinstance(value, dict) and "success" in value:
                    last_json = value
            except ValueError:
                pass
        code = process.wait()
        # SDK 2.5.0 can exit zero while its JSON reports an API error.
        if last_json is not None:
            if last_json.get("success") is False:
                return code or 1
            if sys.argv[1:2] == ["test"]:
                data = last_json.get("data") or {}
                result = data.get("result") or {}
                if result.get("validity") != 1:
                    return code or 1
        return code
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
