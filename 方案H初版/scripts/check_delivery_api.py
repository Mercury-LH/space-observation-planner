"""使用真实 HTTP、算法和数据验证规划→校验→下载→严格重新解析。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obsplan.config import Config
from obsplan.data import load
from obsplan.submission import parse_json, to_schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--algos", nargs="+", default=["H", "E", "B"])
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--stations", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--out", default="outputs/api-acceptance")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    cfg = Config(days=args.days)
    ds = load(cfg)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    with httpx.Client(base_url=args.base, timeout=30, trust_env=False) as client:
        def request(method, path, **kwargs):
            # Every POST below is idempotent. A tunnel disconnect must not turn
            # an already accepted job into a duplicate or discard its session.
            deadline = time.monotonic() + args.timeout
            interrupted = False
            while True:
                try:
                    response = client.request(method, path, **kwargs)
                    if interrupted:
                        print("HTTP connection restored", flush=True)
                    return response
                except httpx.TransportError:
                    if time.monotonic() >= deadline:
                        raise
                    if not interrupted:
                        print("HTTP connection interrupted; retaining session and retrying", flush=True)
                    interrupted = True
                    time.sleep(2)

        request("GET", "/api/config").raise_for_status()

        def wait(rid, field, active):
            deadline = time.monotonic() + args.timeout
            last = None
            while time.monotonic() < deadline:
                response = request("GET", f"/api/runs/{rid}")
                response.raise_for_status()
                row = response.json()
                state = (row[field], row["stage"])
                if state != last:
                    print(rid[:8], field, state, flush=True)
                    last = state
                if row[field] not in active:
                    return row
                time.sleep(2)
            raise TimeoutError(rid)

        for algo in args.algos:
            started = time.monotonic()
            body = {"algo": algo, "date": "2026-07-31", "days": args.days, "stations": args.stations}
            headers = {"Idempotency-Key": uuid.uuid4().hex}
            response = request("POST", "/api/runs", json=body, headers=headers)
            response.raise_for_status()
            rid = response.json()["id"]
            again = request("POST", "/api/runs", json=body, headers=headers)
            assert again.status_code == 200 and again.json()["id"] == rid
            run = wait(rid, "state", {"queued", "running"})
            assert run["state"] == "ready", run
            response = request("POST", f"/api/runs/{rid}/validate")
            response.raise_for_status()
            run = wait(rid, "validation_state", {"queued", "running"})
            assert run["validation_state"] == "passed", run
            result_response = request("GET", f"/api/runs/{rid}/result")
            result_response.raise_for_status()
            result = result_response.json()
            dest = output / rid
            dest.mkdir(exist_ok=True)
            for kind, url in result["files"].items():
                content = request("GET", url)
                content.raise_for_status()
                suffix = ".csv" if kind == "csv" else ".json"
                (dest / f"{kind}{suffix}").write_bytes(content.content)
            encoded = (dest / "submission.json").read_bytes()
            rows = parse_json(encoded)
            back = to_schedule(rows, cfg, ds)
            assert len(back) == len(result["tasks"]) == result["summary"]["metrics"]["tasks"]
            assert hashlib.sha256(encoded).hexdigest() == result["summary"]["submission_sha256"]
            report = json.loads((dest / "validation.json").read_bytes())
            assert report["summary"]["clean"] and report["geometry"]["legal"]
            assert report["submission_sha256"] == result["summary"]["submission_sha256"]
            summary = {"id": rid, "request": body, "summary": result["summary"],
                       "wall_seconds": round(time.monotonic() - started, 2), "roundtrip": True}
            summaries.append(summary)
            print(json.dumps({"id": rid, "algo": algo, **report["summary"]}, ensure_ascii=False), flush=True)
            (output / "acceptance.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
