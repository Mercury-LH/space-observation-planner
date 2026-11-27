"""同源 API。运行：uvicorn obsplan.api:app --host 127.0.0.1 --port 8000 --workers 1。"""
from __future__ import annotations

import json
import os
import re
import secrets
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from .config import SCHEME_DIR
from .jobs import Conflict, QueueFull, Store, public_run
from .submission import START, END, END_EXCLUSIVE, POLICY, utc


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: Date = Date(2026, 7, 31)
    days: StrictInt = Field(default=7, ge=1, le=7)
    stations: list[StrictInt] = Field(default_factory=lambda: list(range(1, 11)), min_length=1, max_length=10)
    algo: Literal["H", "E", "B"] = "H"

    @model_validator(mode="after")
    def bounds(self):
        start = datetime.combine(self.date, datetime.min.time(), tzinfo=timezone.utc)
        if start < START or start + timedelta(days=self.days) > END_EXCLUSIVE:
            raise ValueError("规划日期和天数必须完全位于 2026-07-31 至 2026-08-06")
        if len(set(self.stations)) != len(self.stations) or any(s < 1 or s > 10 for s in self.stations):
            raise ValueError("站点必须为 1 至 10 的不重复整数")
        self.stations.sort()
        return self


def create_app(root=None):
    store = Store(root or os.environ.get("OBSPLAN_STORE", SCHEME_DIR / "outputs" / "service"))
    app = FastAPI(title="Observation Planner", version="1.0.0")
    app.state.store = store

    @app.middleware("http")
    async def session(request: Request, call_next):
        # Native same-origin UI; no permissive CORS, and reject cross-site writes.
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if (origin and urlsplit(origin).netloc != request.headers.get("host")) or \
                    request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "不接受跨站请求"}, status_code=403)
            if int(request.headers.get("content-length", "0") or 0) > 16384:
                return JSONResponse({"detail": "请求过大"}, status_code=413)
        sid = request.cookies.get("obsplan_session", "")
        fresh = not re.fullmatch(r"[A-Za-z0-9_-]{43}", sid)
        if fresh:
            sid = secrets.token_urlsafe(32)
        request.state.owner = sid
        response = await call_next(request)
        if fresh:
            response.set_cookie("obsplan_session", sid, httponly=True, samesite="strict", max_age=30 * 86400,
                                secure=os.environ.get("OBSPLAN_SECURE_COOKIE") == "1", path="/api")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(Conflict)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(QueueFull)
    async def full(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=429, headers={"Retry-After": "10"})

    def owned(rid, request):
        try:
            return store.get(rid, request.state.owner)
        except KeyError:
            raise HTTPException(404, "找不到本会话的运行记录") from None

    @app.get("/api/health")
    def health():
        return {"status": "ok", "rules": POLICY["version"], **store.health()}

    @app.get("/api/config")
    def config():
        return {"start": utc(START), "end": utc(END), "algorithms": ["H", "E", "B"],
                "defaults": PlanRequest().model_dump(mode="json"), "policy": POLICY}

    @app.post("/api/runs", status_code=202)
    def submit(body: PlanRequest, request: Request, idempotency_key: str = Header(min_length=8, max_length=128)):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", idempotency_key):
            raise HTTPException(422, "提交标识只能包含字母、数字、下划线和连字符")
        row, created = store.submit(request.state.owner, idempotency_key, body.model_dump(mode="json"))
        return JSONResponse({**public_run(row), "created_new": created}, status_code=202 if created else 200)

    @app.get("/api/runs")
    def listing(request: Request):
        return {"runs": [public_run(row) for row in store.list(request.state.owner)]}

    @app.get("/api/runs/{rid}")
    def status(rid: str, request: Request):
        return {**public_run(owned(rid, request)), "worker_online": store.health()["worker_online"]}

    @app.post("/api/runs/{rid}/validate", status_code=202)
    def validate_run(rid: str, request: Request):
        owned(rid, request)
        return public_run(store.validate(rid, request.state.owner))

    def artifact(row, filename, revision=None):
        revision = row["revision"] if revision is None else revision
        if revision < 1 or revision > row["revision"]:
            raise HTTPException(409, "此结果版本尚未就绪")
        path = store.artifact_dir(row["id"], revision) / filename
        if not path.is_file():
            raise HTTPException(404, "此版本没有该文件")
        return path, revision

    @app.get("/api/runs/{rid}/result")
    def result(rid: str, request: Request):
        row = owned(rid, request)
        path, revision = artifact(row, "result.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["run"] = public_run(row)
        data["files"] = {key: f"/api/runs/{rid}/files/{key}?revision={revision}"
                         for key in ("submission", "csv", "summary", "validation", "forecast")
                         if (path.parent / FILENAMES[key]).exists()}
        return data

    @app.get("/api/runs/{rid}/files/{kind}")
    def download(rid: str, kind: str, request: Request, revision: int | None = None):
        row = owned(rid, request)
        if kind not in FILENAMES:
            raise HTTPException(404, "未知文件类型")
        path, rev = artifact(row, FILENAMES[kind], revision)
        return FileResponse(path, media_type="text/csv; charset=utf-8" if kind == "csv" else "application/json",
                            filename=f"{rid}-v{rev}-{path.name}")

    return app


FILENAMES = {"submission": "submission.json", "csv": "plan.csv", "summary": "summary.json",
             "validation": "validation.json", "forecast": "forecast.json"}
app = create_app()
