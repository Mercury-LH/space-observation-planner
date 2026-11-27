import json
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from obsplan.api import create_app
from obsplan.jobs import Conflict, QueueFull, Store
from obsplan.worker import publish, work_one

REQUEST = {"date": "2026-07-31", "days": 1, "stations": [1, 2], "algo": "H"}


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        client.get("/api/config")
        yield client


@pytest.mark.parametrize("patch", [{"date": "2026-07-30"}, {"date": "2026-08-06", "days": 2},
                                    {"days": 0}, {"days": 1.0}, {"days": True}, {"stations": []},
                                    {"stations": [1, 1]}, {"stations": [0]}, {"stations": [11]},
                                    {"stations": [True]}, {"algo": "X"}, {"cache_dir": "/tmp"}])
def test_request_rejected_before_computation(client, patch):
    response = client.post("/api/runs", json={**REQUEST, **patch}, headers={"Idempotency-Key": "invalid-test"})
    assert response.status_code == 422
    assert client.get("/api/runs").json() == {"runs": []}


def test_scope_boundaries_canonicalization_and_idempotency(client):
    body = {**REQUEST, "date": "2026-08-06", "stations": [2, 1]}
    first = client.post("/api/runs", json=body, headers={"Idempotency-Key": "boundary-test"})
    assert first.status_code == 202 and first.json()["request"]["stations"] == [1, 2]
    repeated = client.post("/api/runs", json={**body, "stations": [1, 2]}, headers={"Idempotency-Key": "boundary-test"})
    assert repeated.status_code == 200 and repeated.json()["id"] == first.json()["id"]
    assert client.post("/api/runs", json=REQUEST, headers={"Idempotency-Key": "boundary-test"}).status_code == 409
    rid = first.json()["id"]
    assert client.post(f"/api/runs/{rid}/validate").status_code == 409
    assert client.get(f"/api/runs/{rid}/result").status_code == 409
    assert client.get(f"/api/runs/{rid}/files/submission").status_code == 409


def test_sessions_cannot_read_each_others_results(client):
    rid = client.post("/api/runs", json=REQUEST, headers={"Idempotency-Key": "privacy-test"}).json()["id"]
    with TestClient(client.app) as other:
        assert other.get(f"/api/runs/{rid}").status_code == 404
        assert other.get(f"/api/runs/{rid}/files/submission").status_code == 404
        assert other.post(f"/api/runs/{rid}/validate").status_code == 404
    assert client.post("/api/runs", json=REQUEST, headers={"Origin": "https://other.invalid", "Idempotency-Key": "origin-test"}).status_code == 403


def test_concurrent_duplicates_and_capacity(tmp_path):
    store = Store(tmp_path, capacity=2)
    with ThreadPoolExecutor(8) as pool:
        result = list(pool.map(lambda _: store.submit("session", "duplicate", REQUEST), range(20)))
    assert len({item[0]["id"] for item in result}) == 1
    assert sum(created for _, created in result) == 1
    store.submit("session", "second", REQUEST)
    with pytest.raises(QueueFull):
        store.submit("session", "third", REQUEST)
    with pytest.raises(Conflict):
        store.submit("session", "duplicate", {**REQUEST, "algo": "B"})


def test_recovery_and_validation_retry_preserve_plan(tmp_path):
    store = Store(tmp_path)
    first, _ = store.submit("owner", "first", REQUEST)
    job = store.claim()
    Store(tmp_path).recover()
    assert store.get(first["id"])["state"] == "failed"
    second, _ = store.submit("owner", "second", REQUEST)
    job = store.claim()
    store.finish(job, revision=1)
    for _ in range(5):
        store.validate(second["id"], "owner")
    assert store.health()["active_jobs"] == 1
    validation = store.claim()
    store.finish(validation, error="ephemeris interrupted")
    row = store.get(second["id"])
    assert row["state"] == "ready" and row["revision"] == 1 and row["validation_state"] == "failed"
    store.validate(second["id"], "owner")
    store.finish(store.claim(), revision=2, verdict="passed")
    store.validate(second["id"], "owner")
    assert store.health()["active_jobs"] == 0


def test_partial_artifacts_not_published_and_failures_tracked(tmp_path, monkeypatch):
    store = Store(tmp_path)
    row, _ = store.submit("owner", "first", REQUEST)

    def failing(folder):
        (folder / "submission.json").write_text("partial")
        raise OSError("disk full")

    with pytest.raises(OSError):
        publish(store, row["id"], 1, failing)
    assert not store.artifact_dir(row["id"], 1).exists()
    assert not list(store.artifact_dir(row["id"], 1).parent.glob(".building-*"))
    monkeypatch.setattr("obsplan.worker.plan", lambda *args: (_ for _ in ()).throw(RuntimeError("failed solver")))
    assert work_one(store)
    assert store.get(row["id"])["state"] == "failed"


def test_downloads_are_immutable_and_revision_bound(client):
    store = client.app.state.store
    rid = client.post("/api/runs", json=REQUEST, headers={"Idempotency-Key": "download-test"}).json()["id"]
    job = store.claim()
    def build(folder):
        (folder / "submission.json").write_text("[]", encoding="utf-8")
        (folder / "result.json").write_text(json.dumps({"summary": {"run_id": rid}}))
    publish(store, rid, 1, build)
    store.finish(job, revision=1)
    files = client.get(f"/api/runs/{rid}/result").json()["files"]
    assert files["submission"].endswith("revision=1")
    response = client.get(files["submission"])
    assert response.json() == [] and rid in response.headers["content-disposition"]
    assert client.get(f"/api/runs/{rid}/files/submission?revision=2").status_code == 409
    assert client.get(f"/api/runs/{rid}/files/unknown").status_code == 404


def test_retry_preserves_orphaned_publication(tmp_path):
    store = Store(tmp_path)
    row, _ = store.submit("owner", "first", REQUEST)
    job = store.claim()
    def build(folder):
        (folder / "submission.json").write_text("[]")
    publish(store, row["id"], 1, build)
    # Simulate crash after rename and before DB publication; retry retains old bytes.
    publish(store, row["id"], 1, build)
    assert len(list(store.artifact_dir(row["id"], 1).parent.glob(".orphan-1-*"))) == 1
    store.finish(job, revision=1)
    with pytest.raises(FileExistsError):
        publish(store, row["id"], 1, build)
