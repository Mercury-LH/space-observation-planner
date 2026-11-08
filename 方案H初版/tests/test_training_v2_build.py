"""Fault-injection tests for immutable catalog transactions and recovery."""
import json
from pathlib import Path

import pytest

from obsplan.training_data import v2_build as b


SPEC=dict(catalog_id="v2-cat-000",seed=123,split="train",n_targets=1000,n_stations=4)
IDENTITY={"example":"frozen identity"}
IH=b._hash_object(IDENTITY)


def stage(root):
    job=b._job_path(root,SPEC["catalog_id"])
    c=job/"catalogs"/SPEC["catalog_id"]
    s=job/"scenes"/(SPEC["catalog_id"]+"-v0")
    c.mkdir(parents=True)
    s.mkdir(parents=True)
    (c/"targets.npz").write_bytes(b"immutable catalog")
    (s/"plan.npz").write_bytes(b"immutable plan")
    marker=dict(identity_sha256=IH,spec_sha256=b._hash_object(SPEC),
                catalog=dict(SPEC),scene={"scene_id":SPEC["catalog_id"]+"-v0"},
                files=b.file_manifest(job))
    b.atomic_json(job/b.MARKER,marker)
    return job,marker


def test_staged_inventory_does_not_exclude_parent_cache(tmp_path):
    job,marker=stage(tmp_path)
    assert len(marker["files"])==2
    assert all(not x["path"].startswith(".cache") for x in marker["files"])
    result=b.publish_transaction(tmp_path,SPEC,IH)
    assert result==marker and not job.exists()
    b._check_files(tmp_path,result["files"])


def test_crash_between_catalog_and_scene_moves_is_recoverable(tmp_path,monkeypatch):
    job,marker=stage(tmp_path)
    rename=b.os.rename
    def injected(src,dst):
        if Path(src).parent.name=="scenes":
            raise OSError("injected interrupted scene publication")
        return rename(src,dst)
    with monkeypatch.context() as m:
        m.setattr(b.os,"rename",injected)
        with pytest.raises(OSError):
            b.publish_transaction(tmp_path,SPEC,IH)
    assert (tmp_path/"catalogs"/SPEC["catalog_id"]).exists()
    assert (job/b.MARKER).exists()
    result,failure=b.recover_catalog(tmp_path,SPEC,IH)
    assert result==marker and failure is None
    assert not job.exists()


def test_complete_worker_is_recovered_before_parent_registration(tmp_path):
    _,marker=stage(tmp_path)
    b.publish_transaction(tmp_path,SPEC,IH)
    result,failure=b.recover_catalog(tmp_path,SPEC,IH)
    assert result==marker and failure is None


def test_partial_attempt_is_preserved_with_same_seed(tmp_path):
    job=b._job_path(tmp_path,SPEC["catalog_id"])
    job.mkdir(parents=True)
    (job/"partial.npz").write_bytes(b"failed original payload")
    marker,failure=b.recover_catalog(tmp_path,SPEC,IH)
    assert marker is None and failure["retry_allowed"]
    assert failure["seed"]==SPEC["seed"]
    assert (tmp_path/failure["path"]/"partial.npz").read_bytes()==b"failed original payload"
    assert not job.exists()


@pytest.mark.parametrize("retry",[True,False])
def test_orphan_failure_import_preserves_hard_stop(tmp_path,retry):
    # Simulate worker archive completed, but parent never recorded its outcome.
    row=b._archive_attempt(tmp_path,SPEC,IH,dict(type="SyntheticFailure",error="retained",retry_allowed=retry))
    m=dict(identity=IDENTITY,files=[],failures=[])
    assert b.recover_failure_ledger(tmp_path,m,[SPEC],IH) is (not retry)
    assert m["failures"]==[row]
    # Repeat startup must neither duplicate the entry nor forget its hard stop.
    assert b.recover_failure_ledger(tmp_path,m,[SPEC],IH) is (not retry)
    assert len(m["failures"])==1
    assert len((tmp_path/"events.jsonl").read_text().splitlines())==1


def test_event_accepts_retained_failure_timestamp(tmp_path):
    b.event(tmp_path,"attempt_interrupted",time_utc="2027-01-01T00:00:00+00:00",type="KeyboardInterrupt")
    e=json.loads((tmp_path/"events.jsonl").read_text())
    assert e["time_utc"]=="2027-01-01T00:00:00+00:00"


def test_tampered_staged_or_published_payload_rejected(tmp_path):
    job,_=stage(tmp_path)
    target=job/"catalogs"/SPEC["catalog_id"]/"targets.npz"
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError,match="immutable payload mismatch"):
        b.publish_transaction(tmp_path,SPEC,IH)


def test_changed_identity_or_seed_rejected(tmp_path):
    job,_=stage(tmp_path)
    with pytest.raises(ValueError,match="identity mismatch"):
        b.publish_transaction(tmp_path,SPEC,"changed")
    with pytest.raises(ValueError,match="identity mismatch"):
        b.publish_transaction(tmp_path,{**SPEC,"seed":124},IH)
    assert job.exists()


def test_active_worker_lock_prevents_resume_mutation(tmp_path):
    job=b._job_path(tmp_path,SPEC["catalog_id"])
    job.mkdir(parents=True)
    with b.exclusive(b._lock_path(tmp_path,SPEC["catalog_id"])):
        with pytest.raises(RuntimeError,match="another producer"):
            b.recover_catalog(tmp_path,SPEC,IH)
    assert job.exists() and not (tmp_path/"failures").exists()


def test_interrupted_ephemeris_copy_retained_then_completed(tmp_path):
    source=tmp_path/"source.bsp"
    source.write_bytes(b"verified ephemeris")
    cache=tmp_path/"cache"
    cache.mkdir()
    (cache/"de421.bsp.copying").write_bytes(b"partial")
    p=b.prepare_ephemeris(cache,source)
    assert p.read_bytes()==source.read_bytes()
    assert (cache/"interrupted-ephemeris-copy-001").read_bytes()==b"partial"
    before=p.stat().st_mtime_ns
    assert b.prepare_ephemeris(cache,source)==p
    assert p.stat().st_mtime_ns==before


def test_completed_resume_does_not_rewrite_manifest_or_events(tmp_path,monkeypatch):
    from obsplan.training_data import v2_sampling
    root=tmp_path/"data"
    root.mkdir()
    eph=tmp_path/"source.bsp"
    eph.write_bytes(b"ephemeris")
    ref=tmp_path/"v1"
    ref.mkdir()
    identity={"frozen":True}
    monkeypatch.setattr(v2_sampling,"specs",lambda:[SPEC])
    monkeypatch.setattr(v2_sampling,"sampling_report",lambda rows:{"passed":True})
    monkeypatch.setattr(b,"_identity",lambda *args:(identity,[],{}))
    m=dict(identity=identity,status="complete",files=[],summary={"scenes":1})
    b.atomic_json(root/"manifest.json",m)
    (root/"events.jsonl").write_text('{"kind":"build_complete"}\n')
    before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in [root/"manifest.json",root/"events.jsonl"]}
    b.build(root,eph,ref,resume=True)
    assert before=={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in [root/"manifest.json",root/"events.jsonl"]}


def test_completed_resume_rejects_extra_payload(tmp_path,monkeypatch):
    from obsplan.training_data import v2_sampling
    root=tmp_path/"data"
    root.mkdir()
    eph=tmp_path/"source.bsp"
    eph.write_bytes(b"ephemeris")
    monkeypatch.setattr(v2_sampling,"specs",lambda:[SPEC])
    monkeypatch.setattr(v2_sampling,"sampling_report",lambda rows:{"passed":True})
    monkeypatch.setattr(b,"_identity",lambda *args:({},[],{}))
    b.atomic_json(root/"manifest.json",dict(identity={},status="complete",files=[],summary={}))
    (root/"unexpected").write_text("unlisted")
    with pytest.raises(ValueError,match="inventory changed"):
        b.build(root,eph,tmp_path/"ref",resume=True)
