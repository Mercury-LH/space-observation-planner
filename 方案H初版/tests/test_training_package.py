"""Package/download byte-level closure; fixtures are not orbital audit evidence."""
from copy import deepcopy
import io
import json
import tarfile

import pytest

from obsplan.training_data.audit import audit_files
from obsplan.training_data.download import digest, download
from obsplan.training_data.package import INDEX_NAME, METADATA_NAME, package, stage_catalog


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def refresh_manifest(root, manifest):
    manifest["files"] = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=digest(p))
                         for p in sorted(root.rglob("*")) if p.is_file()
                         and p.name not in ("manifest.json", "events.jsonl") and ".cache" not in p.parts]
    write_json(root / "manifest.json", manifest)


@pytest.fixture
def corpus(tmp_path):
    root, metadata, output = tmp_path / "source", tmp_path / "docs", tmp_path / "staged"
    cat = {"catalog_id": "cat-000", "split": "train", "n_targets": 3}
    catdir = root / "catalogs/cat-000"
    write_json(catdir / "catalog.json", cat)
    (catdir / "targets.npz").write_bytes(b"explicit opaque catalog fixture")
    (catdir / "windows.npz").write_bytes(b"explicit opaque window fixture")
    scenes = []
    for variant in range(4):
        name = f"cat-000-v{variant}"
        scene = dict(catalog_id="cat-000", scene_id=name, path=f"scenes/{name}", split="train",
                     variant=variant, legal=True, geometry_checked=True, grid_tasks=2, final_tasks=2)
        scenes.append(scene)
        path = root / scene["path"]
        write_json(path / "scene.json", scene)
        for name in ("actions.npz", "grid_plan.npz", "plan.npz", "pool_windows.npz", "pool_source_indices.npz",
                     "grid_geometry.npz", "final_geometry.npz"):
            (path / name).write_bytes(f"opaque {variant}/{name}".encode())
        check = dict(legal=True, geometry_checked=True, step_s=1.0, tasks=2, violations={})
        write_json(path / "validation.json", {"grid": check, "final": check})
        for name in ("轨道.txt", "站址.txt", "低优先级目标.txt", "中优先级目标.txt", "高优先级目标.txt"):
            p = path / "input" / name
            p.parent.mkdir(exist_ok=True)
            p.write_text(f"explicit text fixture {variant}\n")
    (root / "events.jsonl").write_text('{"kind":"explicit_test_fixture"}\n')
    (root / ".cache").mkdir()
    (root / ".cache/de421.bsp").write_bytes(b"excluded local cache fixture")
    manifest = dict(status="building", identity={"pilot": False}, dataset_version="synthetic-v0.1",
                    catalogs=[cat], scenes=scenes, summary={"catalogs": 1, "scenes": 4})
    refresh_manifest(root, manifest)
    metadata.mkdir()
    (metadata / "DATASET_CARD.md").write_text("Documentation remains outside the extracted dataset.\n")
    return root, metadata, output, manifest


def finish_and_bind_audit(root, metadata, manifest):
    # This test report proves only that package binds an audit's identity. The
    # genuine dataset audit performs orbital/label checks before issuing it.
    manifest["status"] = "complete"
    refresh_manifest(root, manifest)
    write_json(metadata / "audit.json", dict(passed=True, pilot=False, checksums={
        "manifest_sha256": digest(root / "manifest.json"),
        "events_sha256": digest(root / "events.jsonl"),
    }))


def test_stage_only_complete_four_variants_and_refuses_overwrite(corpus):
    root, _, output, manifest = corpus
    incomplete = deepcopy(manifest)
    incomplete["scenes"].pop()
    write_json(root / "manifest.json", incomplete)
    with pytest.raises(ValueError, match="four complete variants"):
        stage_catalog(root, output, "cat-000")
    assert not output.exists()
    write_json(root / "manifest.json", manifest)
    asset = stage_catalog(root, output, "cat-000")
    assert asset["catalog_id"] == "cat-000" and asset["split"] == "train"
    assert digest(output / asset["name"]) == asset["sha256"]
    assert not (output / INDEX_NAME).exists() and not (output / METADATA_NAME).exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        stage_catalog(root, output, "cat-000")


@pytest.mark.parametrize("fault", ["source_bytes", "unlisted_file", "missing_hash", "invalid_validation"])
def test_stage_rejects_mismatched_or_incomplete_payload(corpus, fault):
    root, _, output, manifest = corpus
    if fault == "source_bytes":
        (root / "catalogs/cat-000/targets.npz").write_bytes(b"corrupted source")
    elif fault == "unlisted_file":
        (root / "scenes/cat-000-v0/unlisted.txt").write_bytes(b"not frozen")
    elif fault == "missing_hash":
        manifest["files"] = [f for f in manifest["files"] if f["path"] != "scenes/cat-000-v0/actions.npz"]
        write_json(root / "manifest.json", manifest)
    else:
        path = root / "scenes/cat-000-v0/validation.json"
        report = json.loads(path.read_text())
        report["grid"]["geometry_checked"] = False
        write_json(path, report)
        refresh_manifest(root, manifest)
    with pytest.raises(ValueError):
        stage_catalog(root, output, "cat-000")
    assert not output.exists()


def rewrite_tar(path, mutation):
    with tarfile.open(path, "r:gz") as archive:
        entries = [(member, archive.extractfile(member).read()) for member in archive]
    mutation(entries)
    with tarfile.open(path, "w:gz") as archive:
        for member, body in entries:
            member.size = len(body)
            archive.addfile(member, io.BytesIO(body))


@pytest.mark.parametrize("fault", ["bytes", "duplicate", "escape", "extra", "missing"])
def test_reuse_rechecks_tar_members_and_rejects_tampering(corpus, fault):
    root, metadata, output, manifest = corpus
    asset = stage_catalog(root, output, "cat-000")
    shard = output / asset["name"]

    def mutate(entries):
        if fault == "bytes":
            member, body = entries[0]
            entries[0] = (member, bytes([body[0] ^ 1]) + body[1:])
        elif fault == "duplicate":
            entries.append(deepcopy(entries[0]))
        elif fault == "missing":
            entries.pop()
        else:
            member = tarfile.TarInfo("../escape" if fault == "escape" else "extra.txt")
            entries.append((member, b"unexpected"))

    rewrite_tar(shard, mutate)
    finish_and_bind_audit(root, metadata, manifest)
    with pytest.raises(ValueError):
        package(root, output, metadata, reuse_shards=True)
    assert not (output / INDEX_NAME).exists() and not (output / METADATA_NAME).exists()


def test_staged_archive_is_checked_against_final_manifest_not_original_filename(corpus):
    root, metadata, output, manifest = corpus
    stage_catalog(root, output, "cat-000")
    (root / "catalogs/cat-000/windows.npz").write_bytes(b"changed after staging")
    finish_and_bind_audit(root, metadata, manifest)
    with pytest.raises(ValueError, match="archive (byte size|checksum) mismatch"):
        package(root, output, metadata, reuse_shards=True)


def test_final_reuse_download_extract_and_file_audit_close_the_payload_loop(corpus, tmp_path, monkeypatch):
    root, metadata, output, manifest = corpus
    staged = stage_catalog(root, output, "cat-000")
    shard = output / staged["name"]
    before = (shard.stat().st_mtime_ns, digest(shard))
    finish_and_bind_audit(root, metadata, manifest)
    with pytest.raises(FileExistsError, match="package output exists"):
        package(root, output, metadata)
    index = package(root, output, metadata, reuse_shards=True)
    assert (shard.stat().st_mtime_ns, digest(shard)) == before
    assert len(index["assets"]) == 2
    assert index["total_bytes"] == sum(row["bytes"] for row in index["assets"])
    assert str(tmp_path) not in json.dumps(index)
    with tarfile.open(output / METADATA_NAME, "r:gz") as archive:
        assert set(archive.getnames()) == {"manifest.json", "events.jsonl"}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: pytest.fail("verified cached assets need no network"))
    extracted = tmp_path / "downloaded"
    download(output / INDEX_NAME, extracted, output)
    restored = json.loads((extracted / "manifest.json").read_text())
    audit = audit_files(extracted.resolve(), restored)
    assert audit["verified_files"] == len(manifest["files"])
    assert audit["manifest_sha256"] == index["manifest_sha256"]
    assert audit["events_sha256"] == index["events_sha256"]
    assert not (extracted / "DATASET_CARD.md").exists()
    assert not (extracted / "audit.json").exists()
    assert not (extracted / ".cache").exists()
    with pytest.raises(FileExistsError, match="final package metadata already exists"):
        package(root, output, metadata, reuse_shards=True)


def test_fresh_package_and_staged_package_have_identical_archives(corpus, tmp_path):
    root, metadata, output, manifest = corpus
    stage_catalog(root, output, "cat-000")
    finish_and_bind_audit(root, metadata, manifest)
    staged = package(root, output, metadata, reuse_shards=True)
    fresh = package(root, tmp_path / "fresh", metadata)
    assert staged == fresh


@pytest.mark.parametrize("fault", ["pilot", "failed_audit", "different_manifest", "different_events"])
def test_final_package_requires_current_passing_full_audit(corpus, fault):
    root, metadata, output, manifest = corpus
    finish_and_bind_audit(root, metadata, manifest)
    report = json.loads((metadata / "audit.json").read_text())
    if fault == "pilot":
        report["pilot"] = True
    elif fault == "failed_audit":
        report["passed"] = False
    elif fault == "different_manifest":
        report["checksums"]["manifest_sha256"] = "0" * 64
    else:
        (root / "events.jsonl").write_text('{"kind":"later_run"}\n')
    write_json(metadata / "audit.json", report)
    with pytest.raises(ValueError):
        package(root, output, metadata)
    assert not output.exists()


def test_stage_output_cannot_pollute_dataset_inventory(corpus):
    root, _, _, _ = corpus
    with pytest.raises(ValueError, match="outside the immutable dataset root"):
        stage_catalog(root, root / "shards", "cat-000")
    assert not (root / "shards").exists()
