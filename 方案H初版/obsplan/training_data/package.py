"""Stage immutable catalog shards, then package an audited complete corpus.

This module never uploads, publishes a release, or changes Git. Staged shards
can be uploaded to a draft while other catalogs are still being generated. The
final packager rechecks every archived member against the final file manifest.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import tarfile
from pathlib import Path, PurePosixPath

from .download import digest

REPOSITORY = "Mercury-LH/space-observation-planner"
RELEASE_TAG = "synthetic-data-v0.1"
METADATA_NAME = "synthetic-v0.1-metadata.tar.gz"
INDEX_NAME = "release-assets.json"


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts or "\\" in value
            or path.as_posix() != value or value == "."):
        raise ValueError(f"unsafe or noncanonical dataset path: {value}")
    return path


def _entries(manifest: dict) -> dict[str, dict]:
    result = {}
    for row in manifest.get("files", []):
        name = row["path"]
        _relative_path(name)
        if name in result:
            raise ValueError(f"duplicate manifest file: {name}")
        if (type(row["bytes"]) is not int or row["bytes"] < 0
                or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None):
            raise ValueError(f"invalid file inventory metadata: {name}")
        result[name] = row
    if not result:
        raise ValueError("manifest has no checksummed payload files")
    return result


def _payload_path(root: Path, name: str) -> Path:
    path = root.joinpath(*_relative_path(name).parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing, linked, or escaping payload: {name}")
    return path


def _verify_sources(root: Path, expected: dict[str, dict]) -> None:
    for name, row in expected.items():
        path = _payload_path(root, name)
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"source checksum mismatch: {name}")


def _catalog_payload(root: Path, manifest: dict, catalog_id: str, *,
                     variants: tuple[int, ...] = (0, 1, 2, 3),
                     include_failures: bool = False) -> tuple[dict, dict[str, dict]]:
    if len(_relative_path(catalog_id).parts) != 1:
        raise ValueError("catalog_id must be a single path component")
    if manifest.get("identity", {}).get("pilot") is not False:
        raise ValueError("pilot or unspecified datasets cannot be staged for release")
    catalogs = [c for c in manifest.get("catalogs", []) if c["catalog_id"] == catalog_id]
    if len(catalogs) != 1:
        raise ValueError(f"catalog is not uniquely complete in the manifest: {catalog_id}")
    catalog = catalogs[0]
    scenes = [s for s in manifest.get("scenes", []) if s["catalog_id"] == catalog_id]
    if len(scenes) != len(variants) or {s.get("variant") for s in scenes} != set(variants):
        count = "four" if variants == (0, 1, 2, 3) else str(len(variants))
        raise ValueError(f"catalog must have {count} complete variants: {catalog_id}")
    prefixes = [f"catalogs/{catalog_id}/"]
    if include_failures:
        prefixes.append(f"failures/{catalog_id}/")
    for scene in scenes:
        name = f"{catalog_id}-v{scene['variant']}"
        if (scene.get("scene_id") != name or scene.get("path") != f"scenes/{name}"
                or scene.get("split") != catalog["split"]
                or scene.get("legal") is not True or scene.get("geometry_checked") is not True):
            raise ValueError(f"scene is not complete and independently validated: {name}")
        prefixes.append(f"scenes/{name}/")
    entries = _entries(manifest)
    expected = {name: row for name, row in entries.items() if any(name.startswith(p) for p in prefixes)}
    required = {f"catalogs/{catalog_id}/{name}" for name in ("catalog.json", "targets.npz", "windows.npz")}
    for scene in scenes:
        required.update(f"{scene['path']}/{name}" for name in
                        ("scene.json", "actions.npz", "grid_plan.npz", "plan.npz", "pool_windows.npz",
                         "pool_source_indices.npz", "validation.json", "grid_geometry.npz", "final_geometry.npz"))
    if not required <= expected.keys():
        raise ValueError(f"incomplete catalog payload manifest: {catalog_id}")
    actual = {str(p.relative_to(root)) for prefix in prefixes for p in (root / prefix).rglob("*") if p.is_file()}
    if actual != expected.keys():
        raise ValueError(f"catalog payload inventory differs from manifest: {catalog_id}")
    _verify_sources(root, expected)
    if json.loads((root / f"catalogs/{catalog_id}/catalog.json").read_text()) != catalog:
        raise ValueError(f"catalog metadata differs from manifest: {catalog_id}")
    for scene in scenes:
        path = root / scene["path"]
        if json.loads((path / "scene.json").read_text()) != scene:
            raise ValueError(f"scene metadata differs from manifest: {scene['scene_id']}")
        validation = json.loads((path / "validation.json").read_text())
        for name in ("grid", "final"):
            report = validation.get(name, {})
            if (report.get("legal") is not True or report.get("geometry_checked") is not True
                    or report.get("step_s") != 1.0 or report.get("violations") != {}
                    or report.get("tasks", 0) <= 0 or report["tasks"] != scene[f"{name}_tasks"]):
                raise ValueError(f"missing successful validation evidence: {scene['scene_id']}/{name}")
    return catalog, expected


def verify_shard(path: Path, expected: dict[str, dict]) -> None:
    """Verify actual tar members, never just a shard filename or outer digest."""
    seen = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = member.name
            _relative_path(name)
            if not member.isfile() or name in seen or name not in expected:
                raise ValueError(f"unexpected, duplicate, or unsafe archive member: {name}")
            row = expected[name]
            if member.size != row["bytes"]:
                raise ValueError(f"archive byte size mismatch: {name}")
            stream = archive.extractfile(member)
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
            if h.hexdigest() != row["sha256"]:
                raise ValueError(f"archive checksum mismatch: {name}")
            seen.add(name)
    if seen != expected.keys():
        raise ValueError(f"archive is missing {len(expected.keys() - seen)} manifest files")


def _write_shard(root: Path, path: Path, expected: dict[str, dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite shard: {path.name}")
    temporary = path.with_name(path.name + ".partial")
    # Fixed gzip/tar metadata makes repeated packaging of the same bytes stable.
    # Exclusive creation retains a failed partial and prevents unnoticed overwrites.
    with temporary.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=1) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name in sorted(expected):
                    source = _payload_path(root, name)
                    info = archive.gettarinfo(str(source), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with source.open("rb") as stream:
                        archive.addfile(info, stream)
    verify_shard(temporary, expected)
    # link() atomically refuses an existing final name, unlike replace/rename.
    os.link(temporary, path)
    temporary.unlink()


def _asset(path: Path, **extra) -> dict:
    return dict(name=path.name, bytes=path.stat().st_size, sha256=digest(path),
                url=f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/{path.name}", **extra)


def _outside_dataset(root: Path, output: Path) -> None:
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("package/staging output must be outside the immutable dataset root")


def stage_catalog(root: Path, output: Path, catalog_id: str) -> dict:
    """Stage one completed catalog during generation; no upload and no overwrite."""
    _outside_dataset(root, output)
    manifest = json.loads((root / "manifest.json").read_text())
    catalog, expected = _catalog_payload(root, manifest, catalog_id)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"synthetic-v0.1-{catalog_id}.tar.gz"
    _write_shard(root, path, expected)
    return _asset(path, catalog_id=catalog_id, split=catalog["split"])


def package(root: Path, output: Path, metadata: Path, *, reuse_shards: bool = False) -> dict:
    _outside_dataset(root, output)
    if output.exists() and not reuse_shards:
        raise FileExistsError("package output exists; choose a new directory or explicitly reuse staged shards")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["identity"].get("pilot") is not False:
        raise ValueError("only a complete non-pilot dataset may be released")
    audit = json.loads((metadata / "audit.json").read_text())
    if audit.get("passed") is not True or audit.get("pilot") is not False:
        raise ValueError("metadata/audit.json must show a passing full audit")
    checksums = audit.get("checksums", {})
    metadata_entries = {}
    for name, key in (("manifest.json", "manifest_sha256"), ("events.jsonl", "events_sha256")):
        path = _payload_path(root, name)
        sha = digest(path)
        if checksums.get(key) != sha:
            raise ValueError(f"audit is not bound to the current {name}")
        metadata_entries[name] = dict(path=name, bytes=path.stat().st_size, sha256=sha)
    # Verify the complete file inventory before any new archive is produced.
    from .audit import audit_files
    audit_files(root.resolve(), manifest)
    expected_shards = {f"synthetic-v0.1-{c['catalog_id']}.tar.gz" for c in manifest["catalogs"]}
    if output.exists():
        if (output / INDEX_NAME).exists() or (output / METADATA_NAME).exists():
            raise FileExistsError("final package metadata already exists; refusing overwrite")
        if any(p.name not in expected_shards or not p.is_file() or p.is_symlink() for p in output.iterdir()):
            raise ValueError("staging directory contains unexpected entries or retained partial shards")
    output.mkdir(parents=True, exist_ok=True)
    assets = []
    archived_names = set()
    for catalog in manifest["catalogs"]:
        cid = catalog["catalog_id"]
        _, expected = _catalog_payload(root, manifest, cid)
        if archived_names.intersection(expected):
            raise ValueError("catalog shards overlap")
        archived_names.update(expected)
        path = output / f"synthetic-v0.1-{cid}.tar.gz"
        if path.exists():
            verify_shard(path, expected)
        else:
            _write_shard(root, path, expected)
        assets.append(_asset(path, catalog_id=cid, split=catalog["split"]))
    if archived_names != _entries(manifest).keys():
        raise ValueError("catalog shards do not cover the complete manifest inventory")
    path = output / METADATA_NAME
    _write_shard(root, path, metadata_entries)  # documentation stays in Git, outside dataset root
    assets.append(_asset(path, catalog_id=None, split="all"))
    index = dict(dataset_version=manifest["dataset_version"], release_tag=RELEASE_TAG,
                 repository=REPOSITORY, summary=manifest["summary"], assets=assets,
                 total_bytes=sum(a["bytes"] for a in assets),
                 manifest_sha256=metadata_entries["manifest.json"]["sha256"],
                 events_sha256=metadata_entries["events.jsonl"]["sha256"])
    with (output / INDEX_NAME).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metadata", type=Path, help="directory containing the passing full audit.json")
    parser.add_argument("--reuse-shards", action="store_true", help="verify and reuse existing staged catalog archives")
    parser.add_argument("--stage-catalog", help="stage only this completed catalog; audit metadata is not yet required")
    args = parser.parse_args()
    if args.stage_catalog:
        if args.reuse_shards:
            parser.error("--stage-catalog never reuses or overwrites an existing shard")
        print(json.dumps(stage_catalog(args.root, args.output, args.stage_catalog)))
    else:
        if args.metadata is None:
            parser.error("--metadata is required for final packaging")
        index = package(args.root, args.output, args.metadata, reuse_shards=args.reuse_shards)
        print(json.dumps({"assets": len(index["assets"]), "bytes": index["total_bytes"]}))


if __name__ == "__main__":
    main()
