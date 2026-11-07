"""Stage ten-catalog v0.2 groups, then seal 60 groups plus metadata.

Group membership depends only on the frozen catalog ID order, not completion
order. Retained failed attempts travel with their catalog. All existing archive
members are rehashed against the final manifest before a package is accepted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_files
from .download import digest
from .package import (_catalog_payload, _entries, _outside_dataset, _payload_path,
                      _write_shard, verify_shard)

VERSION = "synthetic-v0.2"
RELEASE_TAG = "synthetic-data-v0.2"
REPOSITORY = "Mercury-LH/space-observation-planner"
METADATA_NAME = f"{VERSION}-metadata.tar.gz"
INDEX_NAME = "release-assets.json"
CATALOG_COUNT = 600
GROUP_SIZE = 10


def group_specs(manifest: dict) -> list[list[dict]]:
    if (manifest.get("dataset_version") != VERSION
            or manifest.get("identity", {}).get("pilot") is not False):
        raise ValueError("only formal v0.2 datasets may be staged for release")
    specs = manifest["identity"].get("catalog_specs", [])
    ids = [s["catalog_id"] for s in specs]
    expected = [f"v2-cat-{k:03d}" for k in range(CATALOG_COUNT)]
    if len(ids) != len(set(ids)) or sorted(ids) != expected:
        raise ValueError("frozen v0.2 catalog spec inventory differs")
    ordered = sorted(specs, key=lambda s: s["catalog_id"])
    return [ordered[k:k+GROUP_SIZE] for k in range(0, len(ordered), GROUP_SIZE)]


def group_name(group_id: int) -> str:
    if type(group_id) is not int or not 0 <= group_id < CATALOG_COUNT // GROUP_SIZE:
        raise ValueError("invalid group ID")
    return f"{VERSION}-group-{group_id:03d}.tar.gz"


def _group_payload(root: Path, manifest: dict, group_id: int) -> tuple[list[dict], dict[str, dict]]:
    name = group_name(group_id)
    specs = group_specs(manifest)[group_id]
    expected = {}
    catalogs = []
    for spec in specs:
        cid = spec["catalog_id"]
        cat, payload = _catalog_payload(root, manifest, cid, variants=(0,), include_failures=True)
        if any(cat.get(key) != value for key, value in spec.items()):
            raise ValueError(f"catalog differs from frozen spec: {cid}")
        native = {f"scenes/{cid}-v0/input/{filename}" for filename in
                  ("轨道.txt", "站址.txt", "高优先级目标.txt", "中优先级目标.txt", "低优先级目标.txt")}
        if not native <= payload.keys():
            raise ValueError(f"missing native inputs: {cid}")
        if expected.keys() & payload.keys():
            raise ValueError(f"overlapping catalog payloads: {name}")
        expected.update(payload)
        catalogs.append(cat)
    return catalogs, expected


def _asset(path: Path, *, group_id: int | None, catalogs: list[dict]) -> dict:
    return dict(name=path.name, bytes=path.stat().st_size, sha256=digest(path),
                url=f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/{path.name}",
                group_id=group_id, catalog_ids=[c["catalog_id"] for c in catalogs],
                splits=sorted({c["split"] for c in catalogs}) if catalogs else ["all"])


def stage_group(root: Path, output: Path, group_id: int) -> dict:
    root, output = Path(root), Path(output)
    _outside_dataset(root, output)
    manifest = json.loads((root / "manifest.json").read_text())
    catalogs, expected = _group_payload(root, manifest, group_id)
    output.mkdir(parents=True, exist_ok=True)
    path = output / group_name(group_id)
    _write_shard(root, path, expected)
    return _asset(path, group_id=group_id, catalogs=catalogs)


def package(root: Path, output: Path, metadata: Path, *, reuse_shards: bool = False) -> dict:
    root, output, metadata = Path(root), Path(output), Path(metadata)
    _outside_dataset(root, output)
    if output.exists() and not reuse_shards:
        raise FileExistsError("package output exists; choose a new directory or explicitly reuse staged shards")
    manifest = json.loads((root / "manifest.json").read_text())
    groups = group_specs(manifest)
    if manifest.get("status") != "complete" or len(manifest.get("catalogs", [])) != CATALOG_COUNT:
        raise ValueError("only a complete v0.2 dataset may be released")
    audit = json.loads((metadata / "audit.json").read_text())
    if (audit.get("passed") is not True or audit.get("pilot") is not False
            or audit.get("dataset_version") != VERSION
            or audit.get("v1_reference", {}).get("checked") is not True
            or audit.get("sampling_report", {}).get("passed") is not True
            or audit.get("plan_provenance", {}).get("passed") is not True):
        raise ValueError("metadata/audit.json must show a passing full v0.2 audit including v0.1 isolation")
    if audit.get("counts") != manifest.get("summary"):
        raise ValueError("audit and manifest counts differ")
    metadata_entries = {}
    for name, key in (("manifest.json", "manifest_sha256"), ("events.jsonl", "events_sha256")):
        path = _payload_path(root, name)
        sha = digest(path)
        if audit.get("checksums", {}).get(key) != sha:
            raise ValueError(f"audit is not bound to current {name}")
        metadata_entries[name] = dict(path=name, bytes=path.stat().st_size, sha256=sha)
    audit_files(root.resolve(), manifest)
    allowed = {group_name(k) for k in range(len(groups))}
    if output.exists():
        if (output / INDEX_NAME).exists() or (output / METADATA_NAME).exists():
            raise FileExistsError("final package metadata already exists; refusing overwrite")
        if any(p.name not in allowed or not p.is_file() or p.is_symlink() for p in output.iterdir()):
            raise ValueError("staging contains unknown entries or retained partial shards")
    output.mkdir(parents=True, exist_ok=True)
    assets, archived = [], set()
    for group_id in range(len(groups)):
        catalogs, expected = _group_payload(root, manifest, group_id)
        if archived & expected.keys():
            raise ValueError("group archives overlap")
        archived.update(expected)
        path = output / group_name(group_id)
        if path.exists():
            verify_shard(path, expected)
        else:
            _write_shard(root, path, expected)
        assets.append(_asset(path, group_id=group_id, catalogs=catalogs))
    if archived != _entries(manifest).keys():
        raise ValueError("group archives do not cover the complete payload including failures")
    path = output / METADATA_NAME
    _write_shard(root, path, metadata_entries)
    assets.append(_asset(path, group_id=None, catalogs=[]))
    index = dict(dataset_version=VERSION, release_tag=RELEASE_TAG, repository=REPOSITORY,
                 group_size=GROUP_SIZE, summary=manifest["summary"], assets=assets,
                 total_bytes=sum(a["bytes"] for a in assets),
                 manifest_sha256=metadata_entries["manifest.json"]["sha256"],
                 events_sha256=metadata_entries["events.jsonl"]["sha256"])
    with (output / INDEX_NAME).open("x") as stream:
        json.dump(index, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return index


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage-group", type=int)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--reuse-shards", action="store_true")
    args = parser.parse_args(argv)
    if args.stage_group is not None:
        if args.reuse_shards:
            parser.error("--stage-group never overwrites or reuses an existing shard")
        result = stage_group(args.root, args.output, args.stage_group)
    else:
        if args.metadata is None:
            parser.error("--metadata required for final package")
        result = package(args.root, args.output, args.metadata, reuse_shards=args.reuse_shards)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
