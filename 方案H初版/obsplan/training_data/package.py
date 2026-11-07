"""Package an already audited complete corpus; never uploads or changes Git."""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from .catalog import dump_json
from .download import digest

REPOSITORY = "Mercury-LH/space-observation-planner"
RELEASE_TAG = "synthetic-data-v0.1"


def add_files(tar, source: Path, archive_root: str):
    for path in sorted(source.rglob("*")):
        if path.is_file():
            info = tar.gettarinfo(str(path), arcname=str(Path(archive_root) / path.relative_to(source)))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as f:
                tar.addfile(info, f)


def package(root: Path, output: Path, metadata: Path) -> dict:
    if output.exists():
        raise FileExistsError("package output exists; choose a new directory")
    m = json.loads((root / "manifest.json").read_text())
    if m["status"] != "complete" or m["identity"]["pilot"]:
        raise ValueError("only a complete non-pilot dataset may be released")
    audit = json.loads((metadata / "audit.json").read_text())
    if audit.get("passed") is not True or audit.get("pilot") is not False:
        raise ValueError("metadata/audit.json must show a passing full audit")
    output.mkdir(parents=True)
    assets = []

    def register(path, **extra):
        assets.append(dict(name=path.name, bytes=path.stat().st_size, sha256=digest(path),
                           url=f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/{path.name}", **extra))

    for c in m["catalogs"]:
        cid = c["catalog_id"]
        path = output / f"synthetic-v0.1-{cid}.tar.gz"
        with tarfile.open(path, "w:gz", compresslevel=1) as tar:
            add_files(tar, root / "catalogs" / cid, f"catalogs/{cid}")
            for scene in m["scenes"]:
                if scene["catalog_id"] == cid:
                    add_files(tar, root / scene["path"], scene["path"])
        register(path, catalog_id=cid, split=c["split"])
    path = output / "synthetic-v0.1-metadata.tar.gz"
    with tarfile.open(path, "w:gz", compresslevel=1) as tar:
        for name in ("manifest.json", "events.jsonl"):
            info = tar.gettarinfo(str(root / name), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with (root / name).open("rb") as f:
                tar.addfile(info, f)
    register(path, catalog_id=None, split="all")
    index = dict(dataset_version=m["dataset_version"], release_tag=RELEASE_TAG,
                 repository=REPOSITORY, summary=m["summary"], assets=assets,
                 total_bytes=sum(a["bytes"] for a in assets), manifest_sha256=digest(root / "manifest.json"),
                 events_sha256=digest(root / "events.jsonl"))
    dump_json(output / "release-assets.json", index)
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--metadata", type=Path, required=True)
    a = p.parse_args()
    index = package(a.root, a.output, a.metadata)
    print(json.dumps({"assets":len(index["assets"]), "bytes":index["total_bytes"]}))


if __name__ == "__main__":
    main()
