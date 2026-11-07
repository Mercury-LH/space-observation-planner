"""Download fixed GitHub Release shards, verify SHA-256, and extract safely.

No Hugging Face account, competition archive, or cloud experiment is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract_checked(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"unsafe archive member: {member.name}")
            target = destination.joinpath(*path.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("archive path escapes destination")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise FileExistsError(f"refusing to overwrite {member.name}")
                stream = tar.extractfile(member)
                with target.open("xb") as out:
                    shutil.copyfileobj(stream, out)


def download(index: Path, destination: Path, cache: Path):
    data = json.loads(index.read_text(encoding="utf-8"))
    if destination.exists():
        raise FileExistsError("destination already exists; choose a fresh directory")
    cache.mkdir(parents=True, exist_ok=True)
    # Download/verify every shard before extraction. Verified downloads are resumable.
    archives = []
    for asset in data["assets"]:
        name = asset["name"]
        if Path(name).name != name:
            raise ValueError("asset name must be a filename")
        path = cache / name
        if not path.exists() or digest(path) != asset["sha256"]:
            temp = path.with_suffix(path.suffix + ".part")
            request = urllib.request.Request(asset["url"], headers={"User-Agent":"obsplan-synthetic-dataset/0.1"})
            with urllib.request.urlopen(request, timeout=90) as response, temp.open("wb") as out:
                shutil.copyfileobj(response, out)
            if temp.stat().st_size != asset["bytes"] or digest(temp) != asset["sha256"]:
                raise ValueError(f"download checksum mismatch: {name}")
            temp.replace(path)
        print(f"verified {name} ({asset['bytes']} bytes)", flush=True)
        archives.append(path)
    destination.mkdir(parents=True)
    for archive in archives:
        extract_checked(archive, destination)
    print(f"extracted {len(archives)} verified shards", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--index", type=Path, default=Path(__file__).resolve().parents[2] / "datasets/synthetic-v0.1/release-assets.json")
    parser.add_argument("--cache", type=Path, default=Path("outputs/dataset-download-cache"))
    args = parser.parse_args()
    download(args.index, args.destination, args.cache)


if __name__ == "__main__":
    main()
