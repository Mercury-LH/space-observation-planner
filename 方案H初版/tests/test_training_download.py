import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from obsplan.training_data.download import download, extract_checked


def archive(path, name="catalogs/cat-000/data.txt", body=b"example"):
    with tarfile.open(path, "w:gz") as t:
        info = tarfile.TarInfo(name)
        info.size = len(body)
        t.addfile(info, io.BytesIO(body))


def test_verified_cached_archive_extracts_without_network(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    path = cache / "shard.tar.gz"
    archive(path)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"assets":[dict(name=path.name, bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), url="https://invalid.example/not-needed")]}))
    out = tmp_path / "new"
    download(index, out, cache)
    assert (out / "catalogs/cat-000/data.txt").read_bytes() == b"example"
    with pytest.raises(FileExistsError):
        download(index, out, cache)


@pytest.mark.parametrize("name", ["../escape", "/absolute"])
def test_archive_rejects_path_escape(tmp_path, name):
    path = tmp_path / "bad.tar.gz"
    archive(path, name=name)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError):
        extract_checked(path, out)
    assert not (tmp_path / "escape").exists()


def test_archive_rejects_symlink(tmp_path):
    path = tmp_path / "link.tar.gz"
    with tarfile.open(path, "w:gz") as t:
        i = tarfile.TarInfo("link")
        i.type = tarfile.SYMTYPE
        i.linkname = "../escape"
        t.addfile(i)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError):
        extract_checked(path, out)
