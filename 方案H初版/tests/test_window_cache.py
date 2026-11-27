from types import SimpleNamespace

import numpy as np
import pytest

from obsplan.config import Config
from obsplan import visibility as v


def window():
    w = v.Windows(np.array([0], np.int16), np.array([0], np.int32), np.array([0], np.int32),
                  np.array([4], np.int32), [[(0, 4)]], 5, np.array([False]))
    w.short = tuple(np.array([], dtype=np.int32) for _ in range(4))
    return w


def test_failed_cache_write_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "windows.npz"
    window().save(path)
    original = path.read_bytes()

    def fail(file, **arrays):
        file.write(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(v.np, "savez_compressed", fail)
    with pytest.raises(OSError):
        window().save(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_corrupt_cache_is_retained_and_rebuilt(tmp_path, monkeypatch):
    cfg = Config(cache_dir=str(tmp_path))
    path = tmp_path / f"windows_{cfg.scenario_key()}.npz"
    path.write_bytes(b"interrupted")
    monkeypatch.setattr(v, "compute_windows", lambda *args, **kwargs: window())
    result = v.get_windows(cfg, SimpleNamespace(n_stations=1))
    assert result.k1.tolist() == [4]
    assert v.Windows.load(path, 1).k1.tolist() == [4]
    assert list(tmp_path.glob("*.corrupt-*.npz"))[0].read_bytes() == b"interrupted"
