"""测试共用夹具。需要赛题数据（仓库根目录 data/）的测试在没有数据时自动跳过。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsplan.config import Config  # noqa: E402


def _has_data(cfg: Config) -> bool:
    root = Path(cfg.data_dir)
    return root.exists() and (any(root.rglob("*.zip")) or any(root.rglob("轨道*.txt")))


@pytest.fixture(scope="session")
def cfg():
    c = Config()
    if not _has_data(c):
        pytest.skip("没有赛题数据（仓库根目录 data/）")
    return c


@pytest.fixture(scope="session")
def ds(cfg):
    from obsplan.data import load
    return load(cfg)


@pytest.fixture(scope="session")
def pb(cfg, ds):
    from obsplan.problem import Problem
    from obsplan.visibility import get_windows
    return Problem.build(cfg, ds, get_windows(cfg, ds))


@pytest.fixture(scope="session")
def sub(pb):
    """小场景：10% 目标 × 3 个站 × 6 天，几秒内能跑完各算法。"""
    from obsplan.ga import sub_problem
    return sub_problem(pb, np.random.default_rng(7), target_frac=0.1, n_st=3)
