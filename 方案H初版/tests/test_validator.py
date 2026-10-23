"""Skyfield 校验器：规划器给出的任务应合法；故意挪出窗口的任务必须被查出。"""
import numpy as np
import pytest

from obsplan import lagrange
from obsplan.schedule import Schedule
from obsplan.validator import validate


@pytest.mark.slow
def test_validator_accepts_plan_and_catches_corruption(cfg, ds, sub):
    sch, _ = lagrange.solve(sub, iters=20, verbose=False)
    idx = np.random.default_rng(0).choice(len(sch), 40, replace=False)
    part = Schedule(sch.station[idx], sch.target[idx], sch.start_s[idx], sch.dur_s[idx])
    assert validate(cfg, ds, part)["legal"]
    bad = Schedule(part.station.copy(), part.target.copy(), part.start_s.copy(), part.dur_s.copy())
    bad.start_s[0] += 6 * 3600                      # 挪到 6 小时后：几乎必然天亮、落地平线或进地影
    chk = validate(cfg, ds, bad)
    assert not chk["legal"] and chk["invalid_mask"][0]
