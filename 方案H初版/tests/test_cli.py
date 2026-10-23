"""命令行：导出的计划 CSV 能被 validate 原样读回。"""
import json

import numpy as np

from obsplan.__main__ import read_plan
from obsplan.greedy import MANUAL_WEIGHTS, algo_weighted
from obsplan.schedule import export_csv


def test_plan_csv_roundtrip(cfg, ds, sub, tmp_path):
    sch = algo_weighted(sub, MANUAL_WEIGHTS, "C")
    path = tmp_path / "plan.csv"
    export_csv(cfg, ds, sch, path)
    back, skipped = read_plan(cfg, ds, path)
    assert skipped == []
    a = sorted(zip(sch.station, sch.target, sch.start_s, sch.dur_s))
    b = sorted(zip(back.station, back.target, back.start_s.astype(int), back.dur_s.astype(int)))
    assert a == b


def test_plan_json_seconds(cfg, ds, tmp_path):
    t = int(ds.satnum[0])
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"observations": [{"station_id": "3", "target_id": t, "start": 600.0, "end": 780.0}]}))
    back, skipped = read_plan(cfg, ds, path, seconds_from_start=True)
    assert skipped == [] and back.station[0] == 2 and back.target[0] == 0
    assert np.isclose(back.start_s[0], 600.0) and np.isclose(back.dur_s[0], 180.0)
