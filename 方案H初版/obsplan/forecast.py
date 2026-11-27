"""为本次提交生成独立的局部可见区间，明确区别于尚未提供的官方预报。"""
from __future__ import annotations

import hashlib
from datetime import timedelta

import numpy as np

from .schedule import Schedule
from .submission import END, START, _record, json_bytes, utc
from .validator import validate


def local_forecasts(cfg, ds, rows):
    index = {int(n): i for i, n in enumerate(ds.satnum)}
    selected = []
    for row in rows:
        try:
            n, s, a, b = _record(row)
        except (ValueError, TypeError):
            continue
        if n not in index or s >= ds.n_stations or not START <= a < b <= END:
            continue
        selected.append((n, s, index[n], (a - cfg.start_dt).total_seconds(), (b - cfg.start_dt).total_seconds()))
    # Bound work for external files with excessively long or repeated records.
    if sum(b - a + 14 for _, _, _, a, b in selected) > 10_000_000:
        raise ValueError("本地逐秒复核超过一千万采样点；请分批复核或提供预报窗口文件")
    station = np.array([s for _, s, _, _, _ in selected], np.int64)
    target = np.array([t for _, _, t, _, _ in selected], np.int64)
    left = np.array([np.floor(a) - 6 for _, _, _, a, _ in selected])
    right = np.array([np.ceil(b) + 6 for _, _, _, _, b in selected])
    sch = Schedule(station, target, left, right - left)
    check = validate(cfg, ds, sch, include_windows=True)
    forecasts = []
    for (n, s, _, _, _), intervals in zip(selected, check["visible_intervals"]):
        for a, b in intervals:
            forecasts.append({"noradId": n, "siteCode": f"site_{s + 1}",
                              "startTime": utc(cfg.start_dt + timedelta(seconds=a)),
                              "endTime": utc(cfg.start_dt + timedelta(seconds=b))})
    source = {"kind": "skyfield-local-1s", "official": False, "sample_step_s": 1,
              "scope": "submitted intervals with 6s padding; not complete pass predictions",
              "sha256": hashlib.sha256(json_bytes(forecasts)).hexdigest()}
    return forecasts, source
