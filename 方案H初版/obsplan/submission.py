"""学生拍摄方案 v3.0：严格提交适配与独立评分，不依赖调度器的 evaluate。

未规定的评分情形保留为 unresolved，final_score 为 None。窗口来源必须显式传入。
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

from .config import Config
from .data import Dataset
from .schedule import Schedule

START = datetime(2026, 7, 31, tzinfo=timezone.utc)
END = datetime(2026, 8, 6, 23, 59, 59, tzinfo=timezone.utc)
END_EXCLUSIVE = END + timedelta(seconds=1)
FIELDS = {"noradId", "siteCode", "startTime", "endTime"}
POLICY = {
    "version": "v3.0-local-1",
    "document": "学生拍摄方案 v3.0, 2026-11-23",
    "repeat_anchor": "previous_end_to_next_start",
    "repeat_adjacency": "all_parseable_records_including_hard_errors",
    "repeat_recovery": "next_adjacent_gap_at_least_48h",
    "deviation": "max_overrun_outside_one_forecast_window",
    "over_5s_with_sufficient_overlap": "unresolved",
    "coverage": "distinct_known_targets_with_normal_or_soft_scoring_records",
    "zero_coverage_factor": 0.0,
    "rounding": "no_per_task_rounding",
}


def utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def timestamp(value) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", value
    ):
        raise ValueError("时间必须是以 Z 结尾的 ISO 8601 UTC 时间")
    return datetime.fromisoformat(value)


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def parse_json(data: str | bytes):
    def reject(value):
        raise ValueError(f"JSON 不允许 {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"重复 JSON 字段：{key}")
            result[key] = value
        return result

    value = json.loads(data, parse_constant=reject, object_pairs_hook=unique)
    if not isinstance(value, list):
        raise ValueError("提交文件必须为标准 JSON 数组")
    return value


def _record(row) -> tuple[int, int, datetime, datetime]:
    if not isinstance(row, dict) or set(row) != FIELDS:
        raise ValueError("每条观测须且仅须包含 noradId、siteCode、startTime、endTime")
    n, site = row["noradId"], row["siteCode"]
    if type(n) is not int or not 10000 <= n <= 99999:
        raise ValueError("noradId 必须是真实的五位整数，不能补零、截断或重编号")
    if not isinstance(site, str) or not re.fullmatch(r"site_(?:[1-9]|10)", site):
        raise ValueError("siteCode 必须是 site_1 至 site_10")
    a, b = timestamp(row["startTime"]), timestamp(row["endTime"])
    if b <= a:
        raise ValueError("结束时间必须晚于开始时间")
    return n, int(site[5:]) - 1, a, b


def records(cfg: Config, ds: Dataset, sch: Schedule) -> list[dict]:
    """不修改原计划，不静默删除或截断不合规格的记录。"""
    out = []
    for i in np.lexsort((sch.start_s, sch.station)):
        s, t, a, d = (values[i] for values in (sch.station, sch.target, sch.start_s, sch.dur_s))
        if not all(np.isfinite(v) and v == int(v) for v in (s, t, a, d)):
            raise ValueError("提交计划的编号和时间必须是有限整数")
        if not 0 <= s < min(10, ds.n_stations) or not 0 <= t < ds.n_targets:
            raise ValueError("计划中的站点或目标下标不存在")
        start = cfg.start_dt + timedelta(seconds=int(a))
        end = start + timedelta(seconds=int(d))
        row = {"noradId": int(ds.satnum[int(t)]), "siteCode": f"site_{int(s) + 1}",
               "startTime": utc(start), "endTime": utc(end)}
        _record(row)
        if not START <= start < end <= END:
            raise ValueError("观测超出 v3.0 时间范围")
        out.append(row)
    return out


def to_schedule(rows: list, cfg: Config, ds: Dataset) -> Schedule:
    """严格重新解析提交文件；任何记录错误都报错，不跳过。"""
    index = {int(n): i for i, n in enumerate(ds.satnum)}
    values = [[], [], [], []]
    previous = None
    for row in rows:
        n, s, a, b = _record(row)
        if n not in index or s >= ds.n_stations:
            raise ValueError("提交记录中的目标或站点不在数据目录中")
        if not START <= a < b <= END:
            raise ValueError("观测超出 v3.0 时间范围")
        key = (s, a)
        if previous is not None and key < previous:
            raise ValueError("提交记录未按站点数字顺序、开始时间升序排列")
        previous = key
        for col, value in zip(values, (s, index[n], (a - cfg.start_dt).total_seconds(), (b - a).total_seconds())):
            col.append(value)
    return Schedule(*(np.asarray(v, dtype=np.int64 if i < 2 else float) for i, v in enumerate(values)))


def submission_problem(pb, stations: list[int]):
    """仅为交付模式筛选可提交编号、站点和最后一秒；保留原始目标下标。"""
    from .problem import Problem
    from .visibility import Windows
    last = pb.cfg.start_dt + timedelta(days=pb.cfg.days, seconds=-1)
    if pb.cfg.start_dt < START or last > END or pb.cfg.start_dt >= END or pb.cfg.days <= 0:
        raise ValueError("规划时间超出 v3.0 范围")
    max_k = int((last - pb.cfg.start_dt).total_seconds() // pb.cfg.cell_s)
    eligible = (pb.ds.satnum >= 10000) & (pb.ds.satnum <= 99999)
    right = np.minimum(pb.w_k1, max_k)
    sel = eligible[pb.w_tg] & np.isin(pb.w_st, stations) & (right - pb.w_k0 >= pb.m_min)
    w = pb.win
    clipped = Windows(w.station[sel], w.target[sel], w.k0[sel], right[sel].astype(np.int32),
                      w.dark, w.n_samples, w.sgp4_failed)
    return Problem.build(pb.cfg, pb.ds, clipped)


def score(rows: list, ds: Dataset, forecasts: list, *, source: dict) -> dict:
    """按预报窗口交集逐条评分；硬/软错误按记录计数而非按原因计数。"""
    if not isinstance(rows, list):
        raise ValueError("提交文件必须是数组")
    priority = {int(n): float(p) for n, p in zip(ds.satnum, ds.priority)}
    windows = defaultdict(list)
    for row in forecasts:
        n, s, a, b = _record(row)
        if n not in priority or s >= ds.n_stations:
            raise ValueError("预报窗口包含未知目标或站点")
        windows[n, s].append((a, b))
    for vals in windows.values():
        vals.sort()
    parsed, reports = {}, []
    format_errors = []
    previous = None
    for i, row in enumerate(rows):
        r = {"task_id": i + 1, "observation": row, "status": "normal", "reasons": [],
             "effective_duration_s": 0.0, "deviation_s": None, "duration_factor": 0.0,
             "priority_score": 0.0, "score": 0.0, "forecast": None}
        reports.append(r)
        try:
            n, s, a, b = _record(row)
            parsed[i] = (n, s, a, b)
            if previous is not None and (s, a) < previous:
                format_errors.append({"task_id": i + 1, "reason": "未按站点数字顺序和开始时间升序排列"})
            previous = (s, a)
            if not START <= a < b <= END:
                r["reasons"].append("超出 v3.0 时间范围")
                format_errors.append({"task_id": i + 1, "reason": "超出 v3.0 时间范围"})
            if n not in priority or s >= ds.n_stations:
                r["reasons"].append("未知目标或站点")
            else:
                r["priority_score"] = priority[n]
        except (ValueError, TypeError) as exc:
            r["reasons"].append(str(exc))
            format_errors.append({"task_id": i + 1, "reason": str(exc)})

    # 相邻关系包括其他原因已出错的记录；不以“最后一次有效观测”为锚点。
    for column, gap, reason in ((1, 30, "同站相邻观测间隔不足30秒"),
                                (0, 48 * 3600, "同星相邻观测间隔不足48小时")):
        groups = defaultdict(list)
        for i, record in parsed.items():
            groups[record[column]].append(i)
        for ids in groups.values():
            ids.sort(key=lambda i: (parsed[i][2], i))
            for prev, curr in zip(ids, ids[1:]):
                if (parsed[curr][2] - parsed[prev][3]).total_seconds() < gap:
                    reports[curr]["reasons"].append(reason)

    for i, (n, s, a, b) in parsed.items():
        r = reports[i]
        candidates = []
        for left, right in windows[n, s]:
            duration = max(0.0, (min(b, right) - max(a, left)).total_seconds())
            deviation = max(0.0, (left - a).total_seconds(), (b - right).total_seconds())
            candidates.append((duration, -deviation, left, right))
        if candidates:
            duration, neg_dev, left, right = max(candidates)
            deviation = -neg_dev
            r.update(effective_duration_s=duration, deviation_s=deviation,
                     forecast={"startTime": utc(left), "endTime": utc(right)})
        else:
            duration, deviation = 0.0, None
        if duration < 120:
            r["reasons"].append("与单个预报窗口的有效交集不足120秒")
        if not r["reasons"]:
            r["duration_factor"] = 0.6 + 0.4 * min(120.0, duration - 120.0) / 120.0
            r["score"] = r["priority_score"] * r["duration_factor"]
            if deviation > 5:
                r.update(status="unresolved", score=None)
                r["reasons"].append("偏差超过5秒但有效交集不少于120秒，文档未定义计分")
            elif deviation > 1:
                r["status"] = "soft"
                r["score"] *= 0.5
                r["reasons"].append("时间偏差大于1秒且不超过5秒")
    for r in reports:
        if r["reasons"] and r["status"] == "normal":
            r.update(status="hard", score=0.0)
    hard = sum(r["status"] == "hard" for r in reports)
    soft = sum(r["status"] == "soft" for r in reports)
    unresolved = sum(r["status"] == "unresolved" for r in reports)
    covered = len({parsed[i][0] for i, r in enumerate(reports) if r["status"] in ("normal", "soft")})
    factor = max(0.0, 1.0 - (hard + soft) / covered) if covered else 0.0
    base = sum(r["score"] or 0.0 for r in reports)
    return {"policy": POLICY, "forecast_source": source, "tasks": reports,
            "format_valid": not format_errors, "format_errors": format_errors,
            "summary": {"tasks": len(rows), "covered": covered, "hard_errors": hard, "soft_errors": soft,
                        "unresolved": unresolved, "base_score": base, "penalty_factor": factor,
                        "final_score": None if unresolved else base * factor,
                        "clean": bool(rows) and not (hard or soft or unresolved or format_errors)},
            "submission_sha256": hashlib.sha256(json_bytes(rows)).hexdigest()}
