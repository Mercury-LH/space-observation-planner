"""命令行：导出的计划 CSV 能被 validate 原样读回。"""
import json

import numpy as np

from obsplan.__main__ import main, read_plan
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


def test_read_plan_submission_format(cfg, ds, tmp_path):
    """主办方提交格式 v3.0 的字段（siteCode / noradId / startTime / endTime）也能读。"""
    from datetime import timedelta
    t = int(ds.satnum[1])
    iso = lambda s: (cfg.start_dt + timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%SZ")   # 规划开始后 s 秒
    path = tmp_path / "sub.json"
    path.write_text(json.dumps([{"noradId": t, "siteCode": "site_3", "startTime": iso(600), "endTime": iso(780)}]))
    back, skipped = read_plan(cfg, ds, path)
    assert skipped == [] and back.station[0] == 2 and back.target[0] == 1
    assert np.isclose(back.start_s[0], 600.0) and np.isclose(back.dur_s[0], 180.0)


def test_export_fit_spec_range(cfg, ds, tmp_path, capsys):
    """export --fit-spec-range：正好结束在 8/7 0 点的任务裁到 8/6 23:59:59，其余不动；不加时原样导出并提醒。"""
    a = int(ds.satnum[0])
    src = tmp_path / "edge.csv"
    src.write_text("station_id,target_id,observation_start_utc,observation_end_utc\n"
                   f"1,{a},2026-08-06T23:56:00Z,2026-08-07T00:00:00Z\n"
                   f"2,{a},2026-08-01T01:00:00Z,2026-08-01T01:03:00Z\n", encoding="utf-8")
    assert main(["export", str(src), "-o", str(tmp_path / "raw.json")]) == 0
    assert "--fit-spec-range" in capsys.readouterr().out
    assert main(["export", str(src), "--fit-spec-range", "-o", str(tmp_path / "fit.json")]) == 0
    assert "裁短 1 条" in capsys.readouterr().out
    raw = json.loads((tmp_path / "raw.json").read_text(encoding="utf-8"))
    fit = json.loads((tmp_path / "fit.json").read_text(encoding="utf-8"))
    assert [r["endTime"] for r in fit] == ["2026-08-06T23:59:59Z", "2026-08-01T01:03:00Z"]
    assert [r["startTime"] for r in fit] == [r["startTime"] for r in raw]


def test_export_csv_to_submission(cfg, ds, tmp_path, capsys):
    """export 子命令：CSV → 规范 JSON，字段齐全、按站点分组、组内按开始时间升序，默认输出 <stem>_submission.json。"""
    a, b = int(ds.satnum[0]), int(ds.satnum[1])
    src = tmp_path / "mini.csv"
    src.write_text("station_id,target_id,observation_start_utc,observation_end_utc\n"
                   f"5,{a},2026-08-01T01:00:00Z,2026-08-01T01:03:00Z\n"
                   f"2,{b},2026-08-01T02:00:00Z,2026-08-01T02:02:00Z\n"
                   f"2,{a},2026-08-01T00:30:00Z,2026-08-01T00:33:00Z\n", encoding="utf-8")
    assert main(["export", str(src)]) == 0
    out = tmp_path / "mini_submission.json"
    recs = json.loads(out.read_text(encoding="utf-8"))
    assert all(set(r) == {"noradId", "siteCode", "startTime", "endTime"} for r in recs)
    assert [r["siteCode"] for r in recs] == ["site_2", "site_2", "site_5"]
    assert [r["startTime"] for r in recs] == ["2026-08-01T00:30:00Z", "2026-08-01T02:00:00Z", "2026-08-01T01:00:00Z"]
    assert isinstance(recs[0]["noradId"], int) and recs[0]["noradId"] == a
    assert "主办方提交格式 v3.0" in capsys.readouterr().out
    # 输入本身是 JSON 时同样默认写 <stem>_submission.json，且能再读回
    assert main(["export", str(out)]) == 0
    assert json.loads((tmp_path / "mini_submission_submission.json").read_text(encoding="utf-8")) == recs


def test_score_cli_hard_error(ds, tmp_path, capsys):
    """score 子命令：未知卫星编号判硬错误，退出码 1，并写出报告与明细（没有可逐秒复算的条目，很快）。"""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"noradId": 99999999, "siteCode": "site_1", "startTime": "2026-08-01T00:00:00Z",
                                 "endTime": "2026-08-01T00:02:00Z"}]))
    rep, det = tmp_path / "rep.json", tmp_path / "det.csv"
    assert main(["score", str(path), "--shrink", "0,1", "--out", str(rep), "--details", str(det)]) == 1
    out = capsys.readouterr().out
    assert "硬错误 1 条" in out and "各短 1 秒" in out
    reports = json.loads(rep.read_text(encoding="utf-8"))["reports"]
    assert [r["shrink_s"] for r in reports] == [0.0, 1.0] and reports[0]["hard"] == 1
    assert det.read_text(encoding="utf-8-sig").splitlines()[1].split(",")[4] == "硬错误"


def _small_plan_csv(cfg, ds, sub, path, n=3):
    """用贪心算法排出的计划取前 n 行写成小 CSV，返回表头和数据行。"""
    sch = algo_weighted(sub, MANUAL_WEIGHTS, "C")
    export_csv(cfg, ds, sch, path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return lines[0], lines[1:1 + n]


def test_score_csv_keeps_unreadable_rows(cfg, ds, sub, tmp_path, capsys):
    """score 读 CSV：目标编号认不出的行也要计为硬错误，不能被悄悄丢掉。"""
    head, rows = _small_plan_csv(cfg, ds, sub, tmp_path / "full.csv")
    cols = head.split(",")
    bad = rows[0].split(",")
    bad[cols.index("target_id")] = "99999999"
    src = tmp_path / "bad_target.csv"
    src.write_text("\n".join([head, ",".join(bad)] + rows[1:]) + "\n", encoding="utf-8")
    assert main(["score", str(src), "--out", str(tmp_path / "rep.json")]) == 1
    rep = json.loads((tmp_path / "rep.json").read_text(encoding="utf-8"))["reports"][0]
    assert rep["records"] == len(rows) and rep["hard"] >= 1


def test_bad_time_export_skips_score_flags(cfg, ds, sub, tmp_path, capsys):
    """时间写成 not-a-time：export 不崩（跳过该行）；score 判为时间格式不对的硬错误。"""
    head, rows = _small_plan_csv(cfg, ds, sub, tmp_path / "full.csv")
    cols = head.split(",")
    bad = rows[0].split(",")
    bad[cols.index("observation_start_utc")] = "not-a-time"
    src = tmp_path / "bad_time.csv"
    src.write_text("\n".join([head, ",".join(bad)] + rows[1:]) + "\n", encoding="utf-8")
    assert main(["export", str(src)]) == 0
    assert "无法解析的开始/结束时间" in capsys.readouterr().out
    assert len(json.loads((tmp_path / "bad_time_submission.json").read_text(encoding="utf-8"))) == len(rows) - 1
    assert main(["score", str(src), "--out", str(tmp_path / "rep.json")]) == 1
    rep = json.loads((tmp_path / "rep.json").read_text(encoding="utf-8"))["reports"][0]
    assert rep["records"] == len(rows) and any("时间格式" in why for why in rep["reasons"])


def test_export_default_name_keeps_existing_json(ds, tmp_path):
    """export 默认写 <stem>_submission.json，不改动输入旁边已有的同名 .json。"""
    a = int(ds.satnum[0])
    src = tmp_path / "plan.csv"
    src.write_text("station_id,target_id,observation_start_utc,observation_end_utc\n"
                   f"1,{a},2026-08-01T01:00:00Z,2026-08-01T01:03:00Z\n", encoding="utf-8")
    other = tmp_path / "plan.json"
    other.write_text('{"摘要": "不要覆盖我"}', encoding="utf-8")
    assert main(["export", str(src)]) == 0
    assert other.read_text(encoding="utf-8") == '{"摘要": "不要覆盖我"}'
    assert (tmp_path / "plan_submission.json").exists()


def test_read_plan_skips_nonfinite_and_non_string_times(cfg, ds, tmp_path):
    """时间是 nan、inf 或列表时记入跳过，不能让 validate / export 崩掉。"""
    t = int(ds.satnum[0])
    rows = [{"station_id": 1, "target_id": t, "start": "nan", "end": 100},
            {"station_id": 1, "target_id": t, "start": 0, "end": "inf"},
            {"station_id": 1, "target_id": t, "start": [1], "end": 130},
            {"station_id": 1, "target_id": t, "start": 0, "end": 130}]
    p = tmp_path / "odd.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    sch, skipped = read_plan(cfg, ds, p, seconds_from_start=True)
    assert len(sch) == 1 and [n for n, _ in skipped] == [0, 1, 2]
    assert all(why == "无法解析的开始/结束时间" for _, why in skipped)


def test_csv_to_records_rounds_like_to_records(cfg, ds, tmp_path):
    """带小数秒的 CSV：score 走的 csv_to_records 与 export 走的 read_plan → to_records 取整结果相同。"""
    from obsplan.__main__ import csv_to_records
    from obsplan.submission import to_records
    t = int(ds.satnum[0])
    p = tmp_path / "frac.csv"
    p.write_text("station_id,target_id,observation_start_utc,observation_end_utc\n"
                 f"1,{t},2026-08-01T00:10:00.6Z,2026-08-01T00:13:00.6Z\n"
                 f"2,{t},2026-08-03T00:10:00.4Z,2026-08-03T00:13:00.4Z\n", encoding="utf-8")
    sch, skipped = read_plan(cfg, ds, p)
    assert not skipped
    key = lambda r: (r["siteCode"], r["startTime"])
    assert sorted(csv_to_records(cfg, p), key=key) == sorted(to_records(cfg, ds, sch), key=key)
    assert csv_to_records(cfg, p)[0]["startTime"] == "2026-08-01T00:10:01Z"
