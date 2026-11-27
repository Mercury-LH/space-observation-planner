"""提交格式规范 v3.0 的导出与复算（obsplan/submission.py）。

纯逻辑部分（计分、状态判定、窗口判定、格式检查、资源冲突）不依赖 Skyfield，也不依赖赛题数据，几毫秒跑完；
只有带 ds / sub 夹具的测试需要赛题数据，没有数据时自动跳过；用 Skyfield 的集成测试标了 slow。
"""
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from obsplan.config import Config
from obsplan.schedule import Schedule, task_score
from obsplan.submission import (HARD, NORMAL, SOFT, Parsed, classify, judge_window, load_json,
                                parse_records, resource_errors, score_submission, site_code,
                                summarize, to_records, write_json, fit_to_range)

TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@pytest.fixture(scope="module")
def c0():
    """不依赖赛题数据的配置（cfg 夹具在没有数据时会跳过）。这些测试按 8/1 0 点起 6 天写的（规划期终点 8/7 0 点
    正好比规范截止时刻晚 1 秒），默认规划期改成规范的 7/31 起 7 天后仍固定用这个，便于手算期望值。"""
    return Config(start_utc="2026-08-01T00:00:00", days=6.0)


@pytest.fixture
def stub():
    """假的数据集：只含这些模块用到的字段。第 4 颗的编号只有 1 位，用来测"不足 5 位"的提醒。"""
    return SimpleNamespace(satnum=np.array([28446, 25544, 33591, 7]), n_stations=10, n_targets=4,
                           priority=np.array([5.0, 3.0, 2.0, 2.0]))


def rec(nid=28446, site="site_1", a="2026-08-01T10:00:00Z", b="2026-08-01T10:03:00Z"):
    return {"noradId": nid, "siteCode": site, "startTime": a, "endTime": b}


# ---------------------------------------------------------------- 1. 规范第 9 节示例

def test_spec_section9_example(c0):
    """三条：正常 / 硬错误（未知编号）/ 软错误。覆盖 2 颗，错误 2 条，系数 0，最终得分 0。"""
    rep = summarize(c0, np.array([5.0, 0.0, 3.0]), np.array([0, -1, 1]), [NORMAL, HARD, SOFT],
                    np.array([210.0, 0.0, 207.0]), n_targets_total=10000, ref=100.0)
    assert rep["task_score"] == pytest.approx([4.5, 0.0, 3 * (0.6 + 0.4 * 87 / 120) / 2])
    assert rep["task_score"][2] == pytest.approx(1.335)          # 规范写 1.33
    assert rep["valid_score"] == pytest.approx(5.835)
    assert rep["covered"] == 2
    assert (rep["normal"], rep["soft"], rep["hard"]) == (1, 1, 1)
    assert rep["factor"] == 0.0
    assert rep["final_score"] == 0.0


def test_summarize_no_error_factor_one(c0):
    rep = summarize(c0, np.array([5.0, 3.0]), np.array([0, 1]), [NORMAL, NORMAL],
                    np.array([210.0, 207.0]), 10000, 100.0)
    assert rep["factor"] == 1.0
    assert rep["final_score"] == pytest.approx(rep["valid_score"]) == pytest.approx(4.5 + 3 * (0.6 + 0.4 * 87 / 120))
    assert rep["covered_scoring"] == 2


def test_summarize_factor_not_negative(c0):
    """错误数（3）超过覆盖数（1）时系数取 0，不能是负数。"""
    rep = summarize(c0, np.array([5.0, 5.0, 5.0]), np.array([0, 0, 0]), [NORMAL, HARD, HARD],
                    np.array([240.0, 240.0, 240.0]), 10000, 100.0)
    assert rep["covered"] == 1 and rep["hard"] == 2
    assert rep["factor"] == 0.0 and rep["final_score"] == 0.0
    rep = summarize(c0, np.array([5.0] * 4), np.array([0, 0, 0, 0]), [HARD, HARD, SOFT, SOFT],
                    np.array([240.0] * 4), 10000, 100.0)
    assert rep["factor"] == 0.0


def test_summarize_partial_factor_and_hard_scores_zero(c0):
    """3 颗认得出、其中 1 颗只有一条硬错误：硬错误那条即使有效时长很长也 0 分。
    错误率的分母取至少有一条计分的卫星数 2，系数 1 − 1/2 = 0.5；按"认得出就算"的 3 颗是 1 − 1/3。"""
    rep = summarize(c0, np.array([5.0, 3.0, 2.0]), np.array([0, 1, 2]), [NORMAL, NORMAL, HARD],
                    np.array([240.0, 240.0, 300.0]), 10000, 100.0)
    assert rep["task_score"] == pytest.approx([5.0, 3.0, 0.0])
    assert rep["covered"] == 3 and rep["covered_scoring"] == 2
    assert rep["factor"] == pytest.approx(0.5) and rep["final_score"] == pytest.approx(4.0)
    assert rep["factor_loose"] == pytest.approx(2 / 3) and rep["final_score_loose"] == pytest.approx(16 / 3)
    assert rep["final_score"] <= rep["final_score_loose"]


def test_summarize_empty_covered(c0):
    """没有任何认得出的卫星：覆盖数为 0，系数按 0 处理，不能除零。"""
    rep = summarize(c0, np.array([0.0]), np.array([-1]), [HARD], np.array([0.0]), 10000, 100.0)
    assert rep["covered"] == 0 and rep["factor"] == 0.0 and rep["final_score"] == 0.0


# ---------------------------------------------------------------- 2. classify 与时长得分

@pytest.mark.parametrize("overlap, dev, eff, want", [
    (False, np.inf, 0.0, HARD),      # 无交集
    (True, 6.0, 200.0, HARD),        # 偏差超过 5 秒
    (True, 5.01, 200.0, HARD),
    (True, 0.0, 119.0, HARD),        # 偏差 0 但有效时长不足
    (True, 1.0, 119.0, HARD),
    (True, 1.0, 120.0, NORMAL),      # 偏差恰好 1 秒、有效 120 秒
    (True, 0.0, 120.0, NORMAL),
    (True, 1.5, 200.0, SOFT),
    (True, 5.0, 200.0, SOFT),
    (True, 1.5, 100.0, HARD),        # 软错误区间但时长不足，仍是硬错误
])
def test_classify(c0, overlap, dev, eff, want):
    status, why = classify(c0, overlap, dev, eff)
    assert status == want
    assert (why is None) == (want == NORMAL)


def test_duration_score_curve(c0):
    """≥240 秒满分；120 秒 0.6 倍；180 秒 0.8 倍；不足 120 秒 0 分（经 summarize 的单条得分验证）。"""
    eff = np.array([100.0, 120.0, 180.0, 240.0, 300.0])
    P = np.full(5, 5.0)
    assert task_score(c0, P, eff) == pytest.approx([0.0, 3.0, 4.0, 5.0, 5.0])
    rep = summarize(c0, P, np.arange(5), [NORMAL] * 5, eff, 10000, 100.0)
    assert rep["task_score"] == pytest.approx([0.0, 3.0, 4.0, 5.0, 5.0])
    assert rep["task_score"][3] == rep["task_score"][4] == 5.0    # 240 与 300 都是满分


# ---------------------------------------------------------------- 3. judge_window

def _ok(first, spans, n=400):
    """first 起 n 秒的布尔数组；spans 里每个 (a, b) 是可见的整秒区间（含两端）。"""
    ok = np.zeros(n, bool)
    for a, b in spans:
        ok[a - first:b - first + 1] = True
    return ok


def test_judge_inside_window():
    first = 1000
    ok = _ok(first, [(1010, 1209)])
    assert judge_window(1020, 1200, first, ok) == (True, 0.0, 180.0)


def test_judge_early_start_by_3s():
    first = 1000
    ok = _ok(first, [(1010, 1209)])
    ov, dev, eff = judge_window(1007, 1200, first, ok)
    assert ov and dev == 3.0 and eff == 190.0       # 观测长 193，有效少 3


def test_judge_both_ends_outside_add_up():
    """开始早 2 秒、结束晚 4 秒：偏差是落在窗口外的秒数之和 6（超过 5 秒，判硬错误）。"""
    first = 1000
    ok = _ok(first, [(1010, 1209)])
    ov, dev, eff = judge_window(1008, 1213, first, ok)
    assert ov and dev == 6.0 and eff == 199.0
    ov, dev, eff = judge_window(1008, 1210, first, ok)      # 早 2 秒、晚 1 秒：3 秒，软错误
    assert ov and dev == 3.0 and eff == 199.0


def test_judge_shrink_tight_boundary():
    """shrink_s=1：窗口起点后移 1 秒，紧贴起点的观测偏差 1、有效时长少 1。"""
    first = 1000
    ok = _ok(first, [(1010, 1209)])
    assert judge_window(1010, 1100, first, ok, shrink_s=0) == (True, 0.0, 90.0)
    assert judge_window(1010, 1100, first, ok, shrink_s=1) == (True, 1.0, 89.0)
    # 终点一侧同理
    assert judge_window(1150, 1209, first, ok, shrink_s=1) == (True, 1.0, 58.0)


def test_judge_array_edges_not_shrunk():
    """窗口碰到数组两头：真实边界在数组之外，那一端不收紧，另一端照常收紧。"""
    first = 1000
    ok = _ok(first, [(1000, 1100)], n=200)                 # 贴着数组起点
    assert judge_window(1000, 1090, first, ok, shrink_s=3) == (True, 0.0, 90.0)
    assert judge_window(1000, 1100, first, ok, shrink_s=3) == (True, 3.0, 97.0)
    ok = _ok(first, [(1100, 1199)], n=200)                 # 贴着数组终点（last=1199）
    assert judge_window(1110, 1199, first, ok, shrink_s=3) == (True, 0.0, 89.0)
    assert judge_window(1100, 1199, first, ok, shrink_s=3) == (True, 3.0, 96.0)


def test_judge_all_false_no_overlap():
    assert judge_window(1020, 1200, 1000, np.zeros(400, bool)) == (False, np.inf, 0.0)


def test_judge_window_too_far_no_overlap():
    first = 1000
    ok = _ok(first, [(1010, 1050)])
    ov, dev, eff = judge_window(1200, 1300, first, ok)
    assert not ov and dev == np.inf and eff == 0.0


def test_judge_shrink_kills_short_window():
    """收紧后窗口消失（原窗口只有 2 秒宽）→ 无交集。"""
    first = 1000
    ok = _ok(first, [(1100, 1102)])
    assert judge_window(1100, 1102, first, ok, shrink_s=0)[0]
    assert judge_window(1100, 1102, first, ok, shrink_s=2) == (False, np.inf, 0.0)


def test_judge_picks_larger_intersection():
    """两段窗口：选与观测交集大的那段。"""
    first = 1000
    ok = _ok(first, [(1010, 1049), (1100, 1199)])
    ov, dev, eff = judge_window(1030, 1180, first, ok)
    assert ov and eff == 80.0                               # 第二段交集 80，第一段只有 19
    assert dev == 70.0                                      # 相对第二段，开始早了 70 秒
    ov, dev, eff = judge_window(1010, 1080, first, ok)      # 反过来，第一段交集大
    assert ov and eff == 39.0 and dev == 31.0       # 结束比第一段终点晚了 31 秒


# ---------------------------------------------------------------- 4. 导出

def _sch(stations, targets, starts, durs=None):
    n = len(stations)
    durs = np.full(n, 180) if durs is None else np.asarray(durs)
    return Schedule(np.asarray(stations, np.int64), np.asarray(targets, np.int64),
                    np.asarray(starts, np.int64), durs.astype(np.int64))


def test_to_records_fields_types_and_order(c0, stub):
    # 站点 2、0、0、1、0，开始时间乱序
    sch = _sch([2, 0, 0, 1, 0], [0, 1, 2, 1, 0], [5000, 90000, 3000, 70000, 200000], [180, 200, 150, 240, 120])
    recs = to_records(c0, stub, sch)
    assert len(recs) == 5
    for r in recs:
        assert set(r) == {"noradId", "siteCode", "startTime", "endTime"}
        assert type(r["noradId"]) is int
        assert re.fullmatch(r"site_\d+", r["siteCode"])
        assert TIME_RE.match(r["startTime"]) and TIME_RE.match(r["endTime"])
    sites = [r["siteCode"] for r in recs]
    assert sites == ["site_1", "site_1", "site_1", "site_2", "site_3"]        # 分组连续、按序
    g1 = [r["startTime"] for r in recs[:3]]
    assert g1 == sorted(g1)                                                  # 组内升序
    assert recs[0] == {"noradId": 33591, "siteCode": "site_1",
                       "startTime": "2026-08-01T00:50:00Z", "endTime": "2026-08-01T00:52:30Z"}
    assert recs[-1]["noradId"] == 28446 and recs[-1]["startTime"] == "2026-08-01T01:23:20Z"


def test_site_code_offset():
    assert [site_code(j) for j in (0, 1, 9)] == ["site_1", "site_2", "site_10"]
    assert site_code(np.int64(4)) == "site_5"


def test_to_records_rounds_to_seconds(c0, stub):
    sch = Schedule(np.array([0]), np.array([0]), np.array([100.4]), np.array([180.2]))
    r = to_records(c0, stub, sch)[0]
    assert r["startTime"] == "2026-08-01T00:01:40Z" and r["endTime"] == "2026-08-01T00:04:41Z"       # 起止各自取整：end=round(100.4+180.2)=281


def test_to_records_real_ds(c0, ds):
    """真实数据：站点 j 对应 site_{j+1}，编号取自 ds.satnum，十个站都能导出。"""
    sch = _sch(np.arange(10)[::-1], np.arange(10) * 7 + 3, np.arange(10) * 1000)
    recs = to_records(c0, ds, sch)
    assert [r["siteCode"] for r in recs] == [f"site_{k}" for k in range(1, 11)]
    for r in recs:
        j = int(r["siteCode"].split("_")[1]) - 1
        i = 9 - j                                   # 原来第 i 条在站 9-i
        assert r["noradId"] == int(ds.satnum[i * 7 + 3])


def test_write_load_roundtrip(c0, stub, tmp_path):
    sch = _sch([1, 0, 0], [0, 1, 2], [4000, 9000, 1000])
    out = write_json(c0, stub, sch, tmp_path / "sub" / "plan.json")
    assert (tmp_path / "sub" / "plan.json").exists()
    assert out["records"] == 3 and out["out_of_range"] == 0 and out["not_5_digit"] == 0
    assert load_json(out["path"]) == to_records(c0, stub, sch)
    assert isinstance(json.loads((tmp_path / "sub" / "plan.json").read_text(encoding="utf-8")), list)


def test_write_json_counts_out_of_range_and_digits(c0, stub, tmp_path):
    t_end = 6 * 86400                                     # 2026-08-07T00:00:00Z
    sch = _sch([0, 0, 0, 0], [0, 1, 2, 3],
               [t_end - 200,                              # 结束在 08-07T00:00:00Z：超出
                t_end - 1 - 180,                          # 结束在 08-06T23:59:59Z：边界上，不算超出
                -86400 - 10,                              # 开始在 07-30T23:59:50Z：超出
                1000],                                    # 第 4 颗编号是 7，不是 5 位
               [200, 180, 180, 180])
    out = write_json(c0, stub, sch, tmp_path / "x.json")
    assert out["out_of_range"] == 2
    assert out["not_5_digit"] == 1 and out["six_digit"] == 0


def test_load_json_with_line_comments(tmp_path):
    p = tmp_path / "spec.json"
    p.write_text('[\n  // ===== site_1 =====\n  {"noradId": 1, "siteCode": "site_1",\n'
                 '   "startTime": "a", "endTime": "b"}\n  // ... 更多\n]\n', encoding="utf-8")
    assert load_json(p) == [{"noradId": 1, "siteCode": "site_1", "startTime": "a", "endTime": "b"}]


def test_load_json_wrapped_and_invalid(tmp_path):
    p = tmp_path / "w.json"
    p.write_text('{"observations": [{"noradId": 1}]}', encoding="utf-8")
    assert load_json(p) == [{"noradId": 1}]
    p.write_text('{"foo": 1}', encoding="utf-8")
    assert load_json(p) == []                       # 没有 observations / tasks 键时当作空列表
    p.write_text('"abc"', encoding="utf-8")
    with pytest.raises(ValueError):
        load_json(p)


def test_load_json_utf8_bom(tmp_path):
    p = tmp_path / "bom.json"
    p.write_text("[]", encoding="utf-8-sig")
    assert load_json(p) == []


# ---------------------------------------------------------------- 5. parse_records

def _hard(c0, stub, r):
    return parse_records(c0, stub, [r]).hard[0]


def test_parse_valid_record(c0, stub):
    p = parse_records(c0, stub, [rec()])
    assert p.hard == [[]] and p.warnings == {}
    assert p.station[0] == 0 and p.target[0] == 0
    assert p.end_s[0] - p.start_s[0] == 180
    assert p.start_s[0] == 10 * 3600


def test_parse_missing_field(c0, stub):
    r = rec()
    del r["endTime"]
    assert _hard(c0, stub, r) == ["缺少必填字段"]
    assert _hard(c0, stub, "not a dict") == ["缺少必填字段"]
    p = parse_records(c0, stub, [r])
    assert p.station[0] == -1 and p.target[0] == -1 and np.isnan(p.start_s[0])


def test_parse_noradid_types(c0, stub):
    assert "noradId 不是整数" in _hard(c0, stub, rec(nid="28446"))
    assert "noradId 不是整数" in _hard(c0, stub, rec(nid=True))     # bool 是 int 的子类，要单独排除
    assert "noradId 不是整数" in _hard(c0, stub, rec(nid=28446.0))
    assert parse_records(c0, stub, [rec(nid=True)]).target[0] == -1


def test_parse_unknown_ids_and_sites(c0, stub, ds):
    unk = 99999 if 99999 not in set(map(int, ds.satnum)) else max(map(int, ds.satnum)) + 1
    assert unk not in set(map(int, ds.satnum))
    assert _hard(c0, ds, rec(nid=unk)) == ["未知卫星编号"]
    assert _hard(c0, stub, rec(nid=99999)) == ["未知卫星编号"]
    assert "未知站点编码" in _hard(c0, stub, rec(site="site_11"))
    assert "未知站点编码" in _hard(c0, stub, rec(site="site_0"))
    assert "未知站点编码" in _hard(c0, stub, rec(site="Site_1"))


def test_parse_time_format(c0, stub):
    bad = "时间格式不对（须为 ISO 8601 UTC，以 Z 结尾）"
    assert bad in _hard(c0, stub, rec(a="2026-08-01T10:00:00"))                 # 不带 Z
    assert bad in _hard(c0, stub, rec(b="2026-08-01T10:03:00+00:00"))
    assert bad in _hard(c0, stub, rec(a="2026-08-01 10:00:00Z"))
    assert bad in _hard(c0, stub, rec(a=1754042400))                            # 不是字符串
    assert bad in _hard(c0, stub, rec(a="2026-13-01T10:00:00Z"))                # 月份非法：正则过、解析失败
    p = parse_records(c0, stub, [rec(a="2026-08-01T10:00:00")])
    assert np.isnan(p.start_s[0]) and np.isnan(p.end_s[0])


def test_parse_end_before_start_and_equal(c0, stub):
    assert "结束时间不晚于开始时间" in _hard(c0, stub, rec(a="2026-08-01T10:03:00Z", b="2026-08-01T10:00:00Z"))
    assert "结束时间不晚于开始时间" in _hard(c0, stub, rec(a="2026-08-01T10:00:00Z", b="2026-08-01T10:00:00Z"))


def test_parse_out_of_range(c0, stub):
    msg = "超出规定时间范围"
    assert msg in _hard(c0, stub, rec(a="2026-07-30T23:58:00Z", b="2026-07-31T00:01:00Z"))
    assert msg in _hard(c0, stub, rec(a="2026-08-06T23:58:00Z", b="2026-08-07T00:00:00Z"))
    # 两端都是闭区间：恰好贴边不算超出
    assert _hard(c0, stub, rec(a="2026-07-31T00:00:00Z", b="2026-07-31T00:03:00Z")) == []
    assert _hard(c0, stub, rec(a="2026-08-06T23:56:59Z", b="2026-08-06T23:59:59Z")) == []
    # 范围可由参数改（复算用）
    p = parse_records(c0, stub, [rec()], spec_start="2026-08-02T00:00:00Z", spec_end="2026-08-03T00:00:00Z")
    assert msg in p.hard[0]


def test_parse_sort_problems_are_warnings_not_hard(c0, stub):
    # 同站不连续：site_1, site_2, site_1
    rows = [rec(site="site_1"), rec(site="site_2", a="2026-08-01T11:00:00Z", b="2026-08-01T11:03:00Z"),
            rec(site="site_1", a="2026-08-01T12:00:00Z", b="2026-08-01T12:03:00Z")]
    p = parse_records(c0, stub, rows)
    assert p.hard == [[], [], []]
    assert p.warnings.get("同一站点的观测没有连续排列") == 1
    # 站点分组顺序反：site_2 在 site_1 之前
    rows = [rec(site="site_2"), rec(site="site_1")]
    p = parse_records(c0, stub, rows)
    assert p.hard == [[], []]
    assert p.warnings == {"站点分组没有按 site_1、site_2 … 的顺序": 1}
    # 组内乱序
    rows = [rec(a="2026-08-01T12:00:00Z", b="2026-08-01T12:03:00Z"), rec()]
    p = parse_records(c0, stub, rows)
    assert p.hard == [[], []]
    assert p.warnings == {"组内没有按开始时间升序": 1}
    # 排序正确则没有警告
    rows = [rec(), rec(a="2026-08-01T12:00:00Z", b="2026-08-01T12:03:00Z"),
            rec(site="site_2", a="2026-08-01T09:00:00Z", b="2026-08-01T09:03:00Z")]
    assert parse_records(c0, stub, rows).warnings == {}


def test_parse_not_5_digit_warning(c0, stub):
    rows = [rec(nid=7), rec(nid=7, a="2026-08-05T10:00:00Z", b="2026-08-05T10:03:00Z"), rec(nid=28446)]
    p = parse_records(c0, stub, rows)
    assert p.warnings.get("noradId 不足 5 位（整数没有前导 0）") == 2
    assert not any("超过 5 位" in k for k in p.warnings)
    assert p.hard == [[], [], []] and p.target[0] == 3


def test_parse_multiple_hard_reasons_accumulate(c0, stub):
    p = parse_records(c0, stub, [rec(nid=99999, site="site_11")])
    assert p.hard[0] == ["未知卫星编号", "未知站点编码"]


# ---------------------------------------------------------------- 6. resource_errors

def _p(stations, targets, starts, ends):
    n = len(stations)
    return Parsed(np.array(stations, np.int64), np.array(targets, np.int64), np.array(starts, float),
                  np.array(ends, float), [[] for _ in range(n)])


def test_station_switch_30s(c0):
    # 同站、不同卫星；第二条在第一条结束后 29 秒开始 → 违规；第三条隔 30 秒 → 不违规
    p = _p([0, 0, 0], [0, 1, 2], [0, 149, 300], [120, 269, 420])
    h = resource_errors(c0, p)
    assert h[0] == [] and h[2] == []
    assert h[1] == ["同站相邻观测间隔不足 30 秒"]


def test_station_switch_exactly_30_ok_and_other_station_ignored(c0):
    p = _p([0, 0, 1], [0, 1, 2], [0, 150, 100], [120, 270, 220])
    assert resource_errors(c0, p) == [[], [], []]            # 150 = 120 + 30 刚好；站 1 的观测与站 0 无关


def test_station_switch_uses_end_not_start(c0):
    """间隔从上一条结束算起：开始到开始相隔 200 秒，但上一条长 190 秒，实际只隔 10 秒。"""
    p = _p([0, 0], [0, 1], [0, 200], [190, 320])
    assert resource_errors(c0, p)[1] == ["同站相邻观测间隔不足 30 秒"]


def test_station_switch_input_order_irrelevant(c0):
    p = _p([0, 0], [1, 0], [149, 0], [269, 120])                # 后出现的在时间上更早
    h = resource_errors(c0, p)
    assert h[0] == ["同站相邻观测间隔不足 30 秒"] and h[1] == []


def test_repeat_48h(c0):
    H = 3600.0
    # 同一卫星、不同站（避免触发 30 秒规则）：47h59m → 违规；48h → 不违规
    p = _p([0, 1], [5, 5], [0, 120 + 47 * H + 59 * 60], [120, 400 + 47 * H + 59 * 60])
    assert resource_errors(c0, p) == [[], ["同一卫星相邻观测间隔不足 48 小时"]]
    p = _p([0, 1], [5, 5], [0, 120 + 48 * H], [120, 400 + 48 * H])
    assert resource_errors(c0, p) == [[], []]


def test_repeat_48h_default_vs_cascade(c0):
    """A 结束后 47h 开始 B（违规）；B 结束后 48h 开始 C。
    默认只判违规的 B；cascade_48h=True 时，B 之后的 C 也判错。"""
    H = 3600.0
    a0, a1 = 0.0, 120.0
    b0 = a1 + 47 * H
    b1 = b0 + 120
    c_0 = b1 + 48 * H
    p = _p([0, 1, 2], [9, 9, 9], [a0, b0, c_0], [a1, b1, c_0 + 120])
    h = resource_errors(c0, p)
    assert h[0] == [] and h[1] == ["同一卫星相邻观测间隔不足 48 小时"] and h[2] == []
    hc = resource_errors(c0, p, cascade_48h=True)
    assert hc[0] == []
    assert hc[1] == ["同一卫星相邻观测间隔不足 48 小时"] and hc[2] == ["同一卫星相邻观测间隔不足 48 小时"]


def test_resource_errors_skip_unparsed(c0):
    """解析不出（station/target 为 -1，时间为 nan）的记录不参与，也不会报错。"""
    p = Parsed(np.array([-1, 0, 0]), np.array([-1, 0, 1]), np.array([np.nan, 0.0, 130.0]),
               np.array([np.nan, 120.0, 250.0]), [[], [], []])
    h = resource_errors(c0, p)
    assert h[0] == [] and h[1] == [] and h[2] == ["同站相邻观测间隔不足 30 秒"]


def test_overlap_covered_by_long_observation(c0):
    """同站 x=[0,1000] 把 y=[100,200]、z=[500,600] 都盖住：只和前一条比会漏掉 z，和此前最晚的结束比两条都查出。"""
    p = _p([0, 0, 0], [0, 1, 2], [0, 100, 500], [1000, 200, 600])
    h = resource_errors(c0, p)
    assert h[0] == [] and h[1] == ["同站相邻观测间隔不足 30 秒"] and h[2] == ["同站相邻观测间隔不足 30 秒"]


def test_repeat_48h_uses_latest_end(c0):
    """同一卫星 x 很长（结束在 50h），y 在 1h 开始，z 在 y 结束 49h 后、但离 x 结束不到 48h：z 也违规。"""
    H = 3600.0
    p = _p([0, 1, 2], [4, 4, 4], [0, 1 * H, 1 * H + 120 + 49 * H], [50 * H, 1 * H + 120, 1 * H + 240 + 49 * H])
    h = resource_errors(c0, p)
    assert h[1] == ["同一卫星相邻观测间隔不足 48 小时"] and h[2] == ["同一卫星相邻观测间隔不足 48 小时"]


@pytest.mark.parametrize("site", [["site_1"], {"a": 1}, 3, None])
def test_parse_site_code_not_string(c0, stub, site):
    """siteCode 写成列表、字典、数字或 null：判"未知站点编码"，不能让整个复算崩掉。"""
    p = parse_records(c0, stub, [rec(site=site), rec(a="2026-08-02T10:00:00Z", b="2026-08-02T10:03:00Z")])
    assert p.hard[0] == ["未知站点编码"] and p.hard[1] == [] and p.station[0] == -1


def test_format_errors_skip_visibility(c0, stub, monkeypatch):
    """格式已判硬错误的记录（这里是超出时间范围、长达一年的观测）不再算逐秒可见性。"""
    import obsplan.submission as sbm

    def boom(*a, **k):
        raise AssertionError("不该计算逐秒可见性")
    monkeypatch.setattr(sbm, "visibility_masks", boom)
    rows = [rec(a="2026-08-01T00:00:00Z", b="2027-08-01T00:00:00Z"), rec(nid=99999)]
    out = sbm.score_submission(c0, stub, rows)
    rep, det = out["reports"][0], out["details"][0]
    assert det["status"] == [HARD, HARD] and rep["hard"] == 2
    assert "超出规定时间范围" in det["reasons"][0] and "未知卫星编号" in det["reasons"][1]
    assert rep["covered"] == 1 and rep["covered_scoring"] == 0 and rep["factor"] == 0.0


# ---------------------------------------------------------------- 7. 集成（Skyfield）

def _plan_rows(cfg, ds, sub, n):
    from obsplan import lagrange
    sch, _ = lagrange.solve(sub, iters=20, verbose=False)
    idx = np.sort(np.random.default_rng(0).choice(len(sch), min(n, len(sch)), replace=False))
    part = Schedule(sch.station[idx], sch.target[idx], sch.start_s[idx], sch.dur_s[idx])
    return to_records(cfg, ds, part)


def test_integration_quick_few_records(cfg, ds, sub):
    """3 条记录：整套流程（导出→复算）。可信的计划全部正常，系数 1；把一条挪后 6 小时，变成硬错误。"""
    rows = _plan_rows(cfg, ds, sub, 3)
    assert len(rows) == 3
    out = score_submission(cfg, ds, rows)
    rep = out["reports"][0]
    assert rep["hard"] == 0 and rep["soft"] == 0 and rep["normal"] == 3
    assert rep["factor"] == 1.0 and rep["final_score"] == pytest.approx(rep["valid_score"]) and rep["valid_score"] > 0
    assert set(rep["warnings"]) <= {"noradId 不足 5 位（整数没有前导 0）"}   # 抽到编号小于 10000 的卫星时只提示、不算错

    from datetime import datetime, timedelta
    bad = [dict(r) for r in rows]
    for key in ("startTime", "endTime"):
        t = datetime.strptime(bad[0][key], "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=6)
        bad[0][key] = t.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = score_submission(cfg, ds, bad)
    assert out["details"][0]["status"][0] == HARD and out["reports"][0]["hard"] == 1


@pytest.mark.slow
def test_integration_20_records_and_corruption(cfg, ds, sub):
    rows = _plan_rows(cfg, ds, sub, 20)
    assert len(rows) == 20
    rep = score_submission(cfg, ds, rows)["reports"][0]
    assert rep["normal"] == 20 and rep["soft"] == 0 and rep["hard"] == 0
    assert rep["factor"] == 1.0

    from datetime import datetime, timedelta
    bad = [dict(r) for r in rows]
    for key in ("startTime", "endTime"):
        t = datetime.strptime(bad[0][key], "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=6)
        bad[0][key] = t.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = score_submission(cfg, ds, bad)
    rep, det = out["reports"][0], out["details"][0]
    assert det["status"][0] == HARD
    assert rep["hard"] >= 1 and rep["normal"] <= 19


# ---------------------------------------------------------------- 裁进规范时间范围

def test_fit_to_range_trims_and_drops(c0):
    """规划期 8/1 0 点 ~ 8/7 0 点，规范截止 8/6 23:59:59：正好结束在 8/7 0 点的裁 1 秒；
    裁完不够 120 秒的删掉；范围内的原样不动；只缩短不挪动。"""
    end = 6 * 86400                                              # 8/7 0 点（相对规划开始的秒数）
    sch = Schedule(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]),
                   np.array([1000, end - 240, end - 120, end + 10]), np.array([180, 240, 120, 150]), algo="x")
    out, fi = fit_to_range(c0, sch)
    assert fi == {"trimmed": 1, "dropped": 2, "trimmed_s": 1}
    assert out.start_s.tolist() == [1000, end - 240] and out.dur_s.tolist() == [180, 239]
    assert out.station.tolist() == [0, 1] and out.algo == "x"
    recs = to_records(c0, SimpleNamespace(satnum=np.array([1, 2, 3, 4])), out)
    assert max(r["endTime"] for r in recs) == "2026-08-06T23:59:59Z"


def test_fit_to_range_noop_inside(c0):
    sch = Schedule(np.array([0]), np.array([0]), np.array([3600.4]), np.array([200.0]))
    out, fi = fit_to_range(c0, sch)
    assert fi == {"trimmed": 0, "dropped": 0, "trimmed_s": 0}
    assert out.start_s.tolist() == [3600] and out.dur_s.tolist() == [200]


def test_default_period_is_spec_range():
    """默认规划期就是规范 v3.0 的时间范围：7/31 0 点起 7 天；任务最晚结束在 08-06 23:59:59（规划期最后 1 秒不能用）。"""
    from obsplan.submission import SPEC_END, SPEC_START, fit_applies
    c = Config()
    assert c.start_utc + "Z" == SPEC_START and c.days == 7.0
    assert c.end_s == 7 * 86400 - 1
    assert (c.start_dt + __import__("datetime").timedelta(seconds=c.end_s)).strftime("%Y-%m-%dT%H:%M:%SZ") == SPEC_END
    assert fit_applies(c)


@pytest.mark.parametrize("start,days,end,fit", [("2026-08-01T00:00:00", 6.0, 6 * 86400 - 1, True),   # 旧口径：同样少最后 1 秒
                                                ("2026-07-31T00:00:00", 1.0, 86400, True),          # 在范围内结束：不截
                                                ("2026-10-01T00:00:00", 6.0, 6 * 86400, False)])    # 范围外的场景：不截也不裁
def test_end_s(start, days, end, fit):
    from obsplan.submission import fit_applies
    c = Config(start_utc=start, days=days)
    assert c.end_s == end and fit_applies(c) == fit
