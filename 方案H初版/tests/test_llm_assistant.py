"""大模型助手：本地规则能把常见说法解析成正确的重规划事件；解析不了时返回 None（不联网）；计划说明模板可用。"""
from datetime import datetime, timezone

from obsplan.llm_assistant import explain_plan, parse_event, parse_rules


def _aug(cfg, day: int) -> int:
    """8 月 day 日 0 点距规划开始的秒数（默认规划期从 7/31 开始，8 月 1 日 0 点 = 第 86400 秒）。"""
    return int((datetime(2026, 8, day, tzinfo=timezone.utc) - cfg.start_dt).total_seconds())


def test_station_down_variants(cfg, ds):
    ev = parse_rules("8月3日12点起6号站停机6小时", cfg, ds)
    assert ev.kind == "down" and ev.station == 5 and ev.hours == 6 and ev.t_s == _aug(cfg, 3) + 12 * 3600
    ev = parse_rules("08-03 12:30 3号站点故障，大约2.5小时", cfg, ds)
    assert ev.kind == "down" and ev.station == 2 and ev.hours == 2.5 and ev.t_s == _aug(cfg, 3) + 12 * 3600 + 1800
    ev = parse_rules("第2天晚上8点 十号站检修三小时", cfg, ds)
    assert ev.kind == "down" and ev.station == 9 and ev.hours == 3 and ev.t_s == (24 + 20) * 3600


def test_priority_and_must(cfg, ds):
    sat0, sat1 = int(ds.satnum[0]), int(ds.satnum[1])
    ev = parse_rules(f"8月2日6点 目标 {sat0} 升为高优先级", cfg, ds)
    assert ev.kind == "priority" and ev.target == 0 and ev.level == "高" and ev.t_s == _aug(cfg, 2) + 6 * 3600
    ev = parse_rules(f"卫星{sat1}必须观测", cfg, ds)
    assert ev.kind == "must" and ev.target == 1 and ev.t_s == 0


def test_unparseable_and_out_of_range(cfg, ds):
    assert parse_rules("今天天气不错", cfg, ds) is None
    assert parse_rules("9月30日12点 6号站停机6小时", cfg, ds) is None          # 不在规划期内
    assert parse_rules("8月3日12点 66号站停机6小时", cfg, ds) is None           # 没有这个站
    ev, how = parse_event("今天天气不错", cfg, ds, use_llm=False)
    assert ev is None and how == "无法解析"


def test_explain_template():
    text, how = explain_plan({"tasks": 13111, "covered": 9987, "score": 29166.0, "combined": 0.9844,
                              "repeat_tasks": 3124, "mean_dur_s": 172.0}, extra={"gap_to_bound_pct": 2.0},
                             use_llm=False)
    assert how == "模板" and "9987" in text and "2.0%" in text


def test_openai_compatible_llm_config(monkeypatch):
    from obsplan import llm_assistant
    monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example.com/v1/")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash")
    cfg = llm_assistant.llm_config()
    assert cfg == {
        "api_key": "sk-test-key",
        "base_url": "https://proxy.example.com/v1/chat/completions",
        "model": "gemini-2.5-flash",
        "label": "大模型",
    }
