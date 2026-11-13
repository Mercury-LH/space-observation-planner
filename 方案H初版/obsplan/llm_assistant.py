"""大模型助手：一句话描述突发事件 → 结构化的重规划事件；用大白话解释一份计划。

赛题文件："大模型可承担任务理解、策略生成、结果解释等角色，但不得以纯文本生成替代实际的轨道/可见性计算、约束校验和任务规划。"
这里大模型只做两件"理解与解释"的事，规划、校验仍由程序完成：

1. parse_event：把"8 月 3 日 12 点起 6 号站停机 6 小时""目标 68131 升为高优先级"这样的话转成 replan.Event。
   先用本地规则解析（不联网也能用）；规则解析不出来、并且配置了千帆 API key 时，再请大模型输出 JSON，
   程序逐个字段校验（站点编号、目标编号、时刻都必须在规划范围内）后才采用。
2. explain_plan：把计划的指标、目标状态汇总、离上界多少交给大模型，写一段给非专业评委看的说明；
   没有 API key 或网络不通时，用模板生成同样内容的说明。

大模型用百度千帆（ModelBuilder v2，兼容 OpenAI 接口，base_url https://qianfan.baidubce.com/v2）。
API key（以 bce-v3/ 开头）放在环境变量 QIANFAN_API_KEY，或方案H初版/.env.qianfan 文件里（一行 QIANFAN_API_KEY=...，
该文件已被 git 忽略，不会上传）。模型名可用 QIANFAN_MODEL 改，默认 ernie-4.5-turbo-128k。
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .replan import Event

BASE_URL = "https://qianfan.baidubce.com/v2/chat/completions"
DEFAULT_MODEL = "ernie-4.5-turbo-128k"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env.qianfan"


def api_key() -> str | None:
    k = os.environ.get("QIANFAN_API_KEY")
    if k:
        return k.strip()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("QIANFAN_API_KEY="):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                return v or None
    return None


def chat(messages: list[dict], temperature: float = 0.2, timeout: float = 30.0) -> str:
    """调用千帆对话接口，返回回复文本。没有 key 或请求失败时抛 RuntimeError。"""
    key = api_key()
    if not key:
        raise RuntimeError("没有配置千帆 API key（环境变量 QIANFAN_API_KEY 或 .env.qianfan）")
    body = json.dumps({"model": os.environ.get("QIANFAN_MODEL", DEFAULT_MODEL), "messages": messages,
                       "temperature": temperature}).encode("utf-8")
    req = urllib.request.Request(BASE_URL, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError(f"千帆接口请求失败：{e}") from e
    if "choices" not in data:
        raise RuntimeError(f"千帆接口返回错误：{str(data)[:200]}")
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------- 突发事件：规则解析
_CN_NUM = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _num(s: str) -> float:
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    if s == "半":
        return 0.5
    if len(s) == 2 and s[0] == "十":
        return 10 + _CN_NUM.get(s[1], 0)
    if len(s) == 2 and s[1] == "十":
        return _CN_NUM.get(s[0], 0) * 10
    if len(s) == 3 and s[1] == "十":
        return _CN_NUM.get(s[0], 0) * 10 + _CN_NUM.get(s[2], 0)
    return float(_CN_NUM.get(s, "nan"))


def _when(text: str, cfg) -> int | None:
    """事件时刻（相对规划开始的秒数）：支持 "8月3日12点" "08-03 12:00" "第3天12点" "开始后30小时"。时间按 UTC。"""
    start = cfg.start_dt
    m = re.search(r"开始后\s*([\d.]+)\s*(小时|h)", text)
    if m:
        return int(float(m.group(1)) * 3600)
    hh = mm = 0
    tm = re.search(r"(\d{1,2})\s*[:：]\s*(\d{2})", text) or re.search(r"(\d{1,2})\s*[点时](半|\d{1,2}分)?", text)
    if tm:
        hh = int(tm.group(1))
        g2 = tm.group(2) or ""
        mm = 30 if g2.startswith("半") else int(re.sub(r"\D", "", g2) or 0)
        if hh > 23 or mm > 59:
            return None
        if re.search(r"(下午|晚上)", text) and hh < 12:
            hh += 12
    m = re.search(r"第\s*(\d+|[一二三四五六七])\s*天", text)
    if m:
        day = int(_num(m.group(1)))
        return int(((day - 1) * 24 + hh) * 3600 + mm * 60)
    m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]", text) or re.search(r"(\d{1,2})-(\d{1,2})(?=\s|$|\D)", text)
    if m:
        try:
            dt = datetime(start.year, int(m.group(1)), int(m.group(2)), hh, mm, tzinfo=start.tzinfo or timezone.utc)
        except ValueError:
            return None
        base = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        return int((dt - base).total_seconds())
    if tm:                                   # 只有钟点：取规划第一天
        return int(hh * 3600 + mm * 60)
    return None


def parse_rules(text: str, cfg, ds) -> Event | None:
    """本地规则解析一句话事件；解析不出返回 None。"""
    t = text.replace("号站点", "号站").replace("站点", "站")
    t_s = _when(t, cfg)
    horizon = int(cfg.n_cells * cfg.cell_s)
    st = re.search(r"(\d{1,2}|[一二三四五六七八九十])\s*号\s*站", t)
    if st and re.search(r"(停机|故障|维护|检修|不可用|下线|宕机)", t):
        hrs = re.search(r"([\d.]+|[一二三四五六七八九十两半]+)\s*(个)?\s*(小时|h)", t)
        station = int(_num(st.group(1))) - 1
        hours = _num(hrs.group(1)) if hrs else float("nan")
        if 0 <= station < ds.n_stations and hours == hours and hours > 0 and t_s is not None and 0 <= t_s < horizon:
            return Event("down", int(t_s), station=station, hours=float(hours))
        return None
    tg = re.search(r"(?:目标|卫星|NORAD|norad)?\s*(\d{4,6})", t)
    if tg is None:
        return None
    sat = int(tg.group(1))
    idx = {int(s): i for i, s in enumerate(ds.satnum)}.get(sat)
    if idx is None:
        return None
    if t_s is None:
        t_s = 0
    if not (0 <= t_s < horizon):
        return None
    if re.search(r"(必须|务必|一定要).{0,4}(观测|看)", t):
        return Event("must", int(t_s), target=idx)
    lv = re.search(r"(升|调|改|设)\s*(为|到|成)?\s*(高|中|低)\s*优先", t) or re.search(r"(高|中|低)\s*优先级", t)
    if lv:
        level = lv.group(lv.lastindex) if lv.lastindex else lv.group(1)
        level = level[-1] if level else "高"
        return Event("priority", int(t_s), target=idx, level=level)
    return None


def _llm_event(text: str, cfg, ds) -> Event | None:
    sys_prompt = (
        "你把空间目标观测规划中的突发事件描述转成 JSON，只输出 JSON，不要解释。字段：\n"
        '{"kind": "down" | "priority" | "must", "time_utc": "YYYY-MM-DD HH:MM", "station": 站点编号（1 起，仅 down）,'
        ' "hours": 停机小时数（仅 down）, "target": 目标的 NORAD 编号（priority/must）, "level": "高" | "中" | "低"（仅 priority）}\n'
        f"规划期从 {cfg.start_dt:%Y-%m-%d %H:%M} UTC 开始，共 {cfg.days:g} 天，有 {ds.n_stations} 个站。"
        "没说时刻时用规划开始时刻。")
    reply = chat([{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}], temperature=0.0)
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    d = json.loads(m.group(0))
    kind = d.get("kind")
    base = cfg.start_dt if cfg.start_dt.tzinfo else cfg.start_dt.replace(tzinfo=timezone.utc)
    try:
        when = datetime.strptime(str(d.get("time_utc", "")), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        t_s = int((when - base).total_seconds())
    except ValueError:
        t_s = 0
    if not (0 <= t_s < int(cfg.n_cells * cfg.cell_s)):
        return None
    if kind == "down":
        station, hours = int(d.get("station", 0)) - 1, float(d.get("hours", 0))
        return Event("down", t_s, station=station, hours=hours) if 0 <= station < ds.n_stations and 0 < hours <= 48 else None
    idx = {int(s): i for i, s in enumerate(ds.satnum)}.get(int(d.get("target", -1)))
    if idx is None:
        return None
    if kind == "must":
        return Event("must", t_s, target=idx)
    if kind == "priority" and d.get("level") in ("高", "中", "低"):
        return Event("priority", t_s, target=idx, level=d["level"])
    return None


def parse_event(text: str, cfg, ds, use_llm: bool = True) -> tuple[Event | None, str]:
    """返回 (事件, 解析方式)。先规则，规则不行再用大模型（结果逐字段校验后才采用）。"""
    ev = parse_rules(text, cfg, ds)
    if ev is not None:
        return ev, "本地规则"
    if use_llm and api_key():
        try:
            ev = _llm_event(text, cfg, ds)
        except (RuntimeError, ValueError, json.JSONDecodeError):
            ev = None
        if ev is not None:
            return ev, "千帆大模型"
    return None, "无法解析"


# ---------------------------------------------------------------- 计划解释
def _facts(metrics: dict, status: dict | None, extra: dict | None) -> dict:
    f = {k: metrics.get(k) for k in ("covered", "score", "combined", "tasks", "repeat_tasks", "mean_dur_s")}
    if status:
        f["target_status"] = status
    if extra:
        f.update(extra)
    return f


def explain_template(metrics: dict, status: dict | None = None, extra: dict | None = None) -> str:
    m = metrics
    parts = [f"这份计划一共安排了 {m['tasks']} 次观测，覆盖了 {m['covered']} 颗目标，总得分 {m['score']:.0f}，"
             f"综合指标 {m['combined']:.4f}。"]
    if m.get("repeat_tasks"):
        parts.append(f"其中 {m['repeat_tasks']} 次是对同一目标间隔 48 小时以上的重复观测，主要给高优先级目标，"
                     f"因为重复观测也计分，而高优先级目标每次得分最高。")
    if m.get("mean_dur_s"):
        parts.append(f"平均每次观测 {m['mean_dur_s']:.0f} 秒：120 秒拿 60% 的分，240 秒拿满分，"
                     "程序在\"多看一颗目标\"和\"把一次观测拉长\"之间按得分取舍。")
    if extra and extra.get("gap_to_bound_pct") is not None:
        parts.append(f"按可证明的上界，这份计划离理论最优最多差 {extra['gap_to_bound_pct']:.1f}%。")
    if extra and extra.get("uncovered_reason"):
        parts.append(f"没覆盖的目标主要原因：{extra['uncovered_reason']}。")
    return "".join(parts)


def explain_plan(metrics: dict, status: dict | None = None, extra: dict | None = None,
                 use_llm: bool = True) -> tuple[str, str]:
    """返回 (说明文字, 来源)。有 key 时请大模型写，失败或没有 key 时用模板。"""
    if use_llm and api_key():
        prompt = ("下面是一份地面望远镜观测空间目标的 6 天计划的统计数据（JSON）。请用 200 字以内的中文、"
                  "给不熟悉航天的评委解释：这份计划做得怎么样、为什么这么安排、还有哪些不足。只依据给出的数据，不要编造数字。\n"
                  + json.dumps(_facts(metrics, status, extra), ensure_ascii=False, default=float))
        try:
            return chat([{"role": "user", "content": prompt}], temperature=0.3).strip(), "千帆大模型"
        except RuntimeError:
            pass
    return explain_template(metrics, status, extra), "模板"
