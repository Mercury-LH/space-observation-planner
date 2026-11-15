"""求解档位：命令行（`python -m obsplan plan`）和网页「新建规划」共用的默认值。

推荐档 = 快速价格协调 + 按秒精修 + 弹射链 RECOMMENDED_CHAIN_MIN 分钟，全程约两分钟（M1 Pro，窗口已缓存）。
实测用时和分数见 README"时间-质量前沿"一节，那张表由 scripts/frontier_table.py 从 results/frontier/*.json 生成。
"""
from __future__ import annotations

import json
from pathlib import Path

FRONTIER_TABLE = Path(__file__).resolve().parents[1] / "results" / "frontier" / "table.json"

RECOMMENDED_CHAIN_MIN = 1.3     # 推荐档：弹射链分钟数（不带参数的 `obsplan plan` 和网页默认）
TEN_MINUTE_CHAIN_MIN = 9.0      # 十分钟档：`obsplan plan --chain 9`
CHAIN_CHOICES = (0.0, 0.5, RECOMMENDED_CHAIN_MIN, 3.0, TEN_MINUTE_CHAIN_MIN)    # 网页滑块上的几档


def chain_label(m: float) -> str:
    if m <= 0:
        return "不跑"
    if m == RECOMMENDED_CHAIN_MIN:
        return f"{m * 60:.0f} 秒（推荐）"
    return f"{m * 60:.0f} 秒" if m < 1 else f"{m:g} 分钟"


def local_search_kw(fast: bool, chain_minutes: float) -> dict:
    """后面要跑弹射链时，快速模式的局部搜索只给高优先级目标做"挤让插入重复观测"（约 2 秒，原来高、中优先级都做约 20 秒）：
    中优先级的重复观测交给弹射链收尾时"往空当里直接塞"，省下的时间弹射链用得更值（实测见 README）。
    不跑弹射链时保持原样，结果与以前逐字节相同。"""
    if fast and chain_minutes > 0:
        return {"repeat_min_priority": 5.0}
    return {}


def frontier_rows() -> list[dict]:
    """实测的"多少时间到多少分"（scripts/frontier_table.py 从 results/frontier/*.json 生成）；没有这个文件返回空表。"""
    try:
        return json.loads(FRONTIER_TABLE.read_text(encoding="utf-8"))["rows"]
    except (OSError, ValueError, KeyError):
        return []


def chain_help() -> str:
    """网页帮助文字里的实测数字：只列快速模式下滑块上的几档，数字都来自 frontier_rows()，不手填。"""
    parts = []
    for m in CHAIN_CHOICES:
        r = next((r for r in frontier_rows() if r.get("mode") == "fast" and abs(r.get("chain_minutes", -1) - m) < 1e-9), None)
        if r is not None:
            parts.append(f"{chain_label(m)}：全程 {r['wall_s']:.0f} 秒、综合 {r['combined']:.4f}")
    return ("M1 Pro 实测（快速模式，从启动到写出计划，三次取中位数）——" + "；".join(parts) + "。") if parts else ""
