"""界面计算档位与已发布证据；档位不是六个已证明的帕累托点。

参考数值固定来自 2026-10-24 的公开实验记录，不是实时耗时预测。四个
controlled 档位使用同一 AC 低电量模式确认组；high/xhigh 只有探索期的
合法质量结果，功耗未逐次记录，因此不提供可比的时间参考值。

真正的算法参数仍来自 portfolio.PRESETS。本模块只选择预算，不修改规则、
计分或校验；演示子集必须关闭 finish_near，防止补入子集之外的目标或站点。
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Literal


EFFORT_ORDER = ("instant", "medium", "high", "xhigh", "max", "ultra")

_CONTROLLED_CONTEXT = (
    "2026-10-24 历史实测：Apple M5，8 线程，接通电源且低电量模式开启；"
    "热几何缓存，10000 目标 × 10 站，2026-08-01 至 08-07 UTC。"
    "仅供参考，不是本次运行耗时承诺。"
)
_EXPLORATORY_CONTEXT = (
    "2026-10-24 探索记录：10000 目标 × 10 站，2026-08-01 至 08-07 UTC；"
    "计划通过独立校验，但功耗未逐次记录。质量只作探索参考，"
    "尚无同功耗确认耗时，不纳入已确认的质量—时间前沿。"
)


@dataclass(frozen=True, slots=True)
class EffortProfile:
    key: str
    title: str
    description: str
    benchmark_name: str
    portfolio_preset: str
    h_iters: int
    h_fast: bool
    evidence_status: Literal["controlled", "exploratory"]
    reference_quality: float | None
    reference_covered: int | None
    reference_solver_s: float | None
    reference_total_s: float | None
    reference_context: str

    @property
    def evidence_label(self) -> str:
        return "同条件确认" if self.evidence_status == "controlled" else "探索结果，待同条件复核"


EFFORT_PROFILES = MappingProxyType({
    "instant": EffortProfile(
        "instant", "instant · 最快", "H+ 1 轮，快速局部搜索与秒级精修；"
        "instant 表示最快档，全量仍需要计算和校验，不是零秒完成。",
        "hplus_1_fast", "hplus_fast", 1, True, "controlled",
        0.9610486361519526, 9911, 72.46453377100988, 179.3684250205115,
        _CONTROLLED_CONTEXT,
    ),
    "medium": EffortProfile(
        "medium", "medium · 轻量", "H+ 10 轮，快速局部搜索与秒级精修；增加少量价格迭代。",
        "hplus_10_fast", "hplus_fast", 10, True, "controlled",
        0.9638608753021533, 9933, 79.2415059999912, 186.0360577920219,
        _CONTROLLED_CONTEXT,
    ),
    "high": EffortProfile(
        "high", "high · 加强", "H+ 30 轮，快速局部搜索与秒级精修；已有探索结果，待同条件复核。",
        "hplus_30_fast", "hplus_fast", 30, True, "exploratory",
        0.9654903687933827, 9958, None, None, _EXPLORATORY_CONTEXT,
    ),
    "xhigh": EffortProfile(
        "xhigh", "xhigh · 深入", "H+ 75 轮，快速局部搜索与秒级精修；已有探索结果，待同条件复核。",
        "hplus_75_fast", "hplus_fast", 75, True, "exploratory",
        0.969325629255095, 9984, None, None, _EXPLORATORY_CONTEXT,
    ),
    "max": EffortProfile(
        "max", "max · 质量优先", "H+ 300 轮，完整局部搜索与秒级精修；不增加 G/E 来源或融合。",
        "hplus_300_full", "hplus_fast", 300, False, "controlled",
        0.9743589271062028, 9994, 176.97187662497163, 284.6944314579596,
        _CONTROLLED_CONTEXT,
    ),
    "ultra": EffortProfile(
        "ultra", "ultra · 完整融合", "从头生成 H+/G+/E，三轮融合、完整局部搜索与秒级精修；"
        "计算成本最高，不保证每次运行都优于其他档位。",
        "fusion_baseline", "fusion_baseline", 300, False, "controlled",
        0.978156658461402, 9998, 1515.10031733301, 1626.7182620000094,
        _CONTROLLED_CONTEXT,
    ),
})


def get_effort(name: str) -> EffortProfile:
    """按界面稳定标识取不可变档位；未知值必须显式失败。"""
    if not isinstance(name, str) or name not in EFFORT_PROFILES:
        raise ValueError(f"未知计算档位：{name!r}；可用档位：{', '.join(EFFORT_ORDER)}")
    return EFFORT_PROFILES[name]


def spec_for_effort(name: str, *, seed: int = 2026, workers: int = 8,
                    full_problem: bool = True) -> dict:
    """返回可直接交给 portfolio.solve 的独立参数字典。

    所有档位保留统一规则和精修流程。full_problem=False 用于演示子集，
    只关闭可能补入全量目标的 near-window 步骤，不放宽硬约束。
    """
    profile = get_effort(name)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed 必须是 0 至 2**31-1 的整数")
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
        raise ValueError("workers 必须是正整数")
    if not isinstance(full_problem, bool):
        raise ValueError("full_problem 必须是 bool")
    # Lazy import keeps displaying the catalog independent of solver initialization.
    from .portfolio import PRESETS

    spec = deepcopy(PRESETS[profile.portfolio_preset])
    spec.update(h_iters=profile.h_iters, h_fast=profile.h_fast,
                seed=int(seed), workers=int(workers), finish_near=full_problem)
    return spec
