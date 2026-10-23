"""伐谋演化、并通过采纳检查后合入求解器的组件。

每个文件保留伐谋生成的原始函数（只加了文件头注释说明来源、实验编号和复评结果）。
采纳检查见 scripts/famou_adopt_check.py：子场景（含伐谋看不到的留出场景）平均超过人工规则、
全量综合指标提升不小于 0.2%、全量计划 Skyfield 逐秒校验 0 违规。
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import numpy as np


def rule_from_module(module):
    """把伐谋格式的 score(names, rows) 包装成 lagrange.recover 用的 rule(X) -> (u, c)。"""
    from ..famou_h_eval import FEATURES

    def rule(X: np.ndarray):
        u, c = module.score(FEATURES, X.tolist())
        return np.asarray(u, float), np.asarray(c, float)
    return rule


def load_rule(name: str = "recover_rule"):
    """已采纳的放置规则；没有采纳时返回 None（H 用人工规则）。"""
    try:
        module = importlib.import_module(f".{name}", __name__)
    except ModuleNotFoundError:
        return None
    return rule_from_module(module)


def rule_from_file(path: str | Path):
    """从文件加载候选规则（只用于已人工审阅过的候选，在本进程内运行）。"""
    spec = importlib.util.spec_from_file_location("famou_candidate_rule", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return rule_from_module(module)
