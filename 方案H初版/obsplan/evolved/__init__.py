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


def rule_from_module(module, vectorized=None, n_check: int = 1000):
    """把伐谋格式的 score(names, rows) 包装成 lagrange.recover 用的 rule(X) -> (u, c)。
    vectorized：同一规则的 numpy 等价实现（evolved/vectorized.py）。给定时先用它整列计算，再均匀抽查
    n_check 行与原始逐行实现比对，逐位相同才用；否则退回原始实现（例如原始文件被改过）。"""
    from ..famou_h_eval import FEATURES

    def rule(X: np.ndarray):
        X = np.asarray(X, float)
        if vectorized is not None and len(X):
            u, c = vectorized(FEATURES, X)
            idx = np.unique(np.linspace(0, len(X) - 1, min(len(X), n_check)).astype(np.int64))
            u0, c0 = module.score(FEATURES, X[idx].tolist())
            if np.array_equal(u[idx], np.asarray(u0, float)) and np.array_equal(c[idx], np.asarray(c0, float)):
                return u, c
        u, c = module.score(FEATURES, X.tolist())
        return np.asarray(u, float), np.asarray(c, float)
    return rule


def load_rule(name: str = "recover_rule"):
    """已采纳的放置规则；没有采纳时返回 None（H 用人工规则）。"""
    try:
        module = importlib.import_module(f".{name}", __name__)
    except ModuleNotFoundError:
        return None
    from .vectorized import VECTORIZED
    return rule_from_module(module, VECTORIZED.get(name))


def rule_from_file(path: str | Path):
    """从文件加载候选规则（只用于已人工审阅过的候选，在本进程内运行）。"""
    spec = importlib.util.spec_from_file_location("famou_candidate_rule", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return rule_from_module(module)
