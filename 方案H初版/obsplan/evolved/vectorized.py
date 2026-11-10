"""已采纳伐谋规则的 numpy 向量化等价实现。

伐谋生成的原始函数（recover_rule.py 里的 score）逐行用 Python 打分，全量约 200 万个候选要 3–4 秒。
这里用同样的运算、同样的先后顺序在 float64 数组上整列计算，结果逐位相同，约 0.05 秒。
原始实现保留不动；load_rule 每次调用都抽查一部分行与原始实现比对，不一致就退回原始实现。
"""
from __future__ import annotations

import numpy as np


def recover_rule(names, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """与 recover_rule.score 逐位相同（加法从左到右，与 Python 表达式的求值顺序一致）。"""
    col = {name: i for i, name in enumerate(names)}
    ts = X[:, col["task_score"]]
    pc = X[:, col["price_cost"]]
    boost = (X[:, col["sub_pick"]] * 2.0 + X[:, col["sub_pick_same_duration"]] * 1.0
             + (X[:, col["duration_cells"]] - 4.0) * 0.2 + X[:, col["urgency"]] * 0.5
             + X[:, col["scarcity"]] * 0.5 + X[:, col["target_dual"]] * 0.2)
    u_base = ts + X[:, col["cover_value"]] - pc + boost
    c_base = ts * 1.2 - pc * 0.9 + boost
    penalty = 1.0 + X[:, col["window_demand"]] * 0.05 + X[:, col["conflict"]] * 0.1 + X[:, col["slack_cells"]] * 0.005
    u = np.where(u_base > 0.0, u_base / penalty, u_base * penalty)
    c = np.where(c_base > 0.0, c_base / penalty, c_base * penalty)
    return u, c


VECTORIZED = {"recover_rule": recover_rule}
