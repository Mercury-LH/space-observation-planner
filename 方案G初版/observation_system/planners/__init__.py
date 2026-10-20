"""可替换规划器实现。

方案G的内部代码位于顶层 `scheme_g/`，本目录只放统一接口适配器。
"""

from .base import Planner
from .registry import create_planner, registered_planners

__all__ = ["Planner", "create_planner", "registered_planners"]

