from __future__ import annotations

from collections.abc import Callable

from scheme_g.models import PlannerConfig

from .base import Planner
from .greedy_adapter import GreedyPlannerAdapter
from .scheme_g_adapter import SchemeGPlannerAdapter


PlannerFactory = Callable[[PlannerConfig], Planner]


_REGISTRY: dict[str, PlannerFactory] = {
    # ==================== 方案G接入注册 ====================
    "scheme_g": lambda config: SchemeGPlannerAdapter(config),
    # ==================== 方案G接入注册结束 =================
    "random_greedy": lambda config: GreedyPlannerAdapter(config, "random"),
    "priority_greedy": lambda config: GreedyPlannerAdapter(config, "priority"),
    "ai_greedy": lambda config: GreedyPlannerAdapter(config, "ai"),
}


def register_planner(name: str, factory: PlannerFactory) -> None:
    if name in _REGISTRY:
        raise ValueError(f"规划器已存在: {name}")
    _REGISTRY[name] = factory


def create_planner(name: str, config: PlannerConfig) -> Planner:
    try:
        return _REGISTRY[name](config)
    except KeyError as exc:
        raise ValueError(
            f"未知规划器{name!r}，可选值: {', '.join(sorted(_REGISTRY))}"
        ) from exc


def registered_planners() -> list[str]:
    return sorted(_REGISTRY)

