from __future__ import annotations

from scheme_g.models import CandidateTask, PlanResult, PlannerConfig
from scheme_g.planner import SchemeGPlanner


class SchemeGPlannerAdapter:
    """方案G接入点。

    如需替换方案G，不修改前后流程，只新增另一个符合Planner协议的适配器。
    """

    name = "scheme_g"

    def __init__(self, config: PlannerConfig):
        self._planner = SchemeGPlanner(config)

    def plan(self, tasks: list[CandidateTask]) -> PlanResult:
        return self._planner.plan(tasks)

