from __future__ import annotations

from typing import Protocol

from scheme_g.models import CandidateTask, PlanResult


class Planner(Protocol):
    """所有调度方案必须实现的最小接口。"""

    name: str

    def plan(self, tasks: list[CandidateTask]) -> PlanResult:
        ...

