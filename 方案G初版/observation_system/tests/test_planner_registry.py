from __future__ import annotations

import unittest

from observation_system.planners import create_planner, registered_planners
from scheme_g.models import PlannerConfig


class PlannerRegistryTests(unittest.TestCase):
    def test_registered_planners(self) -> None:
        names = registered_planners()
        self.assertIn("scheme_g", names)
        self.assertIn("priority_greedy", names)

    def test_scheme_g_is_adapter(self) -> None:
        planner = create_planner("scheme_g", PlannerConfig())
        self.assertEqual(planner.name, "scheme_g")


if __name__ == "__main__":
    unittest.main()

