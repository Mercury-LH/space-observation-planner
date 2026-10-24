from datetime import datetime, timezone

import pytest

from obsplan.config import Config


@pytest.mark.parametrize("start", [
    "2026-08-01T00:00:00",
    "2026-08-01T00:00:00Z",
    "2026-08-01T08:00:00+08:00",
    "2026-07-31T20:00:00-04:00",
])
def test_planning_epoch_preserves_the_instant(start):
    assert Config(start_utc=start).start_dt == datetime(2026, 8, 1, tzinfo=timezone.utc)
