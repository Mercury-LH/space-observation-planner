from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Target:
    target_id: str
    tle_catalog_id: str
    name: str
    tle_line1: str
    tle_line2: str
    priority: str
    base_score: float


@dataclass(frozen=True, slots=True)
class GroundStation:
    station_id: str
    longitude_deg: float
    latitude_deg: float
    elevation_m: float


@dataclass(frozen=True, slots=True)
class ObservationWindow:
    window_id: str
    station_id: str
    target_id: str
    priority: str
    base_score: float
    start: datetime
    end: datetime
    min_elevation: float | None = None
    max_elevation: float | None = None
    max_sun_elevation: float | None = None

    @property
    def duration_seconds(self) -> float:
        return (self.end - self.start).total_seconds()

