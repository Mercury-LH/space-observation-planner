from __future__ import annotations

import csv
from pathlib import Path

from scheme_g.utils import parse_datetime

from .domain import ObservationWindow


WINDOW_FIELDS = [
    "window_id",
    "station_id",
    "target_id",
    "priority",
    "base_score",
    "start",
    "end",
    "duration_seconds",
    "min_elevation",
    "max_elevation",
    "max_sun_elevation",
]


def write_windows(path: str | Path, windows: list[ObservationWindow]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WINDOW_FIELDS)
        writer.writeheader()
        for window in windows:
            writer.writerow(
                {
                    "window_id": window.window_id,
                    "station_id": window.station_id,
                    "target_id": window.target_id,
                    "priority": window.priority,
                    "base_score": window.base_score,
                    "start": window.start.isoformat().replace("+00:00", "Z"),
                    "end": window.end.isoformat().replace("+00:00", "Z"),
                    "duration_seconds": window.duration_seconds,
                    "min_elevation": window.min_elevation,
                    "max_elevation": window.max_elevation,
                    "max_sun_elevation": window.max_sun_elevation,
                }
            )


def _optional_float(value: str) -> float | None:
    return float(value) if value.strip() else None


def read_windows(path: str | Path) -> list[ObservationWindow]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            ObservationWindow(
                window_id=row["window_id"],
                station_id=row["station_id"],
                target_id=row["target_id"],
                priority=row["priority"],
                base_score=float(row["base_score"]),
                start=parse_datetime(row["start"]),
                end=parse_datetime(row["end"]),
                min_elevation=_optional_float(row.get("min_elevation", "")),
                max_elevation=_optional_float(row.get("max_elevation", "")),
                max_sun_elevation=_optional_float(row.get("max_sun_elevation", "")),
            )
            for row in reader
        ]

