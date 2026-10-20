from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import calculate_task_score


def generate_demo_candidates(
    output_path: str | Path,
    target_count: int = 40,
    station_count: int = 3,
    seed: int = 42,
) -> Path:
    rng = random.Random(seed)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    horizon = datetime(2026, 10, 1, tzinfo=timezone.utc)
    task_id = 1
    rows = []

    priorities = [("HIGH", 5.0), ("MEDIUM", 3.0), ("LOW", 2.0)]
    for target_number in range(1, target_count + 1):
        priority, base_score = priorities[(target_number - 1) % len(priorities)]
        window_count = rng.randint(2, 5)
        for window_number in range(window_count):
            station_id = str(rng.randint(1, station_count))
            start_offset = rng.randint(0, 72 * 60) * 60
            window_start = horizon + timedelta(seconds=start_offset)
            window_length = rng.choice([300, 420, 600, 900])
            window_end = window_start + timedelta(seconds=window_length)
            window_id = f"W-{target_number}-{window_number}"

            for duration in (120, 240):
                latest_offset = window_length - duration
                for candidate_offset in sorted({0, latest_offset // 2, latest_offset}):
                    start = window_start + timedelta(seconds=(candidate_offset // 30) * 30)
                    end = start + timedelta(seconds=duration)
                    if end > window_end:
                        continue
                    rows.append(
                        {
                            "task_id": str(task_id),
                            "window_id": window_id,
                            "station_id": station_id,
                            "target_id": str(10000 + target_number),
                            "priority": priority,
                            "window_start": window_start.isoformat().replace("+00:00", "Z"),
                            "window_end": window_end.isoformat().replace("+00:00", "Z"),
                            "start": start.isoformat().replace("+00:00", "Z"),
                            "end": end.isoformat().replace("+00:00", "Z"),
                            "duration_seconds": duration,
                            "base_score": base_score,
                            "task_score": calculate_task_score(base_score, duration),
                            "min_elevation": 12.0,
                            "max_elevation": 70.0,
                            "max_sun_elevation": -8.0,
                        }
                    )
                    task_id += 1

    fieldnames = list(rows[0])
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output

