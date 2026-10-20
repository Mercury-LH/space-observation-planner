from __future__ import annotations

from scheme_g.candidate_builder import VisibilityWindow, build_candidates_from_windows
from scheme_g.models import CandidateTask

from .config import CandidateConfig
from .domain import ObservationWindow


def build_candidates(
    windows: list[ObservationWindow],
    config: CandidateConfig,
) -> list[CandidateTask]:
    adapter_windows = [
        VisibilityWindow(
            window_id=window.window_id,
            station_id=window.station_id,
            target_id=window.target_id,
            priority=window.priority,
            start=window.start,
            end=window.end,
            base_score=window.base_score,
            min_elevation=window.min_elevation,
            max_elevation=window.max_elevation,
            max_sun_elevation=window.max_sun_elevation,
        )
        for window in windows
    ]
    return build_candidates_from_windows(
        adapter_windows,
        durations_seconds=config.durations_seconds,
        grid_seconds=config.grid_seconds,
        max_starts_per_window_duration=config.max_starts_per_window_duration,
    )

