from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from importlib.resources import files
from typing import Callable

from .config import ScenarioConfig, VisibilityConfig
from .domain import GroundStation, ObservationWindow, Target


@dataclass(slots=True)
class VisibilityComputationResult:
    windows: list[ObservationWindow]
    failures: list[str]
    processed_pairs: int


def _compute_visibility_worker(
    scenario: ScenarioConfig,
    config: VisibilityConfig,
    targets: list[Target],
    stations: list[GroundStation],
) -> VisibilityComputationResult:
    worker_config = replace(config, workers=1, target_limit=None, station_limit=None)
    engine = SkyfieldVisibilityEngine(scenario, worker_config)
    return engine._compute_sequential(targets, stations)


class SkyfieldVisibilityEngine:
    """基于SGP4和Skyfield计算合法观测窗口。

    先用10度高度事件定位宽泛过境，再在过境内部联合判断85度上限与
    太阳高度角，避免对完整6天逐秒扫描。
    """

    def __init__(self, scenario: ScenarioConfig, config: VisibilityConfig):
        self.scenario = scenario
        self.config = config
        self._initialize_skyfield()

    def _initialize_skyfield(self) -> None:
        try:
            from skyfield.api import load, load_file
        except ImportError as exc:
            raise RuntimeError(
                "缺少Skyfield，请执行: .venv/Scripts/python.exe -m pip install -r requirements.txt"
            ) from exc

        self.ts = load.timescale(builtin=True)
        ephemeris = files("skyfield_data").joinpath("data", "de421.bsp")
        self.eph = load_file(str(ephemeris))
        self.earth = self.eph["earth"]
        self.sun = self.eph["sun"]

    def _build_station(self, station: GroundStation):
        from skyfield.api import wgs84

        return wgs84.latlon(
            latitude_degrees=station.latitude_deg,
            longitude_degrees=station.longitude_deg,
            elevation_m=station.elevation_m,
        )

    def _build_satellite(self, target: Target):
        from skyfield.api import EarthSatellite

        return EarthSatellite(
            target.tle_line1,
            target.tle_line2,
            target.name,
            self.ts,
        )

    def _altitudes(
        self,
        satellite,
        station_topos,
        datetimes: list[datetime],
    ) -> tuple[list[float], list[float]]:
        times = self.ts.from_datetimes(datetimes)
        target_altitude = (satellite - station_topos).at(times).altaz()[0].degrees
        observer = self.earth + station_topos
        sun_altitude = observer.at(times).observe(self.sun).apparent().altaz()[0].degrees
        return list(target_altitude), list(sun_altitude)

    def _is_legal(self, satellite, station_topos, moment: datetime) -> bool:
        target_altitude, sun_altitude = self._altitudes(
            satellite,
            station_topos,
            [moment],
        )
        elevation = target_altitude[0]
        sun = sun_altitude[0]
        return (
            self.scenario.minimum_elevation_deg <= elevation
            <= self.scenario.maximum_elevation_deg
            and sun < self.scenario.maximum_sun_elevation_deg
        )

    def _refine_transition(
        self,
        satellite,
        station_topos,
        left: datetime,
        right: datetime,
        opening: bool,
    ) -> datetime:
        tolerance = self.config.boundary_tolerance_seconds
        while (right - left).total_seconds() > tolerance:
            midpoint = left + (right - left) / 2
            legal = self._is_legal(satellite, station_topos, midpoint)
            if opening:
                if legal:
                    right = midpoint
                else:
                    left = midpoint
            else:
                if legal:
                    left = midpoint
                else:
                    right = midpoint
        return right if opening else left

    def _broad_passes(self, satellite, station_topos) -> list[tuple[datetime, datetime]]:
        start = self.scenario.start_time.astimezone(timezone.utc)
        end = self.scenario.end_time.astimezone(timezone.utc)
        start_time = self.ts.from_datetime(start)
        end_time = self.ts.from_datetime(end)
        event_times, events = satellite.find_events(
            station_topos,
            start_time,
            end_time,
            altitude_degrees=self.scenario.minimum_elevation_deg,
        )

        start_altitude = (satellite - station_topos).at(start_time).altaz()[0].degrees
        current_start = start if float(start_altitude) >= self.scenario.minimum_elevation_deg else None
        result: list[tuple[datetime, datetime]] = []

        event_datetimes = event_times.utc_datetime()
        for event_time, event in zip(event_datetimes, events):
            moment = event_time.astimezone(timezone.utc)
            if int(event) == 0:
                current_start = max(moment, start)
            elif int(event) == 2 and current_start is not None:
                interval_end = min(moment, end)
                if interval_end > current_start:
                    result.append((current_start, interval_end))
                current_start = None

        if current_start is not None and current_start < end:
            result.append((current_start, end))
        return result

    def _sample_times(self, start: datetime, end: datetime) -> list[datetime]:
        step = timedelta(seconds=self.config.sample_step_seconds)
        count = max(1, math.ceil((end - start).total_seconds() / step.total_seconds()))
        values = [start + step * index for index in range(count)]
        if not values or values[-1] != end:
            values.append(end)
        return [min(value, end) for value in values]

    def _legal_subwindows(
        self,
        satellite,
        station_topos,
        broad_start: datetime,
        broad_end: datetime,
    ) -> list[tuple[datetime, datetime, float, float, float]]:
        sample_times = self._sample_times(broad_start, broad_end)
        target_altitudes, sun_altitudes = self._altitudes(
            satellite,
            station_topos,
            sample_times,
        )
        legal = [
            self.scenario.minimum_elevation_deg <= target_altitude
            <= self.scenario.maximum_elevation_deg
            and sun_altitude < self.scenario.maximum_sun_elevation_deg
            for target_altitude, sun_altitude in zip(target_altitudes, sun_altitudes)
        ]

        intervals: list[tuple[datetime, datetime, float, float, float]] = []
        current_start: datetime | None = sample_times[0] if legal[0] else None
        run_indices: list[int] = [0] if legal[0] else []

        for index in range(1, len(sample_times)):
            if not legal[index - 1] and legal[index]:
                current_start = self._refine_transition(
                    satellite,
                    station_topos,
                    sample_times[index - 1],
                    sample_times[index],
                    opening=True,
                )
                run_indices = [index]
            elif legal[index]:
                run_indices.append(index)

            if legal[index - 1] and not legal[index] and current_start is not None:
                current_end = self._refine_transition(
                    satellite,
                    station_topos,
                    sample_times[index - 1],
                    sample_times[index],
                    opening=False,
                )
                if (current_end - current_start).total_seconds() >= self.scenario.minimum_window_seconds:
                    values = run_indices or [index - 1]
                    intervals.append(
                        (
                            current_start,
                            current_end,
                            min(target_altitudes[item] for item in values),
                            max(target_altitudes[item] for item in values),
                            max(sun_altitudes[item] for item in values),
                        )
                    )
                current_start = None
                run_indices = []

        if current_start is not None:
            current_end = sample_times[-1]
            if (current_end - current_start).total_seconds() >= self.scenario.minimum_window_seconds:
                values = run_indices or [len(sample_times) - 1]
                intervals.append(
                    (
                        current_start,
                        current_end,
                        min(target_altitudes[item] for item in values),
                        max(target_altitudes[item] for item in values),
                        max(sun_altitudes[item] for item in values),
                    )
                )
        return intervals

    def _compute_sequential(
        self,
        targets: list[Target],
        stations: list[GroundStation],
        progress: Callable[[int, int], None] | None = None,
    ) -> VisibilityComputationResult:
        targets = targets[: self.config.target_limit] if self.config.target_limit else targets
        stations = stations[: self.config.station_limit] if self.config.station_limit else stations
        total_pairs = len(targets) * len(stations)
        processed = 0
        failures: list[str] = []
        windows: list[ObservationWindow] = []

        for station in stations:
            station_topos = self._build_station(station)
            for target in targets:
                try:
                    satellite = self._build_satellite(target)
                    window_number = 0
                    for broad_start, broad_end in self._broad_passes(satellite, station_topos):
                        for start, end, minimum, maximum, sun_maximum in self._legal_subwindows(
                            satellite,
                            station_topos,
                            broad_start,
                            broad_end,
                        ):
                            window_number += 1
                            windows.append(
                                ObservationWindow(
                                    window_id=(
                                        f"{station.station_id}:{target.target_id}:"
                                        f"{window_number}"
                                    ),
                                    station_id=station.station_id,
                                    target_id=target.target_id,
                                    priority=target.priority,
                                    base_score=target.base_score,
                                    start=start,
                                    end=end,
                                    min_elevation=minimum,
                                    max_elevation=maximum,
                                    max_sun_elevation=sun_maximum,
                                )
                            )
                except Exception as exc:  # 单颗目标异常不能中断全部10000颗计算
                    failures.append(
                        f"站{station.station_id}/目标{target.target_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                processed += 1
                if progress:
                    progress(processed, total_pairs)

        windows.sort(key=lambda item: (item.station_id, item.start, item.target_id))
        return VisibilityComputationResult(windows, failures, processed)

    def compute(
        self,
        targets: list[Target],
        stations: list[GroundStation],
        progress: Callable[[int, int], None] | None = None,
    ) -> VisibilityComputationResult:
        targets = targets[: self.config.target_limit] if self.config.target_limit else targets
        stations = stations[: self.config.station_limit] if self.config.station_limit else stations
        workers = max(1, min(self.config.workers, len(targets)))
        if workers == 1 or len(targets) < 2:
            return self._compute_sequential(targets, stations, progress=progress)

        chunk_size = math.ceil(len(targets) / workers)
        chunks = [
            targets[index : index + chunk_size]
            for index in range(0, len(targets), chunk_size)
        ]
        total_pairs = len(targets) * len(stations)
        processed = 0
        windows: list[ObservationWindow] = []
        failures: list[str] = []

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _compute_visibility_worker,
                    self.scenario,
                    self.config,
                    chunk,
                    stations,
                )
                for chunk in chunks
            ]
            for future in as_completed(futures):
                result = future.result()
                windows.extend(result.windows)
                failures.extend(result.failures)
                processed += result.processed_pairs
                if progress:
                    progress(processed, total_pairs)

        windows.sort(key=lambda item: (item.station_id, item.start, item.target_id))
        return VisibilityComputationResult(windows, failures, processed)
