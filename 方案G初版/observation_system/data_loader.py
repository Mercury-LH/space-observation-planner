from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from scheme_g.models import PRIORITY_SCORES

from .domain import GroundStation, Target


ALPHA5_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ"
ALPHA5_VALUES = {letter: 10 + index for index, letter in enumerate(ALPHA5_ALPHABET)}


@dataclass(slots=True)
class DataAudit:
    target_count: int
    station_count: int
    priority_counts: dict[str, int]
    missing_priority_ids: list[str]
    priority_ids_without_tle: list[str]
    duplicate_target_ids: list[str]

    @property
    def valid(self) -> bool:
        return not (
            self.missing_priority_ids
            or self.priority_ids_without_tle
            or self.duplicate_target_ids
        )


def decode_alpha5_catalog_id(value: str) -> str:
    token = value.strip().upper()
    if token.isdigit():
        return str(int(token))
    if len(token) != 5 or token[0] not in ALPHA5_VALUES or not token[1:].isdigit():
        raise ValueError(f"无法解析TLE Alpha-5编号: {value!r}")
    return str(ALPHA5_VALUES[token[0]] * 10000 + int(token[1:]))


def load_priority_map(priority_dir: str | Path) -> dict[str, str]:
    directory = Path(priority_dir)
    files = {
        "HIGH": directory / "高优先级目标.txt",
        "MEDIUM": directory / "中优先级目标.txt",
        "LOW": directory / "低优先级目标.txt",
    }
    result: dict[str, str] = {}
    for priority, path in files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        for identifier in re.findall(r"\d+", path.read_text(encoding="utf-8-sig")):
            normalized = str(int(identifier))
            if normalized in result:
                raise ValueError(
                    f"目标{normalized}同时出现在{result[normalized]}和{priority}优先级"
                )
            result[normalized] = priority
    return result


def load_tle_targets(
    tle_path: str | Path,
    priority_map: dict[str, str],
) -> tuple[list[Target], list[str]]:
    lines = [
        line.rstrip("\r\n")
        for line in Path(tle_path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    targets: list[Target] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    index = 0

    while index < len(lines):
        if not lines[index].startswith("0 "):
            raise ValueError(f"TLE第{index + 1}行不是名称行: {lines[index]!r}")
        if index + 2 >= len(lines):
            raise ValueError("TLE文件末尾记录不完整")
        name = lines[index][2:].strip()
        line1 = lines[index + 1]
        line2 = lines[index + 2]
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            raise ValueError(f"目标{name}的TLE两行格式错误")
        raw_catalog = line1[2:7]
        target_id = decode_alpha5_catalog_id(raw_catalog)
        if target_id in seen:
            duplicates.append(target_id)
        seen.add(target_id)
        priority = priority_map.get(target_id, "UNASSIGNED")
        targets.append(
            Target(
                target_id=target_id,
                tle_catalog_id=raw_catalog.strip(),
                name=name,
                tle_line1=line1,
                tle_line2=line2,
                priority=priority,
                base_score=PRIORITY_SCORES.get(priority, 0.0),
            )
        )
        index += 3
    return targets, duplicates


def normalize_longitude(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0


def load_stations(path: str | Path) -> list[GroundStation]:
    stations: list[GroundStation] = []
    pattern = re.compile(
        r"^\s*(\d+)\s*[：:]\s*"
        r"([-+]?\d+(?:\.\d+)?)\s*,\s*"
        r"([-+]?\d+(?:\.\d+)?)\s*,\s*"
        r"([-+]?\d+(?:\.\d+)?)\s*$"
    )
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        station_id, longitude, latitude, elevation = match.groups()
        longitude_value = normalize_longitude(float(longitude))
        latitude_value = float(latitude)
        if not -90.0 <= latitude_value <= 90.0:
            raise ValueError(f"站点{station_id}纬度不合法: {latitude_value}")
        stations.append(
            GroundStation(
                station_id=station_id,
                longitude_deg=longitude_value,
                latitude_deg=latitude_value,
                elevation_m=float(elevation),
            )
        )
    if not stations:
        raise ValueError(f"未从站址文件读取到站点: {path}")
    return stations


def load_competition_data(
    tle_path: str | Path,
    station_path: str | Path,
    priority_dir: str | Path,
) -> tuple[list[Target], list[GroundStation], DataAudit]:
    priority_map = load_priority_map(priority_dir)
    targets, duplicates = load_tle_targets(tle_path, priority_map)
    stations = load_stations(station_path)
    target_ids = {target.target_id for target in targets}
    priority_ids = set(priority_map)
    priority_counts = {
        priority: sum(1 for value in priority_map.values() if value == priority)
        for priority in ("HIGH", "MEDIUM", "LOW")
    }
    audit = DataAudit(
        target_count=len(targets),
        station_count=len(stations),
        priority_counts=priority_counts,
        missing_priority_ids=sorted(target_ids - priority_ids, key=int),
        priority_ids_without_tle=sorted(priority_ids - target_ids, key=int),
        duplicate_target_ids=sorted(set(duplicates), key=int),
    )
    return targets, stations, audit

