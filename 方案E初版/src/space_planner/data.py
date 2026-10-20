"""Strict ingestion of the organizer's local TLE, station and priority files.

Times are timezone-aware UTC; station coordinates are geodetic degrees and
metres. ``Target.priority`` is the published score weight (5, 3 or 2), not an
ordinal rank. Input errors are never repaired or silently dropped.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import calendar
import hashlib
import math
from pathlib import Path
import re
import statistics
from typing import Any

from .types import Station, Target


ALPHA5_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
PRIORITY_FILES = {
    "优先级/高优先级目标.txt": 5,
    "优先级/中优先级目标.txt": 3,
    "优先级/低优先级目标.txt": 2,
}
REQUIRED_FILES = ("数据/轨道.txt", "数据/站址.txt", *PRIORITY_FILES)


@dataclass
class Dataset:
    targets: list[Target]
    stations: list[Station]
    audit: dict[str, Any]


class DataAuditError(ValueError):
    """The input was rejected; ``audit`` contains structured diagnostic data."""

    def __init__(self, audit: dict[str, Any]):
        self.audit = audit
        errors = audit["errors"]
        examples = "; ".join(item["message"] for item in errors[:3])
        super().__init__(f"Dataset rejected: {len(errors)} error(s): {examples}")


def decode_catalog_number(value: str) -> int:
    """Decode the five-column TLE identifier, including Space-Track Alpha5.

    Letters I and O are absent from the alphabet: A0000=100000,
    H0000=170000, J0000=180000, Z9999=339999. Decimal priority-file IDs are
    parsed separately and must not be truncated to five columns.
    """
    if not re.fullmatch(r"[0-9]{5}|[A-HJ-NP-Z][0-9]{4}", value):
        raise ValueError(f"Invalid five-column TLE catalogue number {value!r}")
    if value.isdigit():
        return int(value)
    return (ALPHA5_LETTERS.index(value[0]) + 10) * 10000 + int(value[1:])


def _issue(
    audit: dict[str, Any], code: str, message: str, **context: Any
) -> None:
    audit["errors"].append({"code": code, "message": message, **context})


def _read_inputs(root: Path, audit: dict[str, Any]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for relative in REQUIRED_FILES:
        path = root / relative
        try:
            contents = path.read_bytes()
        except OSError as error:
            _issue(audit, "file_unreadable", str(error), file=relative)
            continue
        audit["files"][relative] = {
            "sha256": hashlib.sha256(contents).hexdigest(),
            "size_bytes": len(contents),
        }
        try:
            texts[relative] = contents.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            _issue(audit, "invalid_encoding", str(error), file=relative)
    return texts


def _load_priorities(texts: dict[str, str], audit: dict[str, Any]) -> dict[int, int]:
    priorities: dict[int, int] = {}
    counts: dict[str, int] = {}
    for filename, weight in PRIORITY_FILES.items():
        if filename not in texts:
            continue
        content = texts[filename].strip()
        # Empty classes are useful for synthetic subsets; no empty CSV fields.
        tokens = content.split(",") if content else []
        counts[str(weight)] = len(tokens)
        for index, token in enumerate(tokens, start=1):
            token = token.strip()
            if not re.fullmatch(r"[0-9]+", token) or int(token) <= 0:
                _issue(
                    audit, "invalid_priority_id", f"Invalid priority ID {token!r}",
                    file=filename, entry=index,
                )
                continue
            target_id = int(token)
            if target_id in priorities:
                _issue(
                    audit, "duplicate_priority_id",
                    f"Target {target_id} has repeated priority membership",
                    file=filename, entry=index, target_id=target_id,
                    previous_weight=priorities[target_id], weight=weight,
                )
            else:
                priorities[target_id] = weight
    audit["priority_counts_by_weight"] = counts
    return priorities


def _validate_tle_line(line: str, number: int) -> None:
    if len(line) != 69:
        raise ValueError(f"TLE line {number} must contain 69 columns, got {len(line)}")
    if not line.startswith(f"{number} ") or not line[68].isdigit():
        raise ValueError(f"Invalid TLE line {number} marker or checksum field")
    if not line.isascii():
        raise ValueError("TLE fields must be ASCII")
    expected = sum(int(c) if c.isdigit() else int(c == "-") for c in line[:68]) % 10
    if expected != int(line[68]):
        raise ValueError(
            f"TLE line {number} checksum {line[68]} does not match {expected}"
        )
    separators = (8, 17, 32, 43, 52, 61, 63) if number == 1 else (7, 16, 25, 33, 42, 51)
    if any(line[index] != " " for index in separators):
        raise ValueError(f"TLE line {number} has invalid field separators")


def _parse_orbit(line1: str, line2: str) -> tuple[int, datetime, float, float, float]:
    _validate_tle_line(line1, 1)
    _validate_tle_line(line2, 2)
    target_id = decode_catalog_number(line1[2:7])
    if target_id == 0:
        raise ValueError("Catalogue number zero is invalid")
    if decode_catalog_number(line2[2:7]) != target_id:
        raise ValueError("Catalogue numbers differ between TLE lines")
    if line1[7] not in "UCS":
        raise ValueError("Invalid TLE classification")
    if not re.fullmatch(r"[0-9 ]{2}[0-9 ]{3}[A-Z0-9 ]{3}", line1[9:17]):
        raise ValueError("Invalid international designator")
    if not re.fullmatch(r"[0-9]{2}[0-9]{3}\.[0-9]{8}", line1[18:32]):
        raise ValueError("Invalid TLE epoch")
    if not re.fullmatch(r"[ +\-]\.[0-9]{8}", line1[33:43]):
        raise ValueError("Invalid first mean-motion derivative")
    for field in (line1[44:52], line1[53:61]):
        if not re.fullmatch(r"[ +\-][0-9]{5}[+\-][0-9]", field):
            raise ValueError("Invalid TLE implied-exponent field")
    if not line1[62].isdigit() or not line1[64:68].strip().isdigit():
        raise ValueError("Invalid TLE ephemeris or element-set field")
    if not re.fullmatch(r"[0-9]{7}", line2[26:33]):
        raise ValueError("Invalid TLE eccentricity")
    if not line2[63:68].strip().isdigit():
        raise ValueError("Invalid TLE revolution number")
    for field in (line2[8:16], line2[17:25], line2[34:42], line2[43:51]):
        if not re.fullmatch(r" *[0-9]{1,3}\.[0-9]{4}", field):
            raise ValueError(f"Invalid TLE angular field {field!r}")
    if not re.fullmatch(r" *[0-9]{1,2}\.[0-9]{8}", line2[52:63]):
        raise ValueError("Invalid mean-motion field")
    yy = int(line1[18:20])
    year = 2000 + yy if yy < 57 else 1900 + yy
    day = Decimal(line1[20:32])
    if not 1 <= day < 366 + calendar.isleap(year):
        raise ValueError("TLE epoch day is outside its calendar year")
    epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
        seconds=float((day - 1) * Decimal(86400))
    )
    inclination = float(line2[8:16])
    angles = [float(line2[a:b]) for a, b in ((17, 25), (34, 42), (43, 51))]
    if not 0 <= inclination <= 180 or any(not 0 <= angle < 360 for angle in angles):
        raise ValueError("TLE angle is outside its physical range")
    mean_motion = float(line2[52:63])
    if mean_motion <= 0:
        raise ValueError("Mean motion must be positive")
    eccentricity = float("0." + line2[26:33])
    return target_id, epoch, 1440.0 / mean_motion, eccentricity, inclination


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def _load_targets(
    text: str, priorities: dict[int, int], audit: dict[str, Any]
) -> list[Target]:
    records = [(i, line) for i, line in enumerate(text.splitlines(), 1) if line.strip()]
    targets: list[Target] = []
    seen: set[int] = set()
    periods: list[float] = []
    eccentricities: list[float] = []
    inclinations: list[float] = []
    alpha5_count = 0
    record_count = 0
    index = 0
    while index < len(records):
        line_number, line1 = records[index]
        if line1.startswith("0 "):
            index += 1
            if index == len(records):
                _issue(audit, "orphan_tle_name", "TLE name has no orbital lines", line=line_number)
                break
            line_number, line1 = records[index]
        if not line1.startswith("1 "):
            _issue(audit, "invalid_tle_sequence", "Expected TLE line 1", line=line_number)
            index += 1
            continue
        if index + 1 >= len(records) or not records[index + 1][1].startswith("2 "):
            _issue(audit, "missing_tle_line2", "Missing TLE line 2", line=line_number)
            index += 1
            continue
        _, line2 = records[index + 1]
        index += 2
        record_count += 1
        try:
            target_id, epoch, period, eccentricity, inclination = _parse_orbit(line1, line2)
        except (ValueError, InvalidOperation) as error:
            _issue(
                audit, "invalid_tle", str(error),
                file="数据/轨道.txt", line=line_number,
            )
            continue
        if target_id in seen:
            _issue(
                audit, "duplicate_tle_id", f"Duplicate orbit for target {target_id}",
                line=line_number, target_id=target_id,
            )
            continue
        seen.add(target_id)
        if target_id not in priorities:
            _issue(
                audit, "missing_priority", f"No priority for target {target_id}",
                target_id=target_id,
            )
            continue
        targets.append(Target(
            target_id=target_id, line1=line1, line2=line2,
            priority=priorities[target_id], epoch=epoch,
        ))
        periods.append(period)
        eccentricities.append(eccentricity)
        inclinations.append(inclination)
        alpha5_count += int(line1[2].isalpha())
    for target_id in sorted(priorities.keys() - seen):
        _issue(
            audit, "priority_without_tle", f"No valid orbit for priority target {target_id}",
            target_id=target_id,
        )
    if not targets:
        _issue(audit, "empty_targets", "No valid complete target records")
    audit.update({
        "tle_record_count": record_count,
        "target_count": len(targets),
        "alpha5_count": alpha5_count,
        "period_minutes": _summary(periods),
        "period_bins": {
            "le_128_minutes": sum(p <= 128 for p in periods),
            "gt_128_le_1200_minutes": sum(128 < p <= 1200 for p in periods),
            "gt_1200_minutes": sum(p > 1200 for p in periods),
        },
        "eccentricity": _summary(eccentricities),
        "inclination_degrees": _summary(inclinations),
        "epoch_min_utc": min(t.epoch for t in targets).isoformat() if targets else None,
        "epoch_max_utc": max(t.epoch for t in targets).isoformat() if targets else None,
    })
    return targets


def _load_stations(text: str, audit: dict[str, Any]) -> list[Station]:
    stations: list[Station] = []
    seen: set[str] = set()
    rows: list[tuple[int, str]] = []
    min_elevation, max_elevation = 10.0, 85.0
    limit_lines = 0
    for line_number, original in enumerate(text.splitlines(), 1):
        line = original.strip().replace("：", ":").replace("，", ",")
        if not line:
            continue
        if line.startswith("站址:") and all(unit in line for unit in ("经度", "纬度", "海拔")):
            continue
        if line.startswith("设备观测限制:"):
            limits = re.fullmatch(
                r"设备观测限制:最低仰角不低于(\d+(?:\.\d+)?)度,"
                r"最高不超过(\d+(?:\.\d+)?)度", line,
            )
            if limits is None:
                _issue(audit, "invalid_elevation_limits", "Cannot parse station elevation limits",
                       line=line_number)
            else:
                min_elevation, max_elevation = map(float, limits.groups())
                limit_lines += 1
            continue
        if re.match(r"[0-9]+:", line):
            rows.append((line_number, line))
        else:
            _issue(audit, "unknown_station_line", "Unrecognized station-file content", line=line_number)
    if limit_lines > 1:
        _issue(audit, "duplicate_elevation_limits", "Multiple station elevation-limit records")
    if not 0 <= min_elevation < max_elevation <= 90:
        _issue(audit, "invalid_elevation_range", "Require 0 <= min elevation < max elevation <= 90")
    if not limit_lines:
        audit["warnings"].append({
            "code": "default_elevation_limits",
            "message": "No station limit line; explicitly using dataset defaults 10–85 degrees",
        })
    normalizations: list[dict[str, Any]] = []
    for line_number, row in rows:
        station_id, values = row.split(":", 1)
        if station_id in seen:
            _issue(audit, "duplicate_station_id", f"Duplicate station {station_id}", line=line_number)
            continue
        seen.add(station_id)
        try:
            fields = [float(value.strip()) for value in values.split(",")]
            if len(fields) != 3:
                raise ValueError("Expected longitude,latitude,altitude")
            longitude, latitude, altitude = fields
            if not all(math.isfinite(value) for value in fields):
                raise ValueError("Station coordinates must be finite")
            if not -180 <= longitude <= 360 or not -90 <= latitude <= 90:
                raise ValueError("Station longitude or latitude out of range")
            normalized = longitude if -180 <= longitude < 180 else longitude - 360
            if normalized != longitude:
                normalizations.append({
                    "station_id": station_id, "original_longitude_deg": longitude,
                    "normalized_longitude_deg": normalized,
                })
            stations.append(Station(
                station_id=station_id, latitude_deg=latitude,
                longitude_deg=normalized, altitude_m=altitude,
                min_elevation_deg=min_elevation, max_elevation_deg=max_elevation,
            ))
        except ValueError as error:
            _issue(audit, "invalid_station", str(error), line=line_number, station_id=station_id)
    if not stations:
        _issue(audit, "empty_stations", "No valid station records")
    audit["station_count"] = len(stations)
    audit["station_longitude_normalizations"] = normalizations
    audit["elevation_limits_degrees"] = {"min": min_elevation, "max": max_elevation}
    return stations


def load_dataset(root: Path) -> Dataset:
    """Load ``开运联合`` itself, not a stale copy or its parent directory.

    The source files remain untouched. On malformed input, this raises
    :class:`DataAuditError` with a JSON-serializable audit, instead of returning
    a partially valid dataset. There is intentionally no skip-bad-record mode.
    """
    root = Path(root).expanduser().resolve()
    audit: dict[str, Any] = {
        "schema_version": 1,
        "source_root": str(root),
        "files": {},
        "errors": [],
        "warnings": [],
        "checksum_policy": "strict_reject",
        "time_scale": "UTC",
        "priority_semantics": "score_weight_5_3_2",
        "expected_size_policy": "counts_audited_not_hardcoded",
    }
    texts = _read_inputs(root, audit)
    priorities = _load_priorities(texts, audit)
    targets = _load_targets(texts.get("数据/轨道.txt", ""), priorities, audit)
    stations = _load_stations(texts.get("数据/站址.txt", ""), audit)
    audit["valid"] = not audit["errors"]
    audit["loaded_priority_counts_by_weight"] = {
        str(key): value for key, value in sorted(Counter(t.priority for t in targets).items())
    }
    if audit["errors"]:
        raise DataAuditError(audit)
    return Dataset(targets=targets, stations=stations, audit=audit)
