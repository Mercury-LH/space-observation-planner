"""Shareable synthetic ingestion fixtures; no private target records."""

from __future__ import annotations

from pathlib import Path

import pytest

from space_planner.data import DataAuditError, decode_catalog_number, load_dataset


def _checksum(line: str) -> str:
    assert len(line) == 68
    checksum = sum(int(c) if c.isdigit() else int(c == "-") for c in line) % 10
    return line + str(checksum)


def _tle(catalog: str = "00001", epoch: str = "26215.50000000") -> str:
    line1 = _checksum(f"1 {catalog}U 26001A   {epoch}  .00000000  00000-0  00000-0 0  999")
    line2 = _checksum(f"2 {catalog}  51.6000 100.0000 0001000  10.0000 350.0000 15.00000000    1")
    return f"0 SYNTHETIC\n{line1}\n{line2}\n"


def _dataset(tmp_path: Path, orbit: str | None = None) -> Path:
    (tmp_path / "数据").mkdir()
    (tmp_path / "优先级").mkdir()
    (tmp_path / "数据/轨道.txt").write_text(orbit if orbit is not None else _tle())
    (tmp_path / "数据/站址.txt").write_text(
        "站址：经度(度)，纬度(度)，海拔(m）\n"
        "1：290.0, 40.0,100\n"
        "设备观测限制：最低仰角不低于10度，最高不超过85度\n"
    )
    for name, content in [("高", "1"), ("中", ""), ("低", "")]:
        (tmp_path / f"优先级/{name}优先级目标.txt").write_text(content)
    return tmp_path


@pytest.mark.parametrize("encoded,decoded", [
    ("00001", 1), ("99999", 99999), ("A0000", 100000), ("H1234", 171234),
    ("J0000", 180000), ("N9999", 229999), ("P0000", 230000), ("Z9999", 339999),
])
def test_alpha5_decoding(encoded, decoded):
    assert decode_catalog_number(encoded) == decoded


@pytest.mark.parametrize("encoded", ["I0000", "O0000", "a0000", "100000", "123", "A00X0"])
def test_alpha5_rejects_invalid_identifiers(encoded):
    with pytest.raises(ValueError):
        decode_catalog_number(encoded)


def test_complete_dataset_with_alpha5_and_longitude_normalization(tmp_path):
    root = _dataset(tmp_path, _tle() + _tle("A0052"))
    (root / "优先级/中优先级目标.txt").write_text("100052")
    dataset = load_dataset(root)
    assert [t.target_id for t in dataset.targets] == [1, 100052]
    assert [t.priority for t in dataset.targets] == [5, 3]
    assert dataset.targets[0].epoch.isoformat() == "2026-08-03T12:00:00+00:00"
    assert dataset.stations[0].longitude_deg == -70
    assert dataset.stations[0].latitude_deg == 40
    assert dataset.stations[0].max_elevation_deg == 85
    assert dataset.audit["alpha5_count"] == 1
    assert dataset.audit["period_minutes"] == {"min": 96, "median": 96, "max": 96}
    assert len(dataset.audit["files"]) == 5
    assert all(len(v["sha256"]) == 64 for v in dataset.audit["files"].values())
    assert dataset.audit["valid"]


def test_priority_csv_accepts_spaces_and_trailing_newline(tmp_path):
    root = _dataset(tmp_path, _tle() + _tle("00002"))
    (root / "优先级/高优先级目标.txt").write_text("1, 2\n")
    assert len(load_dataset(root).targets) == 2


@pytest.mark.parametrize("change,error_code", [
    ("checksum", "invalid_tle"), ("mismatched_catalog", "invalid_tle"),
    ("missing_line", "missing_tle_line2"), ("duplicate_tle", "duplicate_tle_id"),
    ("missing_priority", "missing_priority"), ("extra_priority", "priority_without_tle"),
    ("duplicate_priority", "duplicate_priority_id"), ("invalid_csv", "invalid_priority_id"),
    ("invalid_latitude", "invalid_station"), ("duplicate_station", "duplicate_station_id"),
    ("unknown_station_line", "unknown_station_line"), ("missing_file", "file_unreadable"),
])
def test_malformed_data_rejected_with_audit(tmp_path, change, error_code):
    root = _dataset(tmp_path)
    orbit_path = root / "数据/轨道.txt"
    priority_path = root / "优先级/高优先级目标.txt"
    station_path = root / "数据/站址.txt"
    if change == "checksum":
        text = orbit_path.read_text().splitlines()
        text[1] = text[1][:-1] + str((int(text[1][-1]) + 1) % 10)
        orbit_path.write_text("\n".join(text))
    elif change == "mismatched_catalog":
        lines = orbit_path.read_text().splitlines()
        lines[2] = _checksum(lines[2][:2] + "00002" + lines[2][7:68])
        orbit_path.write_text("\n".join(lines))
    elif change == "missing_line":
        orbit_path.write_text("\n".join(orbit_path.read_text().splitlines()[:2]))
    elif change == "duplicate_tle":
        orbit_path.write_text(_tle() * 2)
    elif change == "missing_priority":
        priority_path.write_text("")
    elif change == "extra_priority":
        priority_path.write_text("1,2")
    elif change == "duplicate_priority":
        (root / "优先级/低优先级目标.txt").write_text("1")
    elif change == "invalid_csv":
        priority_path.write_text("1,")
    elif change == "invalid_latitude":
        station_path.write_text(station_path.read_text().replace("40.0", "91.0"))
    elif change == "duplicate_station":
        station_path.write_text(station_path.read_text() + "1:0,0,0\n")
    elif change == "unknown_station_line":
        station_path.write_text(station_path.read_text() + "mysterious configuration\n")
    elif change == "missing_file":
        priority_path.unlink()
    with pytest.raises(DataAuditError) as error:
        load_dataset(root)
    assert not error.value.audit["valid"]
    assert error_code in {item["code"] for item in error.value.audit["errors"]}


@pytest.mark.parametrize("epoch", ["26366.00000000", "26000.50000000"])
def test_invalid_epoch_day_is_rejected(tmp_path, epoch):
    with pytest.raises(DataAuditError, match="epoch day"):
        load_dataset(_dataset(tmp_path, _tle(epoch=epoch)))


def test_source_elevation_limits_are_loaded(tmp_path):
    root = _dataset(tmp_path)
    station = root / "数据/站址.txt"
    station.write_text(station.read_text().replace("10度", "15度").replace("85度", "80度"))
    result = load_dataset(root)
    assert result.stations[0].min_elevation_deg == 15
    assert result.stations[0].max_elevation_deg == 80


def test_missing_elevation_limits_are_explicit_default(tmp_path):
    root = _dataset(tmp_path)
    (root / "数据/站址.txt").write_text("1:0,0,0\n")
    result = load_dataset(root)
    assert result.audit["warnings"][0]["code"] == "default_elevation_limits"
