from __future__ import annotations

import unittest
from pathlib import Path

from observation_system.config import load_system_config
from observation_system.data_loader import (
    decode_alpha5_catalog_id,
    load_competition_data,
    normalize_longitude,
)


ROOT = Path(__file__).resolve().parents[2]


class DataLoaderTests(unittest.TestCase):
    def test_alpha5(self) -> None:
        self.assertEqual(decode_alpha5_catalog_id("A0045"), "100045")
        self.assertEqual(decode_alpha5_catalog_id("T0288"), "270288")
        self.assertEqual(decode_alpha5_catalog_id("50574"), "50574")

    def test_longitude_normalization(self) -> None:
        self.assertAlmostEqual(normalize_longitude(290.458202), -69.541798)
        self.assertAlmostEqual(normalize_longitude(358.445658), -1.554342)

    def test_real_data_audit(self) -> None:
        config = load_system_config(ROOT / "observation_system/config/default.json")
        if not Path(config.data.tle_path).exists() or not Path(config.data.station_path).exists():
            self.skipTest("未提供不随Git发布的比赛原始数据")
        targets, stations, audit = load_competition_data(
            config.data.tle_path,
            config.data.station_path,
            config.data.priority_dir,
        )
        self.assertEqual(len(targets), 10000)
        self.assertEqual(len(stations), 10)
        self.assertTrue(audit.valid)
        self.assertEqual(audit.priority_counts["HIGH"], 2024)
        self.assertEqual(audit.priority_counts["MEDIUM"], 3990)
        self.assertEqual(audit.priority_counts["LOW"], 3986)


if __name__ == "__main__":
    unittest.main()
