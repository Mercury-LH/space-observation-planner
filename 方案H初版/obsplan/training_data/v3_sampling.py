"""v0.3 补充场景：同一个生成器（v2_sampling.generate_catalog），只改抽样参数，补上 v0.2 覆盖不到的三类情况。

v0.2 与赛题的差距（见 results/judge_data_audit.json）：
- 低轨高度：v0.2 最低 550 km、中位约 1000 km；真实目录里大量目标集中在 450～600 km 的星座层，
  还有一批 98° 左右的太阳同步轨道。低轨越低，过境越快、窗口越短。
- 高纬度站：v0.2 站点最高 52°；高纬度夏天夜很短，站的可用时间少。
- 中轨比例：v0.2 中位 19%，偏高。
- 拥挤：v0.2 大多数场景比较宽松。

做法（不用赛题的任何目标或站点）：
- 低轨高度 = 宽分布（均值 600～1100 km）+ 星座层（每个场景随机一个 460～620 km 的高度和一个倾角，
  近圆轨道）+ 太阳同步组（倾角约 98°）；三者比例每个场景随机；
- 站点里固定有 1～3 个高纬度站（v2_sampling.STATION_ANCHORS 新增的 24～33 号，46°～62°），
  并去掉离赛题站点不到 1° 的 10 号锚点；
- 中轨比例约 5%，目标数偏多、站数偏少，整体更挤。

每个场景的数量参数仍是分层截断正态抽样（和 v0.2 一样），种子与 v0.2 不重叠。
"""
from __future__ import annotations

import calendar
from datetime import datetime, timezone

import numpy as np

from .v2_sampling import (HIGH_LAT_ANCHORS, STATION_ANCHORS, V2_ANCHOR_COUNT, _ppf, _integer,
                          _orbit_parameters, size_bin)

SAMPLING_VERSION = "synthetic-v0.3-contestlike-1"
CONFIG_SEED = 2027111100
CATALOG_SEED_BASE = 202711110000
COUNTS = {"train": 120, "validation": 24, "test": 36}
# v0.2 的 10 号锚点（-110.0, 52.0）离赛题的一个站不到 1°，v0.3 不用它，避免把考题的站址教给网络。
EXCLUDED_ANCHORS = (10,)
# 均值、标准差、下界、上界（截断正态）。
CONTROLS = {
    "n_targets": (9500., 2500., 5000., 16000.),
    "n_stations": (9., 2., 5., 14.),
    "leo_fraction": (.83, .05, .70, .95),
    "meo_remaining_fraction": (.30, .12, .08, .60),
    "high_priority_fraction": (.22, .07, .08, .45),
    "medium_remaining_fraction": (.45, .10, .20, .70),
    "leo_altitude_mean_km": (800., 150., 600., 1100.),
    "leo_inclination_mean_deg": (65., 15., 30., 100.),
    "shell_fraction": (.30, .12, .05, .60),
    "shell_altitude_km": (530., 40., 460., 620.),
    "shell_inclination_deg": (53., 8., 40., 75.),
    "sunsync_fraction": (.25, .10, .05, .50),
}


def _stations(rng: np.random.Generator, s: int) -> np.ndarray:
    high = HIGH_LAT_ANCHORS
    high_north = high[STATION_ANCHORS[high, 1] > 0]
    k = int(min(rng.integers(1, 4), s - 1))
    chosen = [int(rng.choice(high_north))]
    chosen += [int(v) for v in rng.choice(np.setdiff1d(high, chosen), k - 1, replace=False)]
    base = np.setdiff1d(np.arange(V2_ANCHOR_COUNT), EXCLUDED_ANCHORS)
    if not (STATION_ANCHORS[chosen, 1] < 0).any():
        chosen.append(int(rng.choice(base[STATION_ANCHORS[base, 1] < 0])))
    rest = np.setdiff1d(base, chosen)
    chosen += [int(v) for v in rng.choice(rest, s - len(chosen), replace=False)]
    return np.sort(np.array(chosen, dtype=np.int16))


def _make_spec(index: int, split: str, control: dict, quantiles: dict, month: int) -> dict:
    seed = CATALOG_SEED_BASE + index
    rng = np.random.default_rng(np.random.SeedSequence([seed, 11]))
    n, s = _integer(control["n_targets"]), _integer(control["n_stations"])
    anchors = _stations(rng, s)
    day = int(rng.integers(1, calendar.monthrange(2027, month)[1] - 5))
    start = datetime(2027, month, day, tzinfo=timezone.utc)
    leo, meo_rest = control["leo_fraction"], control["meo_remaining_fraction"]
    high, medium_rest = control["high_priority_fraction"], control["medium_remaining_fraction"]
    orbit = _orbit_parameters(control)
    orbit["LEO"]["altitude_km"] = [control["leo_altitude_mean_km"], 250., 400., 1800.]
    orbit["LEO"]["eccentricity"] = [.003, .003, .00005, .012]
    sh_alt, sh_inc = control["shell_altitude_km"], control["shell_inclination_deg"]
    return dict(
        catalog_id=f"v3-cat-{index:03d}", split=split, seed=seed,
        n_targets=n, n_stations=s, size_bin=size_bin(n),
        start_utc=start.isoformat(), sampling_version=SAMPLING_VERSION,
        distribution="contest_like", sampling_group="v0.3", stress_type=None,
        sampling_latents={k: float(v) for k, v in control.items()},
        sampling_quantiles={k: float(v) for k, v in quantiles.items()},
        family_probabilities=[leo, (1 - leo) * meo_rest, (1 - leo) * (1 - meo_rest)],
        priority_probabilities=[(1 - high) * (1 - medium_rest), (1 - high) * medium_rest, high],
        orbit_parameters=orbit,
        LEO_mixture={"shell_fraction": control["shell_fraction"],
                     "shell_altitude_km": [sh_alt, 12., sh_alt - 40., sh_alt + 40.],
                     "shell_eccentricity": [.0002, .0003, .00001, .002],
                     "shell_inclination_deg": [sh_inc, 1., sh_inc - 4., sh_inc + 4.],
                     "sunsync_fraction": control["sunsync_fraction"],
                     "sunsync_inclination_deg": [98., .8, 95.5, 101.]},
        min_perigee_km=300.,
        station_anchor_ids=[int(v) for v in anchors], network_mode="global_high_latitude",
        station_jitter={"longitude_deg": [0., .02, -.06, .06],
                        "latitude_deg": [0., .02, -.06, .06],
                        "height_m": [0., 20., -60., 60.]},
    )


def specs() -> list[dict]:
    """120 / 24 / 36 个 v0.3 场景（训练 / 验证 / 测试），确定性生成。"""
    rows = []
    for split_index, (split, n) in enumerate(COUNTS.items()):
        rng = np.random.default_rng(np.random.SeedSequence([CONFIG_SEED, split_index]))
        quantiles, controls = {}, {}
        for key, params in CONTROLS.items():
            u = (np.arange(n, dtype=float) + rng.random(n)) / n
            rng.shuffle(u)
            quantiles[key] = u
            controls[key] = _ppf(u, params)
        months = np.tile(np.arange(1, 13), n // 12)
        rng.shuffle(months)
        for j in range(n):
            rows.append(_make_spec(len(rows), split, {k: v[j] for k, v in controls.items()},
                                   {k: v[j] for k, v in quantiles.items()}, int(months[j])))
    return rows
