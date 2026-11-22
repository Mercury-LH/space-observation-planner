"""加分项实验：结合真实历史预报的动态重规划（全量 10000 目标 × 10 站 × 6 天）。

检验年 2024、2025（这两年有 Open-Meteo 存档的提前 1 天 / 2 天预报）：
- 实况：ERA5 再分析逐小时云量，云量 ≤ 30% 才算观测成功；
- 气候统计：各站 2016–2022 年同期晴空比例；
- 预报校准：检验 2024 年时用 2025 年的预报与实况校准，反之亦然；
- 完美预报（上限对照）：每天重排时当天的真实云量已知，把当天阴天的时段从窗口中裁掉，衡量"预报"这条路最多能带来多少；
- 事后最优（参照）：6 天的真实天气全部已知（窗口裁剪到晴空时段）时一次规划的结果，现实中不可实现。
输出 outputs/materials/dynamic.json 与 results/dynamic.json。

    uv run python scripts/dynamic_experiment.py                         # 全部策略
    uv run python scripts/dynamic_experiment.py --policies dynamic_oracle  # 只补跑某些策略，合并进已有结果
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.dynamic import calibrate, simulate  # noqa: E402
from obsplan.environment import fetch_cloud_history, fetch_forecast_history  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

THRESHOLD = 30.0
POLICIES = {"static": "静态标准计划", "static_clim": "静态 + 气候统计", "dynamic_clim": "每天重排 + 气候统计",
            "dynamic_forecast": "每天重排 + 预报", "dynamic_oracle": "每天重排 + 完美预报（上限）",
            "hindsight": "事后最优（天气全知）"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="*", choices=list(POLICIES), default=list(POLICIES))
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    prev_path = root / "results" / "dynamic.json"
    prev = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.exists() else {"years": {}}
    cfg = Config()
    ds = load(cfg)
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    cloud = fetch_cloud_history(cfg, ds, verbose=False)          # 2016–2025
    fc = fetch_forecast_history(cfg, ds, verbose=False)           # 2023–2025（2023 无数据）
    clim = np.nanmean(cloud[:7] <= THRESHOLD, axis=(0, 2))        # 2016–2022
    res = {"threshold_pct": THRESHOLD, "clim_years": "2016-2022", "years": {}}
    perfect = lambda x: (x <= THRESHOLD).astype(float)  # noqa: E731
    for year, other in ((2024, 2025), (2025, 2024)):
        yi, oi = fc["years"].index(year), fc["years"].index(other)
        truth = cloud[year - 2016]
        cal1 = calibrate(fc["day1"][oi], cloud[other - 2016], THRESHOLD)
        cal2 = calibrate(fc["day2"][oi], cloud[other - 2016], THRESHOLD)
        old = prev["years"].get(str(year), {}).get("policies", {})
        res["years"][year] = {"calibration_day1": np.round(cal1.table, 3).tolist(),
                              "policies": {k: old[k] for k in POLICIES if k in old and k not in args.policies}}
        for key in args.policies:
            name = POLICIES[key]
            if key == "dynamic_oracle":
                r = simulate(pb, truth, "dynamic_forecast", clim, threshold=THRESHOLD, fc1=truth, fc2=truth,
                             cal1=perfect, cal2=perfect, iters=150, fast=True, clip_p=0.5)
            else:
                r = simulate(pb, truth, key, clim, threshold=THRESHOLD, fc1=fc["day1"][yi], fc2=fc["day2"][yi],
                             cal1=cal1, cal2=cal2, iters=150, fast=True)
            r = {k: float(v) for k, v in r.items()}
            res["years"][year]["policies"][key] = {"name": name, **r}
            print(f"{year} {name:12s} 成功覆盖 {r['covered']:.0f} 得分 {r['score']:.0f} 综合 {r['combined']:.4f} "
                  f"执行 {r['executed']:.0f}（失败 {r['failed']:.0f}）用时 {r['seconds']:.0f}s", flush=True)
        pol = res["years"][year]["policies"]
        res["years"][year]["policies"] = {k: pol[k] for k in POLICIES if k in pol}
    for out in (root / "outputs" / "materials" / "dynamic.json", root / "results" / "dynamic.json"):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
