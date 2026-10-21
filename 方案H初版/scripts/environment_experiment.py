"""加分项实验：标准结果 vs 考虑天气 / 月光后的结果。

- 天气：用 2016–2020 年同期逐小时云量估计各站的平均晴空概率并据此做期望价值规划，
  在 2021–2025 年的真实云量下逐年回放评估（样本外），任务所覆盖的每个小时云量 ≤ 阈值才算成功。
  试过的其他做法（样本外都不如站点平均）：按"站点×每个小时"估计（每格只有 5 个样本，过拟合）、
  按"站点×一天中的小时"估计、给重复观测加"备份覆盖"价值（挤占晴空站点时间，覆盖反而下降）。
- 月光：月亮在地平线以上时目标与月亮角距 ≥ 30°。
输出 outputs/materials/environment.json。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsplan import lagrange  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.environment import evaluate_under_weather, fetch_cloud_history  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.schedule import evaluate  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

THRESHOLD = 30.0
OUT = Path(__file__).resolve().parents[1] / "outputs" / "materials"


def backup_factor(pb: Problem, q: np.ndarray) -> np.ndarray:
    """每颗目标"首次观测因天气失败"的大致概率 = 1 − 其全部窗口的平均晴空概率。"""
    qp = np.concatenate([np.zeros((q.shape[0], 1)), np.cumsum(q, axis=1)], axis=1)
    wq = (qp[pb.w_st, pb.w_k1 + 1] - qp[pb.w_st, pb.w_k0]) / (pb.w_k1 - pb.w_k0 + 1)
    L = (pb.w_k1 - pb.w_k0 + 1).astype(float)
    num = np.bincount(pb.w_tg, weights=wq * L, minlength=pb.n_targets)
    den = np.bincount(pb.w_tg, weights=L, minlength=pb.n_targets)
    return np.where(den > 0, 1 - num / np.maximum(den, 1e-9), 0.0)


def main():
    cfg = Config()
    ds = load(cfg)
    cloud = fetch_cloud_history(cfg, ds)                       # 2016–2025
    train, test = cloud[:5], cloud[5:]
    station_clear = np.nanmean(train <= THRESHOLD, axis=(0, 2))
    res = {"threshold_pct": THRESHOLD, "train_years": "2016-2020", "test_years": "2021-2025",
           "station_clear_train": np.nanmean(train <= THRESHOLD, axis=(0, 2)).round(3).tolist(),
           "station_cloud_mean": np.nanmean(cloud, axis=(0, 2)).round(1).tolist(), "plans": []}

    def record(name, c, sch, runtime):
        nominal = evaluate(c, ds, sch)
        ev = evaluate_under_weather(c, ds, sch, test, THRESHOLD)
        row = {"name": name, "covered": nominal["covered"], "score": nominal["score"], "tasks": nominal["tasks"],
               "exp_covered": ev["coverage_mean"], "exp_score": ev["score_mean"],
               "covered_by_year": ev["coverage_by_year"], "score_by_year": ev["score_by_year"], "runtime_s": runtime}
        res["plans"].append(row)
        print(f"{name:20s} 名义 覆盖 {row['covered']} 得分 {row['score']:.0f} | 2021–2025 真实天气下 期望覆盖 "
              f"{row['exp_covered']:.0f} 期望得分 {row['exp_score']:.0f}  ({runtime:.0f}s)", flush=True)

    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    q = np.repeat(station_clear[:, None], pb.n_samples + 2, axis=1)     # 站点平均晴空概率
    no_backup = np.zeros(pb.n_targets)
    t = time.time()
    sch, _ = lagrange.solve(pb, iters=300, verbose=False)
    record("标准计划", cfg, sch, time.time() - t)

    t = time.time()
    sch_w, _ = lagrange.solve(pb, iters=300, verbose=False, cell_weight=q, backup=no_backup)
    record("考虑天气", cfg, sch_w, time.time() - t)

    cm = cfg.with_(moon_sep_deg=30.0)
    pbm = Problem.build(cm, ds, get_windows(cm, ds))
    t = time.time()
    sch_m, _ = lagrange.solve(pbm, iters=300, verbose=False)
    record("考虑月光", cm, sch_m, time.time() - t)
    t = time.time()
    sch_mw, _ = lagrange.solve(pbm, iters=300, verbose=False, cell_weight=q, backup=no_backup)
    record("考虑天气与月光", cm, sch_mw, time.time() - t)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "environment.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "environment_plans.npz", **{
        f"{k}_{f}": getattr(s, f) for k, s in (("std", sch), ("wx", sch_w), ("moon", sch_m), ("moonwx", sch_mw))
        for f in ("station", "target", "start_s", "dur_s")})


if __name__ == "__main__":
    main()
