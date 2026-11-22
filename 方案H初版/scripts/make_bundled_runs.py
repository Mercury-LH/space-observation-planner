"""生成随仓库提供的两份全量结果（results/app_runs/），全新克隆后第一次打开网页就能看到：

    uv run python scripts/make_bundled_runs.py                 # 两份都生成
    uv run python scripts/make_bundled_runs.py --only best     # 只生成最好计划那一份

1. 推荐档：和不带任何参数的 `python -m obsplan plan` 同一套设置（快速价格协调 + 按秒精修 + 弹射链），现场跑一遍；
2. 最好计划：outputs/final_best/longrun_chain_best.csv（长跑优化 + 弹射链，综合 0.9958），只读入、不重算。

两份都做 Skyfield 逐秒校验，校验结果和每个任务复核出的仰角、太阳角一起存进结果文件。弹射链按时间截停，
推荐档每次生成的分数会有微小差别；文档里的用时和分数以 scripts/frontier_bench.py 的三次中位数为准。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from obsplan.__main__ import read_plan  # noqa: E402
from obsplan.app_backend import Run, run_planning, validate_run  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.schedule import evaluate  # noqa: E402
from obsplan.tiers import RECOMMENDED_CHAIN_MIN  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

OUT = ROOT / "results" / "app_runs"
BEST = ROOT / "outputs" / "final_best" / "longrun_chain_best.csv"
BEST_HOURS = 7.0            # 长跑约 6 小时（64 核服务器）+ 弹射链约 1 小时，见 README


def report(run: Run) -> None:
    m, v = run.metrics, run.validation
    print(f"{run.notes['label']}：{m['covered']} 颗 / {m['score']:.1f} 分 / 综合 {m['combined']:.6f}，"
          f"Skyfield {'全部合法' if v['legal'] else '有违规'}（{v['seconds']} 秒）→ {(OUT / (run.run_id + '.npz')).relative_to(ROOT)}",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["recommended", "best"])
    a = ap.parse_args()
    cfg = Config()
    ds = load(cfg)
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    if a.only != "best":
        run = run_planning(pb, "H", iters=150, fast=True, chain_minutes=RECOMMENDED_CHAIN_MIN)
        run.run_id = "bundled-1-recommended"
        run.notes["label"] = "推荐档（默认命令，约 2 分钟）"
        validate_run(pb, run).save(OUT)
        report(run)
    if a.only != "recommended":
        sch, skipped = read_plan(cfg, ds, BEST)[:2]
        assert not skipped, skipped
        import numpy as np
        near_only = int((pb.max_obs[np.unique(sch.target)] == 0).sum())    # 网格上没有窗口、按秒才看得到的目标
        run = Run("bundled-2-best", "H", json.loads(cfg.to_json()), {}, sch.station, sch.target, sch.start_s, sch.dur_s,
                  evaluate(cfg, ds, sch), BEST_HOURS * 3600,
                  notes={"label": "最好计划（长跑 + 弹射链 + 深度换位置）", "mode": "长跑", "famou_rule": True,
                         "refine": {"refined": True}, "insert": {"added": near_only}, "source": str(BEST.relative_to(ROOT))})
        validate_run(pb, run).save(OUT)
        report(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
