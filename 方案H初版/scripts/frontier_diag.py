"""诊断：还没观测满 3 次的高优先级目标，卡在哪。

    uv run python scripts/frontier_diag.py fast fast_c1 fast_c10 outputs/final_best/longrun_chain_best.csv

对每颗"能排 3 次、现在不到 3 次"的高优先级目标，看它自己的窗口里还有没有"与已有观测隔 48 小时"的时间段：
- 节奏卡死：一个都没有（已有观测的位置让下一次无处可放，只能先挪动已有观测）；
- 有窗口没空位：有这样的时间段，但那里被别的任务占着（要靠弹射链顶出来）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.frontier_lab import load_plan, problem  # noqa: E402
from obsplan.chain import Timeline, _slots  # noqa: E402
from obsplan.second_bounds import all_bounds  # noqa: E402


def diag(pb, wL, wR, sch) -> dict:
    tl = Timeline(pb, wL, wR, sch)
    sa, sb = np.empty(4, np.int64), np.empty(4, np.int64)
    out = {}
    for lab, P in pb.cfg.priority_score.items():
        r = {"cnt1_stuck": 0, "cnt1_open": 0, "cnt2_stuck": 0, "cnt2_open": 0, "cnt0": 0}
        for x in np.nonzero((pb.P == P) & (pb.max_obs >= 3) & (tl.tg_cnt < 3))[0]:
            c = int(tl.tg_cnt[x])
            if c == 0:
                r["cnt0"] += 1
                continue
            n = 0
            for ci in range(tl.c_ptr[x], tl.c_ptr[x + 1]):
                n += _slots(x, -1, tl.c_l[ci], tl.c_r[ci], tl.tk_s, tl.tk_e, tl.tg_tasks, tl.gap, tl.dmin, sa, sb)
                if n:
                    break
            r[f"cnt{c}_{'open' if n else 'stuck'}"] += 1
        out[lab] = r
    return out


def main() -> int:
    cfg, ds, pb = problem()
    wL, wR = all_bounds(pb, Path(cfg.cache_dir) / f"bounds_all_{cfg.scenario_key()}.npz")
    res = {}
    for name in sys.argv[1:]:
        res[name] = diag(pb, wL, wR, load_plan(cfg, ds, name))
        print(name, json.dumps(res[name], ensure_ascii=False))
    out = ROOT / "results" / "frontier" / "diag.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
