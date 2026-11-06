"""加分项实验：突发事件下的动态重规划（全量 10000 目标 × 10 站 × 6 天）。

在默认流水线（H+ 高质量 + 按秒精修）的计划上，于 08-03 12:00 UTC 分别发生三类事件，
用"扰动最小"和"全面重排"两种方式重排事件之后的时段，比较覆盖、得分与对原计划的改动：
- 站点停机：事件之后 6 小时内任务最多的站点停机 6 小时；
- 目标升为高优先级：推荐列表第 1 个（原计划只观测 1 次、48 小时后还有窗口的中低优先级目标）；
- 新增必须观测的目标：推荐列表第 1 个（原计划没覆盖、事件之后还有窗口的目标）。
输出 results/replan.json。

    uv run python scripts/replan_experiment.py                  # 先规划 H+（约 2.5 分钟），再做 6 次重排（各约 20 秒）
    uv run python scripts/replan_experiment.py --plan plan.csv  # 在已有计划上做
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsplan.app_backend import load_problem, run_planning  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.replan import KINDS, MODES, Event, busiest_station, replan, suggest_targets  # noqa: E402
from obsplan.schedule import evaluate  # noqa: E402

T_EVENT = 60 * 3600          # 08-03 12:00 UTC（规划开始后 60 小时）

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, help="已有计划 CSV（默认重新规划 H+）")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = Config()
    ds, pb, _ = load_problem(cfg)
    if args.plan:
        from obsplan.__main__ import read_plan
        sch, skipped = read_plan(cfg, ds, args.plan)
        assert not skipped, f"计划里有 {len(skipped)} 行读不出来"
        base = {"plan": str(args.plan)}
    else:
        t0 = time.time()
        run = run_planning(pb, "H", iters=300)
        sch = run.schedule()
        base = {"plan": "H+ 高质量 + 按秒精修", "seconds": round(time.time() - t0, 1)}
    base["metrics"] = {k: evaluate(cfg, ds, sch)[k] for k in ("covered", "score", "combined", "tasks")}
    print("原计划", base, flush=True)
    events = [Event("down", T_EVENT, station=busiest_station(sch, pb.n_stations, T_EVENT, 6.0), hours=6.0),
              Event("priority", T_EVENT, target=suggest_targets(pb, sch, T_EVENT, "priority")[0]),
              Event("must", T_EVENT, target=suggest_targets(pb, sch, T_EVENT, "must")[0])]
    when = (cfg.start_dt + timedelta(seconds=T_EVENT)).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = {"event_time_utc": when, "base": base, "events": []}
    for ev in events:
        for mode in MODES:
            res = replan(pb, sch, ev, mode=mode)
            rec = {"kind": KINDS[ev.kind], "event": res["describe"], "mode": MODES[mode], "seconds": res["seconds"],
                   "timing": res["timing"], "diff": res["diff"], "structural_violations": len(res["violations"]),
                   "frozen_unchanged": res["frozen_unchanged"], "outage_tasks": res["outage_tasks"],
                   **{k: {m_: res[k][m_] for m_ in ("covered", "score", "combined")} for k in ("before", "no_replan", "after")}}
            if "target" in res:
                rec["target"] = {k: v for k, v in res["target"].items() if k in ("before", "after", "ok", "released_fixed")}
            out["events"].append(rec)
            d = res["diff"]
            print(f"{res['describe']} · {MODES[mode]}：{res['seconds']:.0f} 秒；覆盖 {res['no_replan']['covered']} → "
                  f"{res['after']['covered']}，综合 {res['no_replan']['combined']:.4f} → {res['after']['combined']:.4f}；"
                  f"保留 {d['保留']}/{d['原计划 t 之后']}（完全不变 {d['完全不变']}），移动 {d['移动']}，删除 {d['删除']}，"
                  f"新增 {d['新增']}；结构违规 {len(res['violations'])} 类", flush=True)
    path = root / "results" / "replan.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("已写入", path)
