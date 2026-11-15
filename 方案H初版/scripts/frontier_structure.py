"""几份计划的结构对比：时间都花在哪一档优先级上、每档观测了几次、各站黑夜用了多少。

    uv run python scripts/frontier_structure.py 快速档=fast 快速+弹射链1分钟=fast_c1 最好计划=outputs/final_best/longrun_chain_best.csv

每个参数是"名字=计划"，计划可以是 outputs/frontier/lab/ 下的名字或一份 CSV。结果写到 results/frontier/structure.json，
并打印一张 Markdown 表（README"时间-质量前沿"一节用）。

怎么读：一次观测占用"时长 + 30 秒切换"。同样一分钟，用在高优先级重复观测上值 1.2 分、高优先级延长 1.0 分、
中优先级重复 0.72、中优先级延长 0.6、低优先级重复 0.48、低优先级延长 0.4。黑夜基本用满之后，计划好不好就看
时间在这几档之间怎么分。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.frontier_lab import load_plan, problem  # noqa: E402
from obsplan.schedule import counted_mask, evaluate  # noqa: E402


def structure(cfg, ds, pb, sch) -> dict:
    cnt = counted_mask(cfg, sch)
    st, tg, d = sch.station[cnt], sch.target[cnt], np.asarray(sch.dur_s, float)[cnt]
    e = evaluate(cfg, ds, sch)
    dark_s = np.array([sum((b - a) * cfg.cell_s for a, b in runs) for runs in pb.win.dark], float)
    busy = np.bincount(st, weights=d + cfg.switch_s, minlength=ds.n_stations)
    out = {"covered": int(e["covered"]), "score": round(float(e["score"]), 1), "combined": round(float(e["combined"]), 6),
           "tasks": int(len(d)), "dark_h": round(float(dark_s.sum() / 3600), 1),
           "busy_h": round(float(busy.sum() / 3600), 1), "idle_h": round(float((dark_s - busy).sum() / 3600), 1),
           "idle_h_by_station": ((dark_s - busy) / 3600).round(1).tolist(), "by_priority": {}}
    n_obs = np.bincount(tg, minlength=ds.n_targets)
    observable = pb.max_obs > 0
    for lab, P in cfg.priority_score.items():
        m_t = (ds.priority == P)
        m = m_t[tg]
        dm = d[m]
        base = len(dm) * (cfg.obs_min_s + cfg.switch_s)          # 每次观测的最低占用：120 秒 + 30 秒切换
        ext = float((dm - cfg.obs_min_s).sum())
        no = n_obs[m_t & observable]
        out["by_priority"][lab] = {
            "targets_observable": int((m_t & observable).sum()),
            "obs_0": int((no == 0).sum()), "obs_1": int((no == 1).sum()), "obs_2": int((no == 2).sum()),
            "obs_3": int((no >= 3).sum()),
            "can_3": int((pb.max_obs[m_t] >= 3).sum()), "can_2": int((pb.max_obs[m_t] == 2).sum()),
            "tasks": int(len(dm)), "repeats": int(len(dm) - (no > 0).sum()),
            "hours_total": round(float((dm + cfg.switch_s).sum() / 3600), 1),
            "hours_first": round(float((no > 0).sum() * (cfg.obs_min_s + cfg.switch_s) / 3600), 1),
            "hours_repeat": round(float((len(dm) - (no > 0).sum()) * (cfg.obs_min_s + cfg.switch_s) / 3600), 1),
            "hours_extension": round(ext / 3600, 1), "mean_dur_s": round(float(dm.mean()), 1) if len(dm) else 0.0,
            "at_240": int((dm >= cfg.obs_max_s - 0.5).sum()), "at_120": int((dm <= cfg.obs_min_s + 0.5).sum()),
            "points": round(float((P * (0.6 + 0.4 * (dm - cfg.obs_min_s) / (cfg.obs_max_s - cfg.obs_min_s))).sum()), 1),
            "base_hours": round(base / 3600, 1)}
    return out


def main() -> int:
    cfg, ds, pb = problem()
    res = {}
    for arg in sys.argv[1:]:
        name, plan = arg.split("=", 1)
        res[name] = structure(cfg, ds, pb, load_plan(cfg, ds, plan))
        res[name]["plan"] = plan
    out = ROOT / "results" / "frontier" / "structure.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    names = list(res)
    rows = [("综合", lambda r: f"{r['combined']:.4f}"), ("覆盖 / 得分", lambda r: f"{r['covered']} / {r['score']:.0f}"),
            ("黑夜空闲（小时，共 %.0f）" % res[names[0]]["dark_h"], lambda r: f"{r['idle_h']:.1f}")]
    for lab in cfg.priority_score:
        g = lambda r, k, lab=lab: r["by_priority"][lab][k]            # noqa: E731
        rows += [(f"{lab}：观测 0 / 1 / 2 / 3 次的目标数",
                  lambda r, g=g: f"{g(r, 'obs_0')} / {g(r, 'obs_1')} / {g(r, 'obs_2')} / {g(r, 'obs_3')}"),
                 (f"{lab}：重复观测占用（小时）", lambda r, g=g: f"{g(r, 'hours_repeat')}"),
                 (f"{lab}：延长占用（小时）", lambda r, g=g: f"{g(r, 'hours_extension')}"),
                 (f"{lab}：得分", lambda r, g=g: f"{g(r, 'points'):.0f}")]
    print("| 项目 | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for title, f in rows:
        print(f"| {title} | " + " | ".join(f(res[n]) for n in names) + " |")
    p0 = res[names[0]]["by_priority"]
    print("\n可观测目标数 / 最多能排 3 次的目标数：" + "；".join(
        f"{lab} {p0[lab]['targets_observable']} / {p0[lab]['can_3']}" for lab in cfg.priority_score))
    print("已写出", out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
