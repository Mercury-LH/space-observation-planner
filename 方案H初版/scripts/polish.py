"""深度打磨（随时可停）：在一份已经留好边界余量的计划上，轮换几组弹射链设置反复跑，只接受变好的。

    uv run python scripts/polish.py --margin 5 --plan outputs/final_best/best_7d_edge5.csv --minutes 30 --profile wide
    uv run python scripts/polish.py --margin 5 --plan 某计划.csv --minutes 30 --profile deep --threads 3 --out outputs/polish/deep

弹射链（obsplan.chain.improve）做到头会自己停：一轮涨不到 escalate_gain、深度 2 的换位置也不涨就返回。
这里停了就换下一组设置（随机种子、最低优先级、链深度、时间线切法）接着做，直到总时长用完或连续一整圈都不涨。
设置组（--profile）：
  default：默认设置，只换随机种子（候选目标的尝试顺序不同，会落到不同的局部最优）；
  wide：中、低优先级目标也用弹射链加重复观测（默认只给高优先级找链，其余只往空当里直接塞）；
  deep：更深的链、更多重试、更大的插入范围、四种时间线切法轮换。
每次变好都把当前最好存成 <out>/best.csv（可以随时停，停了用它接着跑）。最后与 scripts/margin_repair.py 相同：
裁进规范的时间范围，按不留余量的真实边界核对每个任务离边界至少 B 秒，Skyfield 逐秒校验，
写 <out>/plan_edge{B}.csv、plan_edge{B}.json（提交格式）和 polish_edge{B}.json（每一步的记录）。
退出码：合法且余量全满足为 0，否则 1。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from margin_repair import raw_bounds, shrink  # noqa: E402

from obsplan.__main__ import read_plan  # noqa: E402
from obsplan.chain import improve  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.schedule import evaluate, export_csv  # noqa: E402
from obsplan.submission import fit_applies, fit_to_range, write_json  # noqa: E402
from obsplan.validator import validate  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

PROFILES = {
    "default": [{}],
    "wide": [{"min_priority": 3.0}, {"min_priority": 2.0}, {}],
    "deep": [{"depth": 6, "retries": 10, "shift": 6, "repattern_depth": 4},
             {"depth": 8, "rounds": "0:0,40:0,40:20,0:0", "retries": 8},
             {"depth": 6, "min_priority": 3.0, "shift": 6}],
}
KEYS = ("covered", "score", "combined", "tasks")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--margin", type=int, required=True, help="边界余量 B（秒，0～10；输入计划须已满足）")
    ap.add_argument("--plan", required=True, help="起点计划（CSV 或提交格式 JSON）")
    ap.add_argument("--minutes", type=float, default=30.0, help="总时长上限（分钟）")
    ap.add_argument("--step", type=float, default=8.0, help="每组设置最多跑几分钟（做到头会提前返回）")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="default")
    ap.add_argument("--seed", type=int, default=1, help="第一组设置的随机种子，之后每步加 1")
    ap.add_argument("--threads", type=int, default=None, help="弹射链线程数（默认按核数，最多 8）")
    ap.add_argument("--out", help="输出目录，默认 方案H初版/outputs/polish/<profile>/")
    ap.add_argument("--no-validate", action="store_true", help="最后不做 Skyfield 校验（只用于试验）")
    args = ap.parse_args()
    B = int(args.margin)
    if not 0 <= B <= 10:
        ap.error("--margin 要在 0～10 秒之间")
    out = Path(args.out).resolve() if args.out else ROOT / "outputs" / "polish" / args.profile
    out.mkdir(parents=True, exist_ok=True)
    plan = Path(args.plan).resolve()

    cfg0 = Config()
    cfgB = cfg0.with_(edge_margin_s=B)
    ds = load(cfg0)
    pb0 = Problem.build(cfg0, ds, get_windows(cfg0, ds))
    pbB = pb0 if B == 0 else Problem.build(cfgB, ds, get_windows(cfgB, ds))
    best = read_plan(cfgB, ds, plan)[0]
    e0 = evaluate(cfgB, ds, best)
    best_ev = e0
    print(f"起点：{plan.name}，{e0['tasks']} 个任务、得分 {e0['score']:.1f}、覆盖 {e0['covered']} 颗；"
          f"设置组 {args.profile}，总时长 {args.minutes:g} 分钟", flush=True)

    t_start = time.time()
    hard = t_start + args.minutes * 60
    steps, last_save = [], [0.0]

    def save(sch, ev, used):
        nonlocal best, best_ev
        if ev["combined"] > best_ev["combined"]:
            best, best_ev = sch, ev
        if time.time() - last_save[0] > 60:                # 最多每分钟存一次
            export_csv(cfgB, ds, best, out / "best.csv")
            last_save[0] = time.time()

    k, dry = 0, 0
    prof = PROFILES[args.profile]
    while time.time() < hard - 30 and dry < len(prof) + 1:
        kw = dict(prof[k % len(prof)])
        kw.setdefault("seed", args.seed + k)
        left = (hard - time.time()) / 60
        m = min(args.step, left)
        before = best_ev["score"]
        t0 = time.time()
        sch, ci = improve(pbB, best, minutes=m, threads=args.threads, on_best=save, **kw)
        ev = evaluate(cfgB, ds, sch)
        if ev["combined"] > best_ev["combined"]:
            best, best_ev = sch, ev
        gain = best_ev["score"] - before
        dry = 0 if gain > 1e-6 else dry + 1
        steps.append({"step": k, "settings": kw, "minutes": round(m, 2), "seconds": round(time.time() - t0, 1),
                      "score_before": round(before, 3), "score_after": round(best_ev["score"], 3),
                      "covered": best_ev["covered"], "tasks": best_ev["tasks"]})
        print(f"第 {k + 1} 步 {kw}：{time.time() - t0:.0f} 秒，得分 {before:.1f} → {best_ev['score']:.1f}"
              f"（{gain:+.1f}），覆盖 {best_ev['covered']} 颗，已用 {(time.time() - t_start) / 60:.1f} 分钟", flush=True)
        export_csv(cfgB, ds, best, out / "best.csv")
        last_save[0] = time.time()
        k += 1

    rep, spec_fit = best, None
    if fit_applies(cfgB):                                    # 与 `obsplan plan` 相同：裁进规范的时间范围
        rep, spec_fit = fit_to_range(cfgB, rep)
    e_rep = evaluate(cfgB, ds, rep)
    end = cfg0.end_s
    L2, R2 = raw_bounds(pb0, rep)                            # 按不留余量的真实边界核对余量
    LB2, RB2 = shrink(L2, R2, B, end)
    rs = np.rint(rep.start_s).astype(np.int64)
    re = rs + np.rint(rep.dur_s).astype(np.int64)
    short_of_margin = int(((L2 < 0) | (rs < LB2) | (re > RB2)).sum())
    chk = None if args.no_validate else validate(cfgB, ds, rep)
    legal = None if chk is None else bool(chk["legal"])
    print(f"打磨 {(time.time() - t_start) / 60:.1f} 分钟：得分 {e0['score']:.1f} → {e_rep['score']:.1f}、覆盖 {e_rep['covered']} 颗、"
          f"任务 {e_rep['tasks']} 个；余量不足 {short_of_margin} 个；Skyfield 校验 "
          f"{'未做' if chk is None else ('合法' if legal else chk['violations'])}", flush=True)
    export_csv(cfgB, ds, rep, out / f"plan_edge{B}.csv", chk)
    notes = {k_: v for k_, v in write_json(cfgB, ds, rep, out / f"plan_edge{B}.json").items() if k_ != "path"}
    rel = lambda p: str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name     # noqa: E731  不写本机绝对路径
    info = {"margin_s": B, "plan": rel(plan), "profile": args.profile, "minutes": args.minutes,
            "input": {k_: e0[k_] for k_ in KEYS}, "polished": {k_: e_rep[k_] for k_ in KEYS}, "steps": steps,
            "short_of_margin": short_of_margin, "legal": legal,
            "violations": None if chk is None else chk["violations"], "spec_fit": spec_fit, "submission_notes": notes}
    (out / f"polish_edge{B}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=float),
                                              encoding="utf-8")
    return 0 if legal is not False and short_of_margin == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
