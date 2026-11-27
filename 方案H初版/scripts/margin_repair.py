"""给一份已有的计划加按秒边界余量：贴边的任务往里收 B 秒，收完不够 120 秒的删掉，再用弹射链修补。

    uv run python scripts/margin_repair.py --margin 2                       # 最好计划，修补 10 分钟
    uv run python scripts/margin_repair.py --margin 1 --plan 某计划.csv --minutes 5 --out outputs/edge1

主办方规范 v3.0 按"观测与预报窗口的交集"算有效时长：主办方的窗口要是比我们算的短几秒，贴边的任务就会出错
（120 秒的任务少 1 秒就不足 120 秒）。--margin B 让每个任务离真实可见边界至少 B 秒（规划期的起止不算边界），
主办方的窗口每端短 B 秒以内时计划不受影响。从头规划时用 `obsplan plan --edge-margin B` 即可，这个脚本用来
修补已经长跑优化过的计划，不用从头再跑。

输出（--out，默认 方案H初版/outputs/edge_margin/；自己给的 --plan、--out 相对路径按当前目录算）：plan_edge{B}.csv（含 Skyfield 复核）、plan_edge{B}.json（提交文件）、
repair_edge{B}.json（裁边、修补前后的指标）。弹射链按时间截停，电脑快慢不同，结果会有细微差别。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsplan.__main__ import read_plan  # noqa: E402
from obsplan.chain import improve  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.schedule import Schedule, evaluate, export_csv  # noqa: E402
from obsplan.second_bounds import all_bounds  # noqa: E402
from obsplan.submission import fit_applies, fit_to_range, write_json  # noqa: E402
from obsplan.validator import validate  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def raw_bounds(pb: Problem, sch: Schedule) -> tuple[np.ndarray, np.ndarray]:
    """每个任务所在窗口不留余量的按秒可见区间 [L, R]；对不上任何窗口的任务给 -1（与弹射链时间线同样的匹配）。"""
    cfg = pb.cfg
    wL, wR = all_bounds(pb, Path(cfg.cache_dir) / f"bounds_all_{cfg.scenario_key()}.npz")
    order = np.argsort(pb.w_tg, kind="stable")
    t_ptr = np.zeros(pb.n_targets + 1, np.int64)
    np.cumsum(np.bincount(pb.w_tg, minlength=pb.n_targets), out=t_ptr[1:])
    s0 = np.rint(sch.start_s).astype(np.int64)
    e0 = s0 + np.rint(sch.dur_s).astype(np.int64)
    L, R = np.full(len(sch), -1, np.int64), np.full(len(sch), -1, np.int64)
    for i, (s, t) in enumerate(zip(sch.station, sch.target)):
        ws = order[t_ptr[t]:t_ptr[t + 1]]
        hit = ws[(pb.w_st[ws] == s) & (wL[ws] <= s0[i]) & (e0[i] <= wR[ws])]
        if len(hit):
            L[i], R[i] = wL[hit[0]], wR[hit[0]]
    return L, R


def shrink(L: np.ndarray, R: np.ndarray, B: int, end: int):
    """不留余量的区间两端各往里收 B 秒；规划期的起点 0 和终点 end（Config.end_s，窗口已截到这里）不收，
    与 refine.probe_bounds 相同。"""
    return np.where(L > 0, L + B, L), np.where(R < end, R - B, R)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--margin", type=int, required=True, help="边界余量 B（秒，1～10）")
    ap.add_argument("--plan", help="要加余量的计划（CSV 或提交格式 JSON），默认最好计划")
    ap.add_argument("--minutes", type=float, default=10.0, help="弹射链修补的时长上限（分钟）")
    ap.add_argument("--out", help="输出目录，默认 方案H初版/outputs/edge_margin/")
    args = ap.parse_args()
    B = int(args.margin)
    if not 1 <= B <= 10:
        ap.error("--margin 要在 1～10 秒之间")
    # 自己给的相对路径按当前目录算；默认值在 方案H初版/ 下
    out = Path(args.out).resolve() if args.out else ROOT / "outputs" / "edge_margin"
    out.mkdir(parents=True, exist_ok=True)
    plan = Path(args.plan).resolve() if args.plan else ROOT / "outputs" / "final_best" / "longrun_chain_best.csv"

    cfg0 = Config()
    ds = load(cfg0)
    pb0 = Problem.build(cfg0, ds, get_windows(cfg0, ds))
    sch = read_plan(cfg0, ds, plan)[0]
    e_in = evaluate(cfg0, ds, sch)
    end = cfg0.end_s                                     # 规划期最后一秒（规范 v3.0 截止 08-06 23:59:59）
    L, R = raw_bounds(pb0, sch)
    LB, RB = shrink(L, R, B, end)
    s0 = np.rint(sch.start_s).astype(np.int64)
    e0 = s0 + np.rint(sch.dur_s).astype(np.int64)
    ns, ne = np.maximum(s0, LB), np.minimum(e0, RB)
    keep = (L >= 0) & (ne - ns >= cfg0.obs_min_s)       # 对不上窗口的任务没法确认余量，一并删掉交给修补
    moved = int(((ns != s0) | (ne != e0)).sum())
    cut = Schedule(sch.station[keep], sch.target[keep], ns[keep].astype(float), (ne - ns)[keep].astype(float),
                   algo=f"边界余量 {B} 秒裁边")
    cfgB = cfg0.with_(edge_margin_s=B)
    pbB = Problem.build(cfgB, ds, get_windows(cfgB, ds))
    e_cut = evaluate(cfgB, ds, cut)
    print(f"余量 {B} 秒：原计划 {len(sch)} 个任务、得分 {e_in['score']:.1f}；对不上窗口 {int((L < 0).sum())} 个；"
          f"裁边 {moved} 个、删掉 {int((~keep).sum())} 个，剩 {len(cut)} 个、得分 {e_cut['score']:.1f}、"
          f"覆盖 {e_cut['covered']} 颗", flush=True)

    t0 = time.time()
    rep, ci = improve(pbB, cut, minutes=args.minutes) if args.minutes > 0 else (cut, None)
    t_rep = time.time() - t0
    spec_fit = None
    if fit_applies(cfgB):                                # 与 `obsplan plan` 相同：裁进规范的时间范围
        rep, spec_fit = fit_to_range(cfgB, rep)
    e_rep = evaluate(cfgB, ds, rep)
    L2, R2 = raw_bounds(pb0, rep)                        # 修补后再按不留余量的边界核对一遍
    LB2, RB2 = shrink(L2, R2, B, end)
    rs = np.rint(rep.start_s).astype(np.int64)
    re = rs + np.rint(rep.dur_s).astype(np.int64)
    short_of_margin = int(((L2 < 0) | (rs < LB2) | (re > RB2)).sum())
    chk = validate(cfgB, ds, rep)
    print(f"{'修补 %.0f 秒' % t_rep if ci is not None else '不修补'}：得分 {e_rep['score']:.1f}、覆盖 {e_rep['covered']} 颗、任务 {e_rep['tasks']} 个；"
          f"余量不足 {short_of_margin} 个；Skyfield 校验 {'合法' if chk['legal'] else chk['violations']}", flush=True)

    export_csv(cfgB, ds, rep, out / f"plan_edge{B}.csv", chk)
    notes = write_json(cfgB, ds, rep, out / f"plan_edge{B}.json")
    rel = lambda p: str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name     # noqa: E731  不写本机绝对路径
    notes = {k: v for k, v in notes.items() if k != "path"}
    info = {"margin_s": B, "plan": rel(plan), "minutes": args.minutes,
            "input": {k: e_in[k] for k in ("covered", "score", "combined", "tasks")},
            "clipped": {"moved": moved, "dropped": int((~keep).sum()), "unmatched": int((L < 0).sum()),
                        **{k: e_cut[k] for k in ("covered", "score", "combined", "tasks")}},
            "repaired": {k: e_rep[k] for k in ("covered", "score", "combined", "tasks")},
            "short_of_margin": short_of_margin, "legal": bool(chk["legal"]), "violations": chk["violations"],
            "chain_seconds": None if ci is None else ci["seconds"], "spec_fit": spec_fit, "submission_notes": notes}
    (out / f"repair_edge{B}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=float),
                                              encoding="utf-8")
    return 0 if chk["legal"] and short_of_margin == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
