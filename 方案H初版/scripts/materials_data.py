"""为项目书和答辩准备图表数据，输出到 outputs/materials/。

- 三种约束口径下的窗口统计（只看仰角 / 加站点黑夜 / 再加目标受照）
- 受照窗口在"黑夜中的位置"分布（低轨 vs 中高轨）
- 方案 H 的时间格价格、计划（用于价格热力图和甘特图）
- 覆盖/得分权重扫描：E（每个权重重训 GA）与 H 的取舍曲线
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsplan import ga, lagrange  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.greedy import algo_weighted  # noqa: E402
from obsplan.local_search import improve  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.schedule import evaluate  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "materials"


def night_position(pb: Problem) -> np.ndarray:
    pos = np.full(len(pb.w_k0), 0.5)
    for s, runs in enumerate(pb.win.dark):
        if not runs:
            continue
        d0 = np.array([a for a, _ in runs])
        d1 = np.array([b for _, b in runs])
        idx = np.nonzero(pb.w_st == s)[0]
        mid = (pb.w_k0[idx] + pb.w_k1[idx]) / 2
        r = np.clip(np.searchsorted(d0, mid, side="right") - 1, 0, len(d0) - 1)
        pos[idx] = np.clip((mid - d0[r]) / np.maximum(d1[r] - d0[r], 1), 0, 1)
    return pos


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    ds = load(cfg)
    mm = np.array([float(l[52:63]) for l in ds.line2])
    leo = mm > 11.25
    stats = []
    for name, c in (("只看仰角 10°–85°", cfg.with_(use_sun=False, use_sunlit=False)),
                    ("+ 站点黑夜（太阳角 < −6°）", cfg.with_(use_sunlit=False)),
                    ("+ 目标受照（答疑口径）", cfg)):
        w = get_windows(c, ds)
        L = (w.k1 - w.k0) * c.cell_s / 3600
        stats.append({"name": name, "windows": int(len(w.k0)), "targets": int(np.unique(w.target).size),
                      "leo_window_h": float(L[leo[w.target]].sum()), "other_window_h": float(L[~leo[w.target]].sum()),
                      "dark_h": float(sum((b - a) * c.cell_s for runs in w.dark for a, b in runs) / 3600)})
        print(stats[-1])
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    pos = night_position(pb)
    wl = (pb.w_k1 - pb.w_k0).astype(float)
    bins = np.linspace(0, 1, 21)
    hist_leo, _ = np.histogram(pos[leo[pb.w_tg]], bins, weights=wl[leo[pb.w_tg]])
    hist_oth, _ = np.histogram(pos[~leo[pb.w_tg]], bins, weights=wl[~leo[pb.w_tg]])

    t = time.time()
    sch, info = lagrange.solve(pb, iters=300, verbose=False)
    print(f"H done {time.time() - t:.0f}s", evaluate(cfg, ds, sch)["combined"])
    np.savez_compressed(OUT / "h_run.npz", mu=info["mu"], station=sch.station, target=sch.target,
                        start_s=sch.start_s, dur_s=sch.dur_s, P=pb.P[sch.target], leo=leo[sch.target],
                        hist=np.array([h[1] for h in info["history"]]), hist_lb=np.array([h[2] for h in info["history"]]),
                        bins=bins, hist_leo=hist_leo, hist_oth=hist_oth,
                        dark=np.array([(s, a, b) for s, runs in enumerate(pb.win.dark) for a, b in runs]))
    (OUT / "window_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    pts = []
    for wc in (1.0, 0.75, 0.5, 0.25, 0.0):
        c2 = cfg.with_(w_cov=wc, w_score=1.0 - wc)
        p2 = Problem.build(c2, ds, pb.win)
        w_ai, gi = ga.train(p2, verbose=False)
        t = time.time()
        _, st = algo_weighted(p2, w_ai, "E", return_state=True)
        improve(p2, st)
        e = st.schedule(p2, "E", gi["train_s"] + time.time() - t)
        h, _ = lagrange.solve(p2, iters=300, verbose=False)
        for s in (e, h):
            ev = evaluate(c2, ds, s)
            pts.append({"w_cov": wc, "algo": s.algo[0], "covered": ev["covered"], "score": ev["score"],
                        "runtime_s": s.runtime_s})
            print(pts[-1])
        (OUT / "sweep.json").write_text(json.dumps(pts, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
