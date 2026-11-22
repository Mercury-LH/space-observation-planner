"""复核伐谋候选：在未参与训练的子场景和全量场景上，与人工权重、GA 权重、新算法 H 比较。

用法（在 方案H初版/ 下）：
    uv run python scripts/famou_review.py 候选1.py 候选2.py ...
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
from obsplan.famou_eval import Case, run_candidate  # noqa: E402
from obsplan.greedy import MANUAL_WEIGHTS, ai_values, feature_matrix, run_greedy, window_features  # noqa: E402
from obsplan.lagrange import objective  # noqa: E402
from obsplan.local_search import improve  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.schedule import evaluate  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

HOLDOUT_SEEDS = (303, 404, 505)
ROOT = Path(__file__).resolve().parents[1]


def greedy_from_scores(case: Case, u, c):
    v1 = np.asarray(u, float)
    v2 = np.minimum(v1, np.asarray(c, float))
    return run_greedy(case.pb, case.cw, case.cm, v1, v2)


def linear_scores(case: Case, w):
    F = feature_matrix(case.pb, window_features(case.pb), case.cw, case.cm)
    return ai_values(F, w)


def main():
    progs = [Path(p) for p in sys.argv[1:]]
    cfg = Config()
    ds = load(cfg)
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    print("训练 GA 权重（与对比实验同一设置）……")
    w_ga, _ = ga.train(pb, verbose=False)
    holdout = [Case(ga.sub_problem(pb, np.random.default_rng(s), target_frac=0.2, n_st=3)) for s in HOLDOUT_SEEDS]
    full = Case(pb)
    rows = []
    # 基线：人工权重、GA 权重
    baselines = {"人工权重（初始程序）": MANUAL_WEIGHTS, "GA 学到的权重（方案 D）": w_ga}
    for name, w in baselines.items():
        r = [objective(c.pb, run_greedy(c.pb, c.cw, c.cm, *linear_scores(c, w))) / c.base for c in holdout]
        t = time.time()
        st = run_greedy(full.pb, full.cw, full.cm, *linear_scores(full, w))
        g = objective(full.pb, st)
        improve(full.pb, st)
        rows.append({"name": name, "holdout": np.round(r, 4).tolist(), "full_greedy": g,
                     "full_ls": objective(full.pb, st), "s": time.time() - t})
    for p in progs:
        src = p.read_bytes()
        try:
            res = run_candidate(src, holdout, timeout_s=120)
            r = [objective(c.pb, greedy_from_scores(c, u, cc)) / c.base for c, (u, cc) in zip(holdout, res)]
            t = time.time()
            (u, cc), = run_candidate(src, [full], timeout_s=900, fsize=1 << 30)
            st = greedy_from_scores(full, u, cc)
            g = objective(full.pb, st)
            improve(full.pb, st)
            rows.append({"name": p.name, "holdout": np.round(r, 4).tolist(), "full_greedy": g,
                         "full_ls": objective(full.pb, st), "s": time.time() - t})
        except Exception as e:  # 候选在新场景出错也是一种结果
            rows.append({"name": p.name, "error": str(e)[:300]})
    sch, _ = lagrange.solve(pb, iters=300, verbose=False)
    rows.append({"name": "H 价格协调", "full_ls": evaluate(cfg, ds, sch)["combined"], "s": sch.runtime_s})
    print(f"\n{'程序':28s} {'留出场景/人工':>22s} {'全量贪心':>9s} {'全量+局部搜索':>13s}")
    for r in rows:
        if "error" in r:
            print(f"{r['name']:28s} 出错：{r['error']}")
            continue
        ho = r.get("holdout")
        print(f"{r['name']:28s} {str(ho) if ho else '':>22s} {r.get('full_greedy', float('nan')):9.4f} {r['full_ls']:13.4f}")
    out = ROOT / "outputs" / "famou_review.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
