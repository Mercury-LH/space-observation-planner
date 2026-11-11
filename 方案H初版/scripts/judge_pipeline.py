"""神经网络放置规则的完整流程：合成场景 → 老师 → 特征与标签 → 训练 → 阈值 → 跨场景基准。

    # 场景（侯治民的生成器重建输入 + 我们自己算窗口）
    uv run python scripts/judge_pipeline.py scenes --split train --n 120 --jobs 8
    # --dataset：v2 = 他的 synthetic-v0.2；v3 = 我们补的"像赛题"场景（低轨更低、有高纬度站、中轨少、更挤）
    uv run python scripts/judge_pipeline.py scenes --dataset v2,v3 --split train --n 60,60 --jobs 8
    # 老师：每个训练场景 30 分钟精简长跑
    uv run python scripts/judge_pipeline.py teacher --split train --n 120 --minutes 30 --jobs 8 --threads 8
    # 特征与标签（H 收敛价格下每个候选的 20 个特征；标签 = 老师计划是否选中）
    uv run python scripts/judge_pipeline.py export --split train --n 120 --jobs 8 --threads 8
    # 训练（只用 numpy），按场景留出 15% 选最好的一轮
    uv run python scripts/judge_pipeline.py train --split train --n 120
    # 阈值：在验证场景上端到端选 (t_u, t_c)
    uv run python scripts/judge_pipeline.py tune --split validation --n 24 --jobs 8 --threads 8
    # 基准：测试场景上 H / H+ / 网络 / 稳健 / 稳健+网络，加按秒口径上界与 Skyfield 校验
    uv run python scripts/judge_pipeline.py bench --split test --jobs 8 --threads 8 --skyfield
    uv run python scripts/judge_pipeline.py summary
    # 赛题本身（网络没见过）：同样五种做法 + 上界 + Skyfield，结果 outputs/judge/contest.json
    uv run python scripts/judge_pipeline.py contest --skyfield

每个场景的结果单独存盘（outputs/judge/<步骤>/<场景>.*），中断后再跑只补缺的。
--jobs > 1 时每个场景开一个子进程（--threads 是每个子进程的 numba 线程数，CP-SAT 用同样多的线程）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from obsplan import scenes  # noqa: E402

OUT = scenes.OUT
RESULT = ROOT / "results" / "judge_benchmark.json"
NEG_KEEP = 0.25           # 负例抽样比例（正例全留），训练时负例权重 1 / NEG_KEEP


def ids_of(args) -> list[str]:
    if args.ids:
        return args.ids.split(",")
    sets = args.dataset.split(",")
    ns = [None] * len(sets) if args.n is None else [int(x) for x in str(args.n).split(",")]
    if len(ns) == 1 and len(sets) > 1:
        ns = ns * len(sets)
    if len(ns) != len(sets):
        raise SystemExit("--n 的个数要和 --dataset 一致，例如 --dataset v2,v3 --n 60,60")
    return [c for d, n in zip(sets, ns) for c in scenes.select(args.split, n, dataset=d)]


def _set_threads(n: int) -> None:
    if n:
        import numba
        numba.set_num_threads(n)


def _done(step: str, cid: str) -> bool:
    ext = {"scenes": None, "teacher": ".json", "export": ".npz", "tune": ".json", "bench": ".json"}[step]
    if step == "scenes":
        d = scenes.scene_dir(cid)
        return (d / "spec.json").exists() and any((OUT / "cache" / cid).glob("windows_*.npz"))
    return (OUT / step / f"{cid}{ext}").exists()


def run_many(step: str, ids: list[str], args, extra: list[str]) -> None:
    """每个场景一个子进程，最多 --jobs 个同时跑。"""
    todo = [c for c in ids if not _done(step, c)]
    print(f"{step}：共 {len(ids)} 个场景，还要跑 {len(todo)} 个，同时 {args.jobs} 个", flush=True)
    logdir = OUT / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    def one(cid):
        cmd = [sys.executable, str(Path(__file__).resolve()), step, "--ids", cid, "--jobs", "1",
               "--threads", str(args.threads), *extra]
        t = time.time()
        with open(logdir / f"{step}_{cid}.log", "w", encoding="utf-8") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        print(f"  {cid} {'完成' if r.returncode == 0 else '失败（见日志）'}，{time.time() - t:.0f} 秒", flush=True)
        return r.returncode

    with ThreadPoolExecutor(args.jobs) as ex:
        codes = list(ex.map(one, todo))
    bad = sum(1 for c in codes if c)
    if bad:
        print(f"{bad} 个场景失败", flush=True)


# ---------------------------------------------------------------- 各步骤（单个场景）
def step_scenes(cid: str, args) -> None:
    t = time.time()
    cfg, ds, pb, spec = scenes.load_scene(cid, workers=args.threads or 8)
    print(f"{cid}：{ds.n_targets} 目标 {ds.n_stations} 站，{len(pb.w_st)} 个窗口，{time.time() - t:.0f} 秒", flush=True)


def step_teacher(cid: str, args) -> None:
    from obsplan.evolved import load_rule
    from obsplan.teacher import teach
    cfg, ds, pb, spec = scenes.load_scene(cid, workers=args.threads or 8)
    d = OUT / "teacher"
    d.mkdir(parents=True, exist_ok=True)
    best, info = teach(pb, load_rule(), minutes=args.minutes, workers=args.threads or 8,
                       log=lambda r: print(json.dumps(r, ensure_ascii=False), flush=True))
    start = info.pop("start")
    np.savez_compressed(d / f"{cid}.npz", station=best.station, target=best.target, start_s=best.start_s,
                        dur_s=best.dur_s, s_station=start.station, s_target=start.target, s_start_s=start.start_s,
                        s_dur_s=start.dur_s)
    info.update(catalog_id=cid, n_targets=ds.n_targets, n_stations=ds.n_stations, minutes=args.minutes)
    (d / f"{cid}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{cid}：老师 {info['start_obj']:.4f} → {info['best_obj']:.4f}（{info['gain_rel']:+.2%}），"
          f"{len(info['rounds'])} 轮", flush=True)


def _teacher_plans(cid: str):
    from obsplan.schedule import Schedule
    z = np.load(OUT / "teacher" / f"{cid}.npz")
    best = Schedule(z["station"], z["target"], z["start_s"], z["dur_s"])
    start = Schedule(z["s_station"], z["s_target"], z["s_start_s"], z["s_dur_s"])
    return best, start


def _prices(pb, iters: int = 300):
    """H 的价格迭代（人工规则恢复、不做局部搜索）；放置规则用的就是这组价格。"""
    from obsplan import lagrange
    _, info = lagrange.solve(pb, iters=iters, local_search=False, verbose=False)
    return info["mu"]


def step_export(cid: str, args) -> None:
    from obsplan import judge
    from obsplan.famou_h_eval import feature_rows
    from obsplan.greedy import candidates
    cfg, ds, pb, spec = scenes.load_scene(cid, workers=args.threads or 8)
    t = time.time()
    mu = _prices(pb)
    cw, cm = candidates(pb)
    X = feature_rows(pb, mu, cw, cm)
    Z = judge.inputs(X)
    best, start = _teacher_plans(cid)
    y = judge.labels(pb, cw, cm, best)
    y0 = judge.labels(pb, cw, cm, start)
    rng = np.random.default_rng(int(hashlib.md5(cid.encode()).hexdigest()[:8], 16))
    keep = (y == 1) | (y0 == 1) | (rng.random(len(y)) < NEG_KEEP)
    w = np.where((y == 1) | (y0 == 1), 1.0, 1.0 / NEG_KEEP).astype(np.float32)
    d = OUT / "export"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{cid}.npz", Z=Z[keep], y=y[keep], y_start=y0[keep], w=w[keep],
                        n_candidates=len(y), n_pos=int(y.sum()), n_pos_start=int(y0.sum()))
    print(f"{cid}：{len(y)} 个候选，老师选中 {int(y.sum())}，起点选中 {int(y0.sum())}，"
          f"两者都选 {int(((y == 1) & (y0 == 1)).sum())}，存 {int(keep.sum())} 行，{time.time() - t:.0f} 秒", flush=True)


def _end_to_end(pb, mu, rule, rmp=None):
    """固定价格 mu：按规则恢复 + 完整局部搜索，返回网格目标值。"""
    from obsplan.greedy import candidates
    from obsplan.lagrange import objective, recover
    from obsplan.local_search import improve
    st = recover(pb, mu, candidates(pb), rule=rule)
    improve(pb, st, repeat_min_priority=rmp)
    return objective(pb, st)


TUNE_GRID = [(tu, tc) for tu in (-1.0, 0.0, 1.0, 2.0) for tc in (0.0, 1.0, 2.0, 3.0) if tc >= tu]


def step_tune(cid: str, args) -> None:
    from obsplan import judge
    from obsplan.evolved import load_rule
    cfg, ds, pb, spec = scenes.load_scene(cid, workers=args.threads or 8)
    model = judge.load_model(args.model)
    famou = load_rule()
    mu = _prices(pb)
    row = {"catalog_id": cid, "manual": _end_to_end(pb, mu, None), "famou": _end_to_end(pb, mu, famou), "judge": {}}
    for tu, tc in TUNE_GRID:
        row["judge"][f"{tu},{tc}"] = _end_to_end(pb, mu, judge.make_rule(model, tu, tc, famou))
        print(cid, tu, tc, row["judge"][f"{tu},{tc}"], flush=True)
    d = OUT / "tune"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{cid}.json").write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")


def _metrics(cfg, ds, pb, sch, seconds, near=None) -> dict:
    from obsplan.checks import structural_violations
    from obsplan.schedule import evaluate
    e = evaluate(cfg, ds, sch)
    v = structural_violations(pb, sch, by_second=True, extra_windows=near)
    return {"covered": int(e["covered"]), "score": round(float(e["score"]), 3), "combined": round(float(e["combined"]), 6),
            "tasks": int(e["tasks"]), "repeat_tasks": int(e["repeat_tasks"]), "seconds": round(seconds, 1),
            "violations": int(sum(len(x) for x in v.values()))}


METHODS = ("H", "H+", "网络", "稳健", "稳健+网络")


def solve_method(pb, name: str, famou, judge_rule, fast: bool = False):
    from obsplan import lagrange
    rule = {"H": None, "H+": famou, "网络": judge_rule, "稳健": famou, "稳健+网络": [famou, judge_rule]}[name]
    return lagrange.solve(pb, iters=150 if fast else 300, fast=fast, verbose=False, rule=rule,
                          portfolio=name.startswith("稳健"))


def step_bench(cid: str, args) -> None:
    cfg, ds, pb, spec = scenes.load_scene(cid, workers=args.threads or 8)
    row = {"catalog_id": cid, "split": spec["split"], "group": spec["sampling_group"], "stress_type": spec["stress_type"],
           "n_targets": int(ds.n_targets), "n_stations": int(ds.n_stations), "start": spec["start_utc"][:10]}
    row = bench_problem(cfg, ds, pb, row, args)
    d = OUT / "bench"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{cid}.json").write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")


def do_contest(args) -> None:
    """赛题本身（网络只在合成场景上训练过，从没见过它）：同样的五种做法 + 上界 + Skyfield。"""
    from obsplan.config import Config
    from obsplan.data import load
    from obsplan.problem import Problem
    from obsplan.visibility import get_windows
    _set_threads(args.threads)
    cfg = Config()
    ds = load(cfg)
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    row = bench_problem(cfg, ds, pb, {"catalog_id": "赛题", "split": "contest", "group": "contest", "stress_type": None,
                                      "n_targets": int(ds.n_targets), "n_stations": int(ds.n_stations),
                                      "start": cfg.start_utc[:10]}, args)
    (OUT / "contest.json").write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False, indent=1))


def bench_problem(cfg, ds, pb, row: dict, args) -> dict:
    from obsplan import judge
    from obsplan.bounds import by_second_problem, flow_upper_bound
    from obsplan.evolved import load_rule
    from obsplan.refine import finish, near_windows
    famou = load_rule()
    judge_rule = judge.make_rule(judge.load_model(args.model), famou=famou)
    row.update(observable=int((pb.max_obs > 0).sum()), windows=int(len(pb.w_st)),
               mode="fast" if args.fast else "quality")
    plans = {}
    for name in METHODS:
        t = time.time()
        sch, info = solve_method(pb, name, famou, judge_rule, args.fast)
        sch, rinfo = finish(pb, sch)
        near = near_windows(pb) if rinfo.get("insert", {}).get("added") else None
        row[name] = _metrics(cfg, ds, pb, sch, time.time() - t, near)
        if "portfolio" in info:
            row[name]["picked"] = info["portfolio"]["picked"]
        plans[name] = sch
        print(cid, name, row[name], flush=True)
    t = time.time()
    ub = flow_upper_bound(by_second_problem(pb), verbose=False)["combined_ub"]
    row["bound"] = {"combined_ub": round(float(ub), 6), "seconds": round(time.time() - t, 1), "scope": "按秒口径网络流"}
    for name in METHODS:
        row[name]["gap_pct"] = round(100 * (ub - row[name]["combined"]) / ub, 4)
    if args.skyfield:
        from obsplan.validator import validate
        seen = {}
        for name in METHODS:
            sch = plans[name]
            key = hashlib.md5(np.concatenate([sch.station, sch.target, sch.start_s, sch.dur_s]).astype(np.int64)
                              .tobytes()).hexdigest()
            if key not in seen:
                v = validate(cfg, ds, sch)
                seen[key] = {"legal": bool(v["legal"]), "violations": v["violations"]}
            row[name]["skyfield"] = seen[key]
    return row


# ---------------------------------------------------------------- 训练、阈值、汇总（全体场景）
def load_export(ids: list[str]):
    Zs, ys, ws, sid = [], [], [], []
    for i, cid in enumerate(ids):
        p = OUT / "export" / f"{cid}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        Zs.append(z["Z"])
        ys.append(z["y"].astype(np.float32))
        ws.append(z["w"])
        sid.append(np.full(len(z["y"]), i, np.int32))
    return np.concatenate(Zs), np.concatenate(ys), np.concatenate(ws), np.concatenate(sid)


def do_train(ids: list[str], args) -> None:
    from obsplan import judge
    have = [c for c in ids if (OUT / "export" / f"{c}.npz").exists()]
    rng = np.random.default_rng(20261111)
    hold = set(rng.choice(have, max(1, int(round(len(have) * args.holdout))), replace=False))
    tr = [c for c in have if c not in hold]
    print(f"训练场景 {len(tr)} 个，留出场景 {len(hold)} 个", flush=True)
    Z, y, w, _ = load_export(tr)
    Zv, yv, wv, _ = load_export(sorted(hold))
    mean = (Z * w[:, None]).sum(0) / w.sum()
    std = np.sqrt(((Z - mean) ** 2 * w[:, None]).sum(0) / w.sum()) + 1e-6
    S, Sv = judge.standardize(Z, mean, std), judge.standardize(Zv, mean, std)
    print(f"训练 {len(y)} 行（正例 {int(y.sum())}），留出 {len(yv)} 行", flush=True)
    # 对照：直接用伐谋规则 / 人工规则的打分排序，看网络是否比它们更接近老师
    base = {k: judge.evaluate_logits(Zv[:, judge.INPUTS.index(k)].astype(np.float64), yv, wv)
            for k in ("manual_u", "famou_u")}
    for k, v in base.items():
        print(f"  对照 {k}：AUC {v['auc']:.4f}  前 2% 命中率 {v['prec_at_pos']:.3f}", flush=True)
    hidden = tuple(int(x) for x in args.hidden.split(","))
    t = time.time()
    p, hist = judge.fit(S, y, w, Sv, yv, wv, hidden=hidden, epochs=args.epochs, lr=args.lr)
    meta = {"train_scenes": tr, "holdout_scenes": sorted(hold), "hidden": hidden, "epochs": args.epochs,
            "history": hist, "baselines": base, "train_rows": int(len(y)), "seconds": round(time.time() - t, 1),
            "teacher": "精简长跑（obsplan/teacher.py）", "neg_keep": NEG_KEEP}
    judge.save(Path(args.model), p, mean, std, 0.0, 0.0, meta)
    print(f"已保存 {args.model}（{sum(v.size for v in p.values())} 个参数，训练 {time.time() - t:.0f} 秒）", flush=True)


def do_tune_pick(ids: list[str], args) -> None:
    """汇总各验证场景的阈值结果，选相对伐谋规则平均最好的一组，写进模型文件。"""
    from obsplan import judge
    rows = [json.loads((OUT / "tune" / f"{c}.json").read_text(encoding="utf-8")) for c in ids
            if (OUT / "tune" / f"{c}.json").exists()]
    if not rows:
        print("没有阈值结果")
        return
    keys = list(rows[0]["judge"])
    rel = {k: np.mean([r["judge"][k] / r["famou"] - 1 for r in rows]) for k in keys}
    rel_manual = np.mean([r["manual"] / r["famou"] - 1 for r in rows])
    for k in keys:
        print(f"  t_u,t_c = {k}：相对伐谋规则 {rel[k]:+.4%}")
    print(f"  人工规则相对伐谋规则 {rel_manual:+.4%}（{len(rows)} 个验证场景）")
    k = max(keys, key=lambda k: rel[k])
    tu, tc = (float(x) for x in k.split(","))
    model = judge.load_model(args.model)
    meta = {**model["meta"], "tune": {"scenes": [r["catalog_id"] for r in rows], "rel_vs_famou": rel,
                                     "manual_vs_famou": rel_manual, "picked": k}}
    judge.save(Path(args.model), model["p"], model["mean"], model["std"], tu, tc, meta)
    print(f"选 t_u={tu}、t_c={tc}，已写入 {args.model}")


def _bootstrap_ci(x: np.ndarray, n: int = 10000, seed: int = 20261111):
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def do_summary(args) -> None:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "bench").glob("*.json"))]
    if not rows:
        print("没有基准结果")
        return

    def diff(sub, a, b):
        d = np.array([r[a]["combined"] - r[b]["combined"] for r in sub])
        lo, hi = _bootstrap_ci(d)
        return {"scenes": len(sub), "mean": round(float(d.mean()), 6), "ci95": [round(lo, 6), round(hi, 6)],
                "median": round(float(np.median(d)), 6), "min": round(float(d.min()), 6),
                "better": int((d > 1e-9).sum()), "worse": int((d < -1e-9).sum())}

    def per(sub, m):
        g = np.array([r[m]["gap_pct"] for r in sub])
        return {"gap_pct_median": round(float(np.median(g)), 3), "gap_pct_mean": round(float(g.mean()), 3),
                "gap_pct_max": round(float(g.max()), 3),
                "seconds_mean": round(float(np.mean([r[m]["seconds"] for r in sub])), 1),
                "violations": int(sum(r[m]["violations"] for r in sub)),
                "skyfield_illegal": int(sum(1 for r in sub if not r[m].get("skyfield", {}).get("legal", True)))}

    groups = {"全部": rows, "主体": [r for r in rows if r["group"] == "core"],
              "压力": [r for r in rows if r["group"] == "stress"],
              "v0.3 像赛题": [r for r in rows if r["group"] == "v0.3"]}
    for k in sorted({r["stress_type"] for r in rows if r["stress_type"]}):
        groups[f"压力:{k}"] = [r for r in rows if r["stress_type"] == k]
    summary = {}
    for g, sub in groups.items():
        if not sub:
            continue
        summary[g] = {"methods": {m: per(sub, m) for m in METHODS},
                      "网络 相对 H+": diff(sub, "网络", "H+"), "网络 相对 H": diff(sub, "网络", "H"),
                      "稳健+网络 相对 稳健": diff(sub, "稳健+网络", "稳健")}
    res = {"说明": "测试场景从未参与训练、阈值选择或任何调参；综合指标按各场景自己的目标数归一化；"
                   "gap_pct = 离按秒口径网络流上界的差距（%），上界对任何合法计划成立。",
           "rows": rows, "summary": summary}
    RESULT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary.get("全部", {}), ensure_ascii=False, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["scenes", "teacher", "export", "train", "tune", "pick", "bench", "contest", "summary"])
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--dataset", default="v2", help="v2（侯治民的 synthetic-v0.2）、v3（像赛题的补充）或 v2,v3")
    ap.add_argument("--n", default=None, help="均匀抽几个场景（默认全部）；多个数据集时逗号分隔，如 60,60")
    ap.add_argument("--ids", help="逗号分隔的场景编号（优先于 --split/--n）")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=30.0, help="老师每个场景的时间")
    ap.add_argument("--model", default=str(ROOT / "obsplan" / "models" / "judge_mlp.npz"))
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--fast", action="store_true", help="基准用快速模式")
    ap.add_argument("--skyfield", action="store_true", help="基准里对每份不同的计划做 Skyfield 逐秒校验")
    args = ap.parse_args()
    if args.step == "summary":
        do_summary(args)
        return 0
    if args.step == "contest":
        do_contest(args)
        return 0
    ids = ids_of(args)
    if args.step == "train":
        do_train(ids, args)
        return 0
    if args.step == "pick":
        do_tune_pick(ids, args)
        return 0
    if args.jobs > 1 and len(ids) > 1:
        extra = ["--minutes", str(args.minutes), "--model", args.model] + (["--fast"] if args.fast else []) \
            + (["--skyfield"] if args.skyfield else [])
        run_many(args.step, ids, args, extra)
        return 0
    _set_threads(args.threads)
    fn = {"scenes": step_scenes, "teacher": step_teacher, "export": step_export, "tune": step_tune,
          "bench": step_bench}[args.step]
    for cid in ids:
        if _done(args.step, cid) and len(ids) > 1:
            continue
        fn(cid, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
