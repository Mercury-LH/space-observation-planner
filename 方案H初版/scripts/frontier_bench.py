"""时间-质量前沿测速：把 `python -m obsplan plan <参数>` 当成黑盒，端到端跑 N 次，取中位数。

    uv run python scripts/frontier_bench.py --name fast_chain1 --runs 3 --validate -- --mode fast --chain 1

每次运行都是一个全新的进程（含读数据、读窗口缓存、规划、精修、弹射链、写文件），计时不含 Skyfield 校验。
记录：端到端墙钟秒数、进程 CPU 秒数（用户态 + 内核态，含子进程）→ 平均占用核数、各阶段用时（来自 summary.json）、
覆盖 / 得分 / 综合。`--validate` 时再对每次的计划单独跑 `python -m obsplan validate`（不计入用时）。
结果写到 results/frontier/<name>.json；README 的"多少时间到多少分"表由 scripts/frontier_table.py 从这些文件生成。
计时期间不要同时跑别的实验。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _children_cpu() -> float | None:
    try:
        import resource
    except ImportError:                     # Windows 没有 resource：不记 CPU 时间
        return None
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def _load_avg() -> float | None:
    try:
        return round(os.getloadavg()[0], 2)
    except (AttributeError, OSError):
        return None


def run_once(plan_args: list[str], out: Path, validate: bool, env: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "obsplan", "plan", *plan_args, "--out", str(out)]
    load = _load_avg()
    c0, t0 = _children_cpu(), time.perf_counter()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    c1 = _children_cpu()
    (out / "stdout.log").write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
    if p.returncode != 0:
        raise SystemExit(f"规划失败（返回码 {p.returncode}）：\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    m = s["metrics"]
    rec = {"wall_s": round(wall, 1), "cpu_s": None if c0 is None else round(c1 - c0, 1),
           "avg_cores": None if c0 is None else round((c1 - c0) / wall, 2), "load_before": load,
           "phases_s": s["seconds"], "covered": m["covered"], "score": round(m["score"], 2),
           "combined": round(m["combined"], 6), "tasks": m["tasks"], "repeat_tasks": m["repeat_tasks"],
           "chain": s.get("chain"), "plan": str((out / "plan.csv").relative_to(ROOT))}
    if validate:
        rep = out / "validate.json"
        t0 = time.perf_counter()
        v = subprocess.run([sys.executable, "-m", "obsplan", "validate", str(out / "plan.csv"), "--out", str(rep)],
                           cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        r = json.loads(rep.read_text(encoding="utf-8"))
        rec["skyfield"] = {"legal": bool(r["legal"]), "violations": r["violations"], "covered_valid": r["covered_valid"],
                           "score_valid": round(r["score_valid"], 2), "seconds": round(time.perf_counter() - t0, 1),
                           "returncode": v.returncode}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="这组设置的名字（结果文件名）")
    ap.add_argument("--label", help="写进对照表的档位名（默认用 name）")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--validate", action="store_true", help="每次的计划都做 Skyfield 逐秒校验（不计入用时）")
    ap.add_argument("--note", default="", help="这组设置的一句话说明")
    ap.add_argument("--env", action="append", default=[], help="额外环境变量 KEY=VALUE（可多次）")
    ap.add_argument("--out-dir", default="results/frontier")
    ap.add_argument("plan_args", nargs=argparse.REMAINDER, help="`--` 之后是传给 obsplan plan 的参数")
    a = ap.parse_args()
    plan_args = [x for x in a.plan_args if x != "--"]
    env = dict(os.environ)
    extra = dict(kv.split("=", 1) for kv in a.env)
    env.update(extra)
    runs = []
    for i in range(a.runs):
        rec = run_once(plan_args, ROOT / "outputs" / "frontier" / a.name / f"run{i + 1}", a.validate, env)
        runs.append(rec)
        sky = "" if "skyfield" not in rec else ("，Skyfield 合法" if rec["skyfield"]["legal"] else "，Skyfield 违规！")
        print(f"[{a.name}] 第 {i + 1}/{a.runs} 次：{rec['wall_s']} 秒，平均 {rec['avg_cores']} 核，"
              f"综合 {rec['combined']:.6f}（{rec['covered']} 颗 / {rec['score']} 分）{sky}", flush=True)
    med = lambda k: statistics.median(r[k] for r in runs)        # noqa: E731
    res = {"name": a.name, "label": a.label or a.name, "note": a.note, "plan_args": plan_args, "env": extra,
           "machine": {"platform": platform.platform(), "processor": platform.processor(), "cpus": os.cpu_count()},
           "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "median": {"wall_s": med("wall_s"), "combined": med("combined"), "covered": med("covered"),
                      "score": med("score"),
                      "avg_cores": None if runs[0]["avg_cores"] is None else med("avg_cores")},
           "all_legal": all(r["skyfield"]["legal"] for r in runs) if a.validate else None,
           "runs": runs}
    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    m = res["median"]
    print(f"[{a.name}] 中位数：{m['wall_s']} 秒，综合 {m['combined']:.6f}，平均 {m['avg_cores']} 核"
          + ("" if res["all_legal"] is None else ("，全部合法" if res["all_legal"] else "，有违规")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
