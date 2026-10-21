"""命令行入口。

    uv run python -m obsplan plan [--mode fast|quality] [--start 2026-08-01] [--moon 30] [--weather] [--validate] [--no-refine]
    uv run python -m obsplan validate 计划文件.csv|.json [--start 2026-08-01] [--seconds-from-start]
    uv run python -m obsplan windows [--refresh]

validate 可以复核任何团队成员的计划：CSV 需含 站点/station_id、目标编号/target_id、开始/start、结束/end 列
（时间为 UTC 的 ISO 格式，或配合 --seconds-from-start 使用相对规划开始的秒数）；
JSON 为观测列表或 {"observations": [...]}，字段 station_id、target_id、start、end。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import Config

STATION_KEYS = ("station_id", "站点", "station", "观测站")
TARGET_KEYS = ("target_id", "目标编号", "norad", "satnum", "目标")
START_KEYS = ("observation_start_utc", "start", "start_time", "开始", "开始时间")
END_KEYS = ("observation_end_utc", "end", "end_time", "结束", "结束时间")


def _cfg(args) -> Config:
    kw = {}
    if getattr(args, "start", None):
        kw["start_utc"] = f"{args.start}T00:00:00" if len(args.start) == 10 else args.start
    if getattr(args, "days", None):
        kw["days"] = float(args.days)
    if getattr(args, "moon", None):
        kw["moon_sep_deg"] = float(args.moon)
    return Config(**kw)


def _pick(row: dict, keys) -> str | None:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def _to_seconds(cfg: Config, value, seconds_from_start: bool) -> float:
    if seconds_from_start or isinstance(value, (int, float)):
        return float(value)
    v = str(value).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - cfg.start_dt).total_seconds()


def read_plan(cfg: Config, ds, path: Path, seconds_from_start: bool = False):
    from .schedule import Schedule
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("observations", data.get("tasks", data)) if isinstance(data, dict) else data
        rows = [{k: v for k, v in r.items()} for r in rows]
    else:
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    idx = {int(n): i for i, n in enumerate(ds.satnum)}
    st, tg, s0, du, skipped = [], [], [], [], []
    for n, r in enumerate(rows):
        s, t, a, b = _pick(r, STATION_KEYS), _pick(r, TARGET_KEYS), _pick(r, START_KEYS), _pick(r, END_KEYS)
        if None in (s, t, a, b):
            skipped.append((n, "缺少站点/目标/开始/结束字段"))
            continue
        try:
            sj = int(str(s).replace("号站", "").replace("号", "").strip()) - 1
            ti = idx[int(t)]
        except (ValueError, KeyError):
            skipped.append((n, f"无法识别的站点 {s} 或目标 {t}"))
            continue
        ta, tb = _to_seconds(cfg, a, seconds_from_start), _to_seconds(cfg, b, seconds_from_start)
        st.append(sj)
        tg.append(ti)
        s0.append(ta)
        du.append(tb - ta)
    sch = Schedule(np.array(st, np.int64), np.array(tg, np.int64), np.array(s0, float), np.array(du, float),
                   algo=path.name)
    return sch, skipped


def cmd_validate(args) -> int:
    from .data import load
    from .schedule import evaluate
    from .validator import validate
    cfg = _cfg(args)
    ds = load(cfg)
    sch, skipped = read_plan(cfg, ds, Path(args.plan), args.seconds_from_start)
    print(f"读入 {len(sch)} 个任务（跳过 {len(skipped)} 行）；规划期 {cfg.start_utc} 起 {cfg.days:g} 天；"
          f"规则：仰角 {cfg.elev_min:g}–{cfg.elev_max:g}°，太阳角 < {cfg.sun_max:g}°，目标受照 {'是' if cfg.use_sunlit else '否'}"
          + (f"，月光规避 {cfg.moon_sep_deg:g}°" if cfg.moon_sep_deg > 0 else ""))
    for n, why in skipped[:5]:
        print(f"  第 {n + 2} 行：{why}")
    t = time.time()
    chk = validate(cfg, ds, sch)
    ev = evaluate(cfg, ds, sch, ~chk["invalid_mask"])
    print(f"Skyfield 逐秒校验 {time.time() - t:.0f} 秒：{'全部合法' if chk['legal'] else '发现违规'}")
    for k, v in chk["violations"].items():
        print(f"  {k}：{v} 个任务")
    print(f"按规则复算（只计合法任务）：覆盖 {ev['covered']} 颗，得分 {ev['score']:.1f}，重复观测 {ev['repeat_tasks']} 次")
    if args.out:
        rep = {"plan": str(args.plan), "tasks": len(sch), "skipped": len(skipped), "legal": chk["legal"],
               "violations": chk["violations"], "covered_valid": ev["covered"], "score_valid": ev["score"],
               "config": json.loads(cfg.to_json())}
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print("报告已写入", args.out)
    return 0 if chk["legal"] else 1


def cmd_plan(args) -> int:
    from . import lagrange
    from .data import load
    from .problem import Problem
    from .schedule import evaluate, export_csv
    from .visibility import get_windows
    cfg = _cfg(args)
    t0 = time.time()
    ds = load(cfg)
    if ds.issues:
        print("数据检查：", "；".join(ds.issues))
    win = get_windows(cfg, ds)
    pb = Problem.build(cfg, ds, win)
    t1 = time.time()
    kw = {}
    if args.weather:
        from .environment import fetch_cloud_history
        cloud = fetch_cloud_history(cfg, ds, verbose=False)
        sc = np.nanmean(cloud <= 30.0, axis=(0, 2))
        kw = {"cell_weight": np.repeat(sc[:, None], pb.n_samples + 2, axis=1), "backup": np.zeros(pb.n_targets)}
    fast = args.mode == "fast"
    sch, info = lagrange.solve(pb, iters=150 if fast else 300, fast=fast, verbose=False, **kw)
    rinfo = None
    if not args.no_refine:
        from .refine import finish
        sch, rinfo = finish(pb, sch)
    t2 = time.time()
    ev = evaluate(cfg, ds, sch)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    check = None
    if args.validate:
        from .validator import validate
        check = validate(cfg, ds, sch)
    export_csv(cfg, ds, sch, out / "plan.csv", check)
    summary = {"config": json.loads(cfg.to_json()), "mode": args.mode, "weather_aware": bool(args.weather),
               "metrics": ev, "observable_targets": int((pb.max_obs > 0).sum()), "sgp4_failed": int(win.sgp4_failed.sum()),
               "seconds": {"data_and_windows": round(t1 - t0, 1), "planning": round(t2 - t1, 1)}, "refine": rinfo,
               "validation": None if check is None else {"legal": check["legal"], "violations": check["violations"]}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"覆盖 {ev['covered']} 颗（可观测 {summary['observable_targets']}），得分 {ev['score']:.1f}，"
          f"任务 {ev['tasks']} 个（重复 {ev['repeat_tasks']}）；数据与窗口 {t1 - t0:.0f} 秒，规划 {t2 - t1:.0f} 秒")
    if check is not None:
        print("Skyfield 校验：", "全部合法" if check["legal"] else check["violations"])
    print("已写出", out / "plan.csv", "和", out / "summary.json")
    return 0


def cmd_windows(args) -> int:
    from .data import load
    from .visibility import get_windows
    cfg = _cfg(args)
    ds = load(cfg)
    w = get_windows(cfg, ds, refresh=args.refresh)
    print(f"{len(w.k0)} 个窗口，{np.unique(w.target).size} 颗目标可观测，外推失败 {int(w.sgp4_failed.sum())} 颗")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="obsplan", description="空间目标观测规划（方案 H）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "validate", "windows"):
        p = sub.add_parser(name)
        p.add_argument("--start", help="规划开始 UTC，如 2026-08-01")
        p.add_argument("--days", type=float)
        p.add_argument("--moon", type=float, help="月光规避角（度），0 为不考虑")
        if name == "plan":
            p.add_argument("--mode", choices=["fast", "quality"], default="fast")
            p.add_argument("--weather", action="store_true", help="按历史晴空概率做期望价值规划（需联网一次）")
            p.add_argument("--validate", action="store_true", help="规划后用 Skyfield 逐秒校验")
            p.add_argument("--no-refine", action="store_true", help="不做按秒精修与按秒补覆盖（任务起止保持在 30 秒网格上）")
            p.add_argument("--out", default="outputs/plan")
        elif name == "validate":
            p.add_argument("plan")
            p.add_argument("--seconds-from-start", action="store_true", help="计划中的时间是相对规划开始的秒数")
            p.add_argument("--out", help="把校验报告写成 JSON")
        else:
            p.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "validate": cmd_validate, "windows": cmd_windows}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
