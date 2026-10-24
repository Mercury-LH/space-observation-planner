"""从头生成来源计划、融合和按秒精修的统一求解入口。

``solve(pb, {"preset": "hplus_e", "workers": 8, "seed": 2026})``。
spec 也可用 PRESETS 中的完整字典；嵌套覆盖只改指定字段。所有计时包含
来源生成，绝不读取已有计划 CSV。窗口构建和最终独立几何校验由评测器计时。
融合预算是旧求解器在块边界检查的软预算，包含建模的实测耗时另行记录。
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from math import isfinite
from time import perf_counter

import numba
import numpy as np

from . import cpsat, fusion_merge as fm, ga, lagrange
from .checks import structural_violations
from .evolved import load_rule
from .greedy import algo_weighted
from .lns import state_from_schedule
from .local_search import improve
from .problem import Problem
from .refine import finish
from .schedule import Schedule, evaluate


_BASE = {
    "sources": ["G+", "E", "H+"],
    "h_iters": 300,
    "h_fast": False,
    "workers": 8,
    "seed": 2026,
    # 旧 fusion_merge.py 的默认 seed 是 0；优化配置默认跟随 seed。
    "fusion_seed": 0,
    "ga": {"n_scen": 3, "pop": 16, "gens": 12},
    "g": {"block_h": 4.0, "commit_h": 3.0, "time_limit_s": 10.0,
          "k_win": 1, "k_dur": 1},
    "fusion": {"rounds": 3, "durations": [4, 5, 6, 7, 8], "shifts": [-1, 1],
               "block_h": 24.0, "step_h": 12.0, "time_limit_s": 30.0,
               "budget_s": 300.0, "passes": 1, "local_search": True},
    # gain 是 combined 的绝对增量，gain_per_s 是每秒 combined 增量。
    # patience=0 禁用早停，基准必须完整执行三轮。
    "early_stop": {"patience": 0, "min_gain": 0.0, "min_gain_per_s": 0.0},
    "finish_near": True,
}


def _merge(base: dict, changes: dict) -> dict:
    out = deepcopy(base)
    for key, value in changes.items():
        if key not in out:
            raise ValueError(f"未知 portfolio 参数: {key}")
        if isinstance(out[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{key} 必须是字典")
            out[key] = _merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _preset(name: str, **changes) -> dict:
    return {"preset": name, **_merge(_BASE, changes)}


PRESETS: dict[str, dict] = {
    "fusion_baseline": _preset("fusion_baseline"),
    "hplus_fast": _preset("hplus_fast", sources=["H+"], h_iters=150, h_fast=True,
                          fusion_seed=None, fusion={"rounds": 0}),
    "hplus_self": _preset("hplus_self", sources=["H+"], h_iters=150, h_fast=True,
                          fusion_seed=None,
                          fusion={"rounds": 2, "time_limit_s": 8.0, "budget_s": 60.0},
                          early_stop={"patience": 1, "min_gain": 0.00005}),
    "hplus_e": _preset("hplus_e", sources=["E", "H+"], h_iters=150, h_fast=True,
                       fusion_seed=None,
                       fusion={"rounds": 2, "time_limit_s": 10.0, "budget_s": 90.0},
                       early_stop={"patience": 1, "min_gain": 0.00005}),
    "fusion_short_g": _preset("fusion_short_g", h_iters=150, h_fast=True,
                              fusion_seed=None, g={"time_limit_s": 4.0},
                              fusion={"rounds": 2, "time_limit_s": 10.0, "budget_s": 90.0},
                              early_stop={"patience": 1, "min_gain": 0.00005}),
}


def _config(spec: dict, pb: Problem) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("spec 必须是字典")
    name = spec.get("preset", "hplus_fast")
    if name not in PRESETS:
        raise ValueError(f"未知 preset: {name}; 可用: {', '.join(PRESETS)}")
    cfg = _merge(PRESETS[name], spec)
    if cfg["fusion_seed"] is None:
        cfg["fusion_seed"] = cfg["seed"]

    def integer(value, field, minimum=0):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{field} 必须是 >= {minimum} 的整数")

    def positive(value, field, allow_zero=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"{field} 必须是有限数值")
        if value < 0 or (value == 0 and not allow_zero):
            raise ValueError(f"{field} 必须{'非负' if allow_zero else '为正'}")

    for key in ("h_iters", "workers"):
        integer(cfg[key], key, 1)
    for key in ("seed", "fusion_seed"):
        integer(cfg[key], key)
        if cfg[key] > 2**31 - 1:
            raise ValueError(f"{key} 超出 CP-SAT 的 int32 种子范围")
    if cfg["workers"] > numba.config.NUMBA_NUM_THREADS:
        raise ValueError(f"workers 超过 NUMBA_NUM_THREADS={numba.config.NUMBA_NUM_THREADS}")
    for key in ("h_fast", "finish_near"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{key} 必须是 bool")
    sources = cfg["sources"]
    if not isinstance(sources, (list, tuple)) or not sources:
        raise ValueError("sources 必须是非空列表")
    if any(s not in ("H+", "E", "G+") for s in sources) or len(set(sources)) != len(sources):
        raise ValueError("sources 仅支持不重复的 H+、E、G+")
    for key in ("n_scen", "gens"):
        integer(cfg["ga"][key], f"ga.{key}", 1)
    integer(cfg["ga"]["pop"], "ga.pop", 3)
    for key in ("block_h", "commit_h", "time_limit_s"):
        positive(cfg["g"][key], f"g.{key}")
    if cfg["g"]["commit_h"] > cfg["g"]["block_h"]:
        raise ValueError("g.commit_h 不得超过 g.block_h")
    for key in ("k_win", "k_dur"):
        integer(cfg["g"][key], f"g.{key}", 1)
    f = cfg["fusion"]
    integer(f["rounds"], "fusion.rounds")
    integer(f["passes"], "fusion.passes", 1)
    for key in ("block_h", "step_h", "time_limit_s", "budget_s"):
        positive(f[key], f"fusion.{key}")
    if f["step_h"] > f["block_h"]:
        raise ValueError("fusion.step_h 不得超过 fusion.block_h")
    if not isinstance(f["local_search"], bool):
        raise ValueError("fusion.local_search 必须是 bool")
    for key in ("durations", "shifts"):
        if not isinstance(f[key], (list, tuple)):
            raise ValueError(f"fusion.{key} 必须是列表")
        for value in f[key]:
            integer(value, f"fusion.{key}", -2**31)
    if any(d < pb.m_min or d > pb.m_max for d in f["durations"]):
        raise ValueError("fusion.durations 必须在观测时长允许范围内（单位为格）")
    for value in (cfg["g"]["commit_h"], f["step_h"]):
        if value * 3600 < pb.cfg.cell_s:
            raise ValueError("滚动步长不得小于一个时间格")
    e = cfg["early_stop"]
    integer(e["patience"], "early_stop.patience")
    positive(e["min_gain"], "early_stop.min_gain", allow_zero=True)
    positive(e["min_gain_per_s"], "early_stop.min_gain_per_s", allow_zero=True)
    # 上游 State/CP-SAT 把切换占用写成一个格，融合目标默认等权。
    if (pb.cfg.cell_s != 30 or pb.cfg.switch_s != 30
            or (pb.cfg.obs_min_s, pb.cfg.obs_max_s) != (120, 240)
            or (pb.m_min, pb.m_max) != (4, 8)):
        raise ValueError("当前 portfolio 仅支持 30 秒格、30 秒切换、120–240 秒观测")
    if pb.cfg.w_cov != 0.5 or pb.cfg.w_score != 0.5 or pb.cfg.score_ref != "sumP":
        raise ValueError("当前 portfolio 仅支持覆盖/累计得分各 0.5、sumP 归一化；时间权重应由外部评测器计算")
    return cfg


def _plain(value):
    """仅对小型诊断数据调用；不保存 H 的整张价格矩阵。"""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    return value.item() if isinstance(value, np.generic) else value


def _metrics(pb, sch):
    ev = evaluate(pb.cfg, pb.ds, sch)
    return {k: ev[k] for k in ("covered", "score", "combined", "tasks")}


def _check_grid(pb, sch, label):
    violations = structural_violations(pb, sch)
    if violations:
        raise ValueError(f"{label} 产生非法网格计划: { {k: len(v) for k, v in violations.items()} }")


def _origin(pb, best, plans):
    """任意来源组合都适用，不假定 H/G/E 某个来源一定存在。"""
    keysets = {name: set(zip(*fm._cells(pb, sch))) for name, sch in plans.items()}
    keys = list(zip(*fm._cells(pb, best)))
    return {"tasks": len(keys),
            "same_as": {n: sum(k in ks for k in keys) for n, ks in keysets.items()},
            "exclusive": {n: sum(k in ks and not any(k in other for m, other in keysets.items() if m != n)
                                 for k in keys) for n, ks in keysets.items()},
            "new_or_variant": sum(not any(k in ks for ks in keysets.values()) for k in keys)}


def solve(pb: Problem, spec: dict, progress=None) -> tuple[Schedule, dict]:
    """统一求解接口；progress 接收 {event, stage, elapsed_s, ...} 事件字典。

    ``seed`` 控制 GA；``fusion_seed=None`` 时融合也跟随它。旧 G 不暴露
    random_seed，因此保留上游默认并显式记录。多线程 CP-SAT 和墙钟限时
    仍可能产生运行间差异，固定 seed 不代表逐位确定性。没有在线伐谋调用。
    """
    started = perf_counter()
    cfg = _config(spec, pb)
    timings, details, plans, rounds = {}, {}, {}, []

    def emit(event, stage, **extra):
        if progress is not None:
            progress({"event": event, "stage": stage,
                      "elapsed_s": perf_counter() - started, **extra})

    @contextmanager
    def stage(name):
        emit("start", name)
        t0 = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - t0
            timings[name] = timings.get(name, 0.0) + elapsed
            emit("complete", name, duration_s=elapsed)

    previous_threads = numba.get_num_threads()
    numba.set_num_threads(cfg["workers"])
    try:
        # E 和 G+ 共享一次训练及未经局部搜索的 D 提示。
        if any(s in cfg["sources"] for s in ("E", "G+")):
            with stage("ga_train"):
                weights, details["ga"] = ga.train(pb, **cfg["ga"], seed=cfg["seed"], verbose=False)
                details["ga"]["weights"] = np.asarray(weights).tolist()
            with stage("e_construct"):
                hint, state = algo_weighted(pb, weights, "D learned greedy", return_state=True)
                details["e_construct"] = _metrics(pb, hint)
            if "E" in cfg["sources"]:
                with stage("e_local_search"):
                    details["e_local_search"] = improve(pb, state)
                    plans["E"] = state.schedule(pb, "E", 0.0)
                    details["e_after_local_search"] = _metrics(pb, plans["E"])
            if "G+" in cfg["sources"]:
                g = cfg["g"]
                with stage("g_construct"):
                    raw = cpsat.solve(pb, weights, block_h=g["block_h"], commit_h=g["commit_h"],
                                      k_win=g["k_win"], k_dur=g["k_dur"], time_limit=g["time_limit_s"],
                                      workers=cfg["workers"], hint=hint, verbose=False)
                    details["g_construct"] = _metrics(pb, raw)
                with stage("g_local_search"):
                    state = cpsat.to_state(pb, raw)
                    details["g_local_search"] = improve(pb, state)
                    plans["G+"] = state.schedule(pb, "G+", 0.0)
                    details["g_after_local_search"] = _metrics(pb, plans["G+"])
        if "H+" in cfg["sources"]:
            with stage("hplus"):
                rule = load_rule()
                if rule is None:
                    raise RuntimeError("缺少已采纳的本地伐谋 recover_rule，不能把人工 H 标成 H+")
                plans["H+"], hinfo = lagrange.solve(
                    pb, iters=cfg["h_iters"], fast=cfg["h_fast"], rule=rule,
                    verbose=False, name="H+",
                    progress=lambda fraction, message: emit("progress", "hplus", fraction=fraction, message=message))
                details["hplus"] = {k: v for k, v in hinfo.items() if k != "mu"}
                details["hplus"]["iterations_completed"] = len(hinfo.get("history", []))

        # 来源顺序影响候选池的并列解，显式按配置排列。
        plans = {name: plans[name] for name in cfg["sources"]}
        with stage("source_checks"):
            for name, sch in plans.items():
                _check_grid(pb, sch, name)
            source_scores = {name: _metrics(pb, sch) for name, sch in plans.items()}
        hint_name = max(plans, key=lambda name: source_scores[name]["combined"])
        best, best_ev = plans[hint_name], source_scores[hint_name]
        f, early = cfg["fusion"], cfg["early_stop"]
        stop_reason = "no_fusion_requested" if f["rounds"] == 0 else "round_limit"
        stalled = 0
        for rd in range(f["rounds"]):
            round_started, before = perf_counter(), best_ev["combined"]
            prefix = f"fusion_{rd + 1}"
            with stage(f"{prefix}_pool"):
                sources = list(plans.values()) + ([best] if rd else [])
                pool = fm.build_pool(pb, sources, durations=tuple(f["durations"]), shifts=tuple(f["shifts"]))
                if len(pool) == 0:
                    stop_reason = "empty_candidate_pool"
                    break
                cur = fm.select_of(pool, fm._cells(pb, best))
            with stage(f"{prefix}_merge"):
                cur, logs = fm.merge_blocks(
                    pb, pool, cur, block_h=f["block_h"], step_h=f["step_h"], time_limit=f["time_limit_s"],
                    workers=cfg["workers"], seed=cfg["fusion_seed"], budget_s=f["budget_s"],
                    passes=f["passes"], verbose=False)
                merged = fm.to_schedule(pb, pool, cur, prefix)
                _check_grid(pb, merged, prefix)
                merged_ev = _metrics(pb, merged)
            candidates = [(merged, merged_ev)]
            if f["local_search"]:
                with stage(f"{prefix}_local_search"):
                    state = state_from_schedule(pb, merged)
                    ls = improve(pb, state)
                    polished = state.schedule(pb, f"{prefix}+LS", 0.0)
                    _check_grid(pb, polished, f"{prefix}+LS")
                    candidates.append((polished, _metrics(pb, polished)))
            for candidate, ev in candidates:
                if ev["combined"] > best_ev["combined"]:
                    best, best_ev = candidate, ev
            duration = perf_counter() - round_started
            gain = best_ev["combined"] - before
            row = {"round": rd + 1, "pool_size": len(pool), "before_combined": before,
                   "merged": merged_ev, "best": best_ev, "gain": gain,
                   "duration_s": duration, "gain_per_s": gain / max(duration, 1e-12), "blocks": logs}
            if f["local_search"]:
                row.update(after_local_search=candidates[-1][1], local_search=ls)
            rounds.append(row)
            emit("round_complete", prefix, gain=gain, combined=best_ev["combined"])
            low_gain = gain <= early["min_gain"] or row["gain_per_s"] <= early["min_gain_per_s"]
            stalled = stalled + 1 if low_gain else 0
            if early["patience"] and stalled >= early["patience"]:
                stop_reason = "low_marginal_gain"
                break

        with stage("grid_summary"):
            origin = _origin(pb, best, plans)
        with stage("finish"):
            final, finish_info = finish(pb, best, near=cfg["finish_near"], verbose=False)
        with stage("final_summary"):
            final_ev = _metrics(pb, final)
        final.algo = f"portfolio/{cfg['preset']}"
        runtime = perf_counter() - started
        final.runtime_s = runtime
        info = {"actual_config": cfg, "runtime_s": runtime, "stage_timings_s": timings,
                "timing_scope": "solve_including_all_source_generation; excludes_problem_build_and_independent_validation",
                "budget_semantics": "soft_fusion_budget_checked_between_blocks; finish_always_runs",
                "stop_reason": stop_reason, "sources": source_scores, "initial_best_source": hint_name,
                "rounds": rounds, "grid_final": best_ev, "grid_final_origin": origin,
                "final": final_ev, "finish": finish_info, "details": details,
                "famou": {"mode": "local_adopted_rule" if "H+" in plans else "unused",
                          "runtime_cloud_calls": 0},
                "g_random_seed": int(cpsat.cp_model.CpSolver().parameters.random_seed) if "G+" in plans else None,
                "g_seed_control": "upstream_default_not_overridden" if "G+" in plans else "unused",
                "cp_sat_deterministic": False}
        return final, _plain(info)
    finally:
        numba.set_num_threads(previous_threads)
