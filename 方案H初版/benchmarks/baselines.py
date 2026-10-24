"""统一评测的简单外部基线；入口 ``benchmarks.baselines:solve``。

mode=manual_greedy 使用固定人工权重，mode=priority 使用固定优先级贪心。
两者均从头构造并执行 finish，不训练 GA、不运行局部搜索或在线伐谋。
预处理和最终独立几何校验遵循公共 worker，求解时间包含构造与精修。
"""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

import numba
import numpy as np

from obsplan.benchmark import check_config
from obsplan.checks import structural_violations
from obsplan.greedy import MANUAL_WEIGHTS, algo_priority, algo_weighted
from obsplan.problem import Problem
from obsplan.refine import finish
from obsplan.schedule import Schedule, evaluate


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    return value.item() if isinstance(value, np.generic) else value


def solve(pb: Problem, spec: dict, progress=None) -> tuple[Schedule, dict]:
    """progress 接收阶段事件字典，与 obsplan.portfolio.solve 的调用方式一致。"""
    started = perf_counter()
    defaults = {"mode": "manual_greedy", "workers": 8, "seed": 2026, "finish_near": True}
    if not isinstance(spec, dict) or set(spec) - defaults.keys():
        raise ValueError("baseline spec 仅支持 mode、workers、seed、finish_near")
    cfg = {**defaults, **spec}
    if cfg["mode"] not in ("manual_greedy", "priority"):
        raise ValueError("mode 必须是 manual_greedy 或 priority")
    for key, minimum in (("workers", 1), ("seed", 0)):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < minimum:
            raise ValueError(f"{key} 必须是 >= {minimum} 的整数")
    if cfg["workers"] > numba.config.NUMBA_NUM_THREADS:
        raise ValueError(f"workers 超过 NUMBA_NUM_THREADS={numba.config.NUMBA_NUM_THREADS}")
    if not isinstance(cfg["finish_near"], bool):
        raise ValueError("finish_near 必须是 bool")
    check_config(pb.cfg)
    timings = {}

    def emit(event, stage, **extra):
        if progress is not None:
            progress({"event": event, "stage": stage,
                      "elapsed_s": perf_counter() - started, **extra})

    @contextmanager
    def stage(name):
        emit("start", name)
        stage_started = perf_counter()
        try:
            yield
        finally:
            duration = perf_counter() - stage_started
            timings[name] = duration
            emit("complete", name, duration_s=duration)

    previous_threads = numba.get_num_threads()
    numba.set_num_threads(cfg["workers"])
    try:
        with stage("construct"):
            if cfg["mode"] == "manual_greedy":
                grid = algo_weighted(pb, MANUAL_WEIGHTS.copy(), "C manual weighted greedy")
            else:
                grid = algo_priority(pb)
        with stage("grid_check"):
            violations = structural_violations(pb, grid)
            if violations:
                raise ValueError(f"基线产生非法网格计划: { {k: len(v) for k, v in violations.items()} }")
            grid_metrics = evaluate(pb.cfg, pb.ds, grid)
        with stage("finish"):
            final, finish_info = finish(pb, grid, near=cfg["finish_near"], verbose=False)
        with stage("final_summary"):
            final_metrics = evaluate(pb.cfg, pb.ds, final)
        final.algo = f"baseline/{cfg['mode']}+finish"
        runtime = perf_counter() - started
        final.runtime_s = runtime
        info = {"actual_config": cfg, "runtime_s": runtime, "stage_timings_s": timings,
                "timing_scope": "solve_including_fresh_construction_and_finish; excludes_common_preprocessing_and_independent_validation",
                "stop_reason": "construction_and_finish_complete",
                "grid_final": grid_metrics, "final": final_metrics, "finish": finish_info,
                "manual_weights": MANUAL_WEIGHTS.tolist() if cfg["mode"] == "manual_greedy" else None,
                "seed_used": False, "randomness": "constructors_do_not_use_random_numbers",
                "famou": {"mode": "unused", "runtime_cloud_calls": 0}}
        return final, _plain(info)
    finally:
        numba.set_num_threads(previous_threads)
