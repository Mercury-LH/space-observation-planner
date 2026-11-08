"""Same-parent counterfactual labels under a bounded deterministic repair.

Timeouts are retained as censored observations, never zero-valued labels. A
separate child process makes the wall timeout enforceable even if a numerical
call stalls. Feature construction is measured outside the repair budget.
"""
from __future__ import annotations

import multiprocessing as mp
import resource
import sys
from time import perf_counter

from .search import (apply_operation, copy_schedule, generate_operations, metrics,
                     rule_score, schedule_hash)


def repair(pb, initial, *, max_trials=64, deadline=float("inf"), seed=2026):
    best = copy_schedule(initial); before = metrics(pb, best); trials = 0
    for rd in range(max_trials):
        if trials >= max_trials or perf_counter() >= deadline:
            break
        ops, _ = generate_operations(pb, best, round_index=rd, seed=seed, deadline=deadline)
        order = sorted(ops, key=lambda op: (-rule_score(pb, best, op), op.operation_id))
        for op in order:
            if trials >= max_trials or perf_counter() >= deadline:
                break
            trials += 1
            candidate, status = apply_operation(pb, best, op, deadline=deadline)
            if candidate is not None:
                after = metrics(pb, candidate)
                if after["C"] >= before["C"] and after["Q"] > before["Q"] + 1e-12:
                    best, before = candidate, after
                    break
    return best, trials


def _worker(conn, pb, parent, operation, max_trials, timeout_s, seed):
    began = perf_counter()
    try:
        initial = parent
        if operation is not None:
            initial, reason = apply_operation(pb, parent, operation, deadline=began + timeout_s)
            if initial is None:
                conn.send({"status": reason, "censored": reason == "timeout", "metrics": None,
                           "repair_trials": 0, "worker_s": perf_counter() - began})
                return
        final, trials = repair(pb, initial, max_trials=max_trials, deadline=began + timeout_s, seed=seed)
        elapsed = perf_counter() - began
        timed_out = elapsed >= timeout_s
        # ru_maxrss is bytes on Darwin and KiB on Linux; retain explicit unit.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        conn.send({"status": "timeout" if timed_out else "completed", "censored": timed_out,
                   "metrics": None if timed_out else metrics(pb, final),
                   "final_state_hash": schedule_hash(final), "repair_trials": trials,
                   "worker_s": elapsed, "peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024)})
    except BaseException as exc:
        conn.send({"status": "error", "censored": False, "metrics": None,
                   "error": f"{type(exc).__name__}: {exc}", "worker_s": perf_counter() - began})
    finally:
        conn.close()


def bounded_repair(pb, parent, operation=None, *, max_trials=64, timeout_s=2.0, seed=2026):
    if not 0 < timeout_s <= 2.0 or not 0 <= max_trials <= 64:
        raise ValueError("repair exceeds the frozen v1 budget")
    # Never fork the process that already initialized Numba/OpenMP or CUDA.
    # Spawn startup and problem serialization are deliberately inside 2 seconds.
    context = mp.get_context("spawn")
    recv, send = context.Pipe(duplex=False)
    child = context.Process(target=_worker, args=(send, pb, parent, operation, max_trials, timeout_s, seed))
    began = perf_counter()
    child.start(); send.close()
    result = None
    try:
        remaining = max(0, timeout_s - (perf_counter() - began))
        if recv.poll(remaining):
            try:
                result = recv.recv()
            except EOFError:
                result = {"status": "worker_exit", "censored": False, "metrics": None}
        if result is None or perf_counter() - began >= timeout_s:
            result = {"status": "timeout", "censored": True, "metrics": None,
                      "partial_result": result}
    finally:
        recv.close()
        if child.is_alive():
            child.terminate()
        child.join(timeout=0.2)
        if child.is_alive():
            child.kill(); child.join()
    result["wall_s"] = perf_counter() - began
    result["timeout_scope"] = "worker_start_and_repair_including_operation; feature_generation_separate"
    result["start_method"] = "spawn"
    result["exitcode"] = child.exitcode
    return result


def collect_state(pb, parent, feature_context, *, catalog_id, scene_id, split="train",
                  initial_plan_hash, seed=2026, round_index=0, max_candidates=16,
                  max_trials=64, timeout_s=2.0, code_fingerprint="", dependency_fingerprint=""):
    if split != "train":
        raise ValueError("counterfactual training collection may only use train catalogs")
    if not 1 <= max_candidates <= 16:
        raise ValueError("v1 collects at most 16 candidates per parent")
    began = perf_counter()
    operations, generation = generate_operations(pb, parent, round_index=round_index, seed=seed, limit=max_candidates)
    generation_s = perf_counter() - began
    feature_started = perf_counter(); static, dynamic = feature_context.rows(parent, operations)
    feature_s = perf_counter() - feature_started
    if not operations:
        return []
    reference = bounded_repair(pb, parent, max_trials=max_trials, timeout_s=timeout_s, seed=seed)
    before = metrics(pb, parent); rows = []
    for i, operation in enumerate(operations):
        outcome = bounded_repair(pb, parent, operation, max_trials=max_trials, timeout_s=timeout_s, seed=seed)
        valid = reference["status"] == outcome["status"] == "completed"
        status = outcome["status"] if reference["status"] == "completed" else "reference_" + reference["status"]
        row = {"catalog_id": catalog_id, "scene_id": scene_id, "split": split,
               "parent_state_hash": schedule_hash(parent), "initial_plan_hash": initial_plan_hash,
               "operation": operation.to_dict(), "operation_id": operation.operation_id,
               "static_features": static[i].tolist(), "dynamic_features": dynamic[i].tolist(),
               **feature_context.fingerprint, "status": status,
               "censored": bool(reference["censored"] or outcome["censored"]),
               "rule_check_status": outcome["status"], "before": before,
               "after": outcome["metrics"], "reference": reference["metrics"],
               "reference_state_hash": reference.get("final_state_hash"),
               "repair_budget": {"max_trials": max_trials, "timeout_s": timeout_s},
               "reference_evaluation": reference, "candidate_evaluation": outcome,
               "elapsed_s": outcome["wall_s"], "candidate_generation_s": generation_s,
               "dynamic_features_s": feature_s, "seed": seed,
               "code_fingerprint": code_fingerprint, "dependency_fingerprint": dependency_fingerprint,
               "generation": generation, "label_semantics": "measured_delta_vs_same_parent_same_repair_budget_not_optimal_value"}
        for name in ("C", "S", "Q"):
            row["delta_" + name] = outcome["metrics"][name] - reference["metrics"][name] if valid else None
        rows.append(row)
    return rows
