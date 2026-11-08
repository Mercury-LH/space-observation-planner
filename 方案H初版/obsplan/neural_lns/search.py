"""Bounded, transactional neighbourhoods on the public 30-second grid.

Every operation owns the hash of its parent, identifies original task rows, and
is applied to a fresh Schedule. A rejected/interrupted trial never mutates its
parent. Public structural checks inspect the ENTIRE schedule, including other
stations and observations outside the local four-hour region.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from time import perf_counter
from typing import Callable

import numpy as np

from ..checks import structural_violations
from ..schedule import Schedule, evaluate, task_score


def schedule_hash(schedule: Schedule) -> str:
    digest = hashlib.sha256()
    for name in ("station", "target", "start_s", "dur_s"):
        a = np.ascontiguousarray(getattr(schedule, name), dtype="<i8")
        digest.update(name.encode()); digest.update(a.shape[0].to_bytes(8, "little")); digest.update(a.tobytes())
    return digest.hexdigest()


def copy_schedule(schedule: Schedule) -> Schedule:
    return Schedule(*(np.asarray(getattr(schedule, n)).copy() for n in
                      ("station", "target", "start_s", "dur_s")),
                    algo=schedule.algo, runtime_s=schedule.runtime_s, info=dict(schedule.info))


def metrics(pb, schedule):
    ev = evaluate(pb.cfg, pb.ds, schedule)
    return {"C": int(ev["covered"]), "S": float(ev["score"]), "Q": float(ev["combined"])}


@dataclass(frozen=True)
class Task:
    station: int
    target: int
    start_s: int
    dur_s: int
    window: int


@dataclass(frozen=True)
class Operation:
    kind: str
    parent_state_hash: str
    remove_indices: tuple[int, ...]
    additions: tuple[Task, ...]  # insertion FIRST, then replacement(s)
    region_station: int
    region_start_s: int
    region_end_s: int
    related_targets: tuple[int, ...]
    source: str = "fixed"

    @property
    def operation_id(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def to_dict(self):
        return {**asdict(self), "operation_id": self.operation_id}


def check_schedule(pb, schedule):
    arrays = [np.asarray(getattr(schedule, n)) for n in ("station", "target", "start_s", "dur_s")]
    if any(a.ndim != 1 or len(a) != len(arrays[0]) for a in arrays):
        return {"array_shape": [0]}
    if any(not np.issubdtype(a.dtype, np.integer) for a in arrays):
        return {"integer_grid_required": [0]}
    if np.any((arrays[0] < 0) | (arrays[0] >= pb.n_stations)) or np.any((arrays[1] < 0) | (arrays[1] >= pb.n_targets)):
        return {"invalid_index": [0]}
    return structural_violations(pb, schedule)


def apply_operation(pb, parent, operation, *, deadline=float("inf"), clock=perf_counter):
    """Return (candidate or None, reason); parent is never written."""
    if clock() >= deadline:
        return None, "timeout"
    if operation.parent_state_hash != schedule_hash(parent):
        return None, "stale_parent"
    removed = operation.remove_indices
    if len(removed) > 3 or len(set(removed)) != len(removed) or any(i < 0 or i >= len(parent) for i in removed):
        return None, "invalid_removed_indices"
    if not operation.additions or len(operation.related_targets) > 64:
        return None, "invalid_neighbourhood"
    if (operation.kind not in ("insert", "migrate", "trim", "replace")
            or len(operation.additions) > 1 + len(removed)
            or (operation.kind == "insert" and (removed or len(operation.additions) != 1))
            or (operation.kind in ("migrate", "trim") and
                (not removed or len(operation.additions) != 1 + len(removed)
                 or sorted(t.target for t in operation.additions[1:]) != sorted(int(parent.target[i]) for i in removed)))):
        return None, "invalid_operation_shape"
    # Only tasks originally fully inside the selected region can be changed.
    for i in removed:
        if (parent.station[i] != operation.region_station or parent.start_s[i] < operation.region_start_s
                or parent.start_s[i] + parent.dur_s[i] > operation.region_end_s):
            return None, "outside_region"
    related = set(operation.related_targets)
    if any(int(parent.target[i]) not in related for i in removed):
        return None, "outside_related_targets"
    for t in operation.additions:
        if t.target not in related or not 0 <= t.window < len(pb.w_st):
            return None, "invalid_task_window_mapping"
        if (t.station != int(pb.w_st[t.window]) or t.target != int(pb.w_tg[t.window])
                or t.start_s < int(pb.w_k0[t.window]) * pb.cfg.cell_s
                or t.start_s + t.dur_s > int(pb.w_k1[t.window]) * pb.cfg.cell_s):
            return None, "invalid_task_window_mapping"
    keep = np.ones(len(parent), dtype=bool); keep[list(removed)] = False
    out = Schedule(*(np.concatenate([getattr(parent, name)[keep], np.asarray([getattr(t, name) for t in operation.additions], dtype=np.int64)])
                     for name in ("station", "target", "start_s", "dur_s")), algo="neural-lns/grid")
    bad = check_schedule(pb, out)
    if clock() >= deadline:
        return None, "timeout"
    return (None, "constraint:" + ",".join(sorted(bad))) if bad else (out, "completed")


def _task(pb, window, start_s, dur_s):
    return Task(int(pb.w_st[window]), int(pb.w_tg[window]), int(start_s), int(dur_s), int(window))


def _window_for_task(pb, schedule, row):
    t, s = int(schedule.target[row]), int(schedule.station[row])
    windows = pb.t_win[pb.t_ptr[t]:pb.t_ptr[t + 1]]
    hit = windows[(pb.w_st[windows] == s) & (pb.w_k0[windows] * pb.cfg.cell_s <= schedule.start_s[row])
                  & (pb.w_k1[windows] * pb.cfg.cell_s >= schedule.start_s[row] + schedule.dur_s[row])]
    return int(hit[0]) if len(hit) else -1


def _conflicts(pb, schedule, task, ignored=()):
    end = schedule.start_s + schedule.dur_s
    sw = pb.cfg.switch_s
    idx = np.nonzero((schedule.station == task.station) & (schedule.start_s < task.start_s + task.dur_s + sw)
                     & (end + sw > task.start_s))[0]
    return tuple(int(i) for i in idx if int(i) not in ignored)


def _starts(pb, schedule, w, duration, lo=None, hi=None, limit=8, offset=0):
    cell = pb.cfg.cell_s
    a = int(pb.w_k0[w]) * cell if lo is None else max(int(pb.w_k0[w]) * cell, lo)
    b = int(pb.w_k1[w]) * cell - duration if hi is None else min(int(pb.w_k1[w]) * cell - duration, hi - duration)
    if b < a:
        return []
    s, t = int(pb.w_st[w]), int(pb.w_tg[w])
    starts = [a, b, ((a + b) // (2 * cell)) * cell]
    # Resource and revisit boundaries are real search opportunities, not labels.
    nearby = np.nonzero((schedule.station == s) & (schedule.start_s >= a - pb.cfg.obs_max_s)
                        & (schedule.start_s <= b + pb.cfg.obs_max_s))[0]
    starts.extend(int(schedule.start_s[i] + schedule.dur_s[i] + pb.cfg.switch_s) for i in nearby)
    same = np.nonzero(schedule.target == t)[0]
    gap = int(pb.cfg.repeat_gap_h * 3600)
    starts.extend(int(schedule.start_s[i] + schedule.dur_s[i] + gap) for i in same)
    starts.extend(int(schedule.start_s[i] - gap - duration) for i in same)
    unique = list(dict.fromkeys(x for x in starts if a <= x <= b and x % cell == 0))
    if unique:
        rotate = offset % len(unique)
        unique = unique[rotate:] + unique[:rotate]
    return unique[:limit]


def generate_operations(pb, parent, *, round_index=0, seed=2026, limit=32, max_targets=64,
                        block_s=14400, deadline=float("inf"), clock=perf_counter):
    """Deterministic mixed proposals, with raw/rejected/deferred provenance.

    Ranking never changes this candidate generator. Region and target traversal
    rotate between rounds; model scores never remove proposals from the pool.
    """
    if not 1 <= limit <= 32 or not 1 <= max_targets <= 64:
        raise ValueError("candidate/target limits exceed frozen v1 bounds")
    if check_schedule(pb, parent):
        raise ValueError("local search requires a valid grid schedule")
    nblocks = max(1, int(np.ceil(pb.cfg.days * 86400 / block_s)))
    region = round_index % (pb.n_stations * nblocks)
    station, block = region % pb.n_stations, region // pb.n_stations
    lo, hi = block * block_s, min((block + 1) * block_s, int(pb.cfg.days * 86400))
    rng = np.random.default_rng(seed + round_index)
    w = np.nonzero((pb.w_st == station) & (pb.w_k1 * pb.cfg.cell_s >= lo + pb.cfg.obs_min_s)
                   & (pb.w_k0 * pb.cfg.cell_s <= hi - pb.cfg.obs_min_s))[0]
    targets = np.unique(pb.w_tg[w])
    counts = np.bincount(parent.target, minlength=pb.n_targets)
    nw = np.diff(pb.t_ptr)
    fixed = sorted(targets.tolist(), key=lambda t: (counts[t] > 0, -pb.P[t], t))
    scarce = sorted(targets.tolist(), key=lambda t: (nw[t], -pb.P[t], t))
    random = rng.permutation(targets).tolist()
    streams = [fixed, scarce, random]
    selected, sources = [], {}
    offset = round_index // (pb.n_stations * nblocks)
    streams = [stream[offset % len(stream):] + stream[:offset % len(stream)] if stream else [] for stream in streams]
    cursors = [0, 0, 0]
    for _ in range(max(len(targets), 1)):
        for j, (source, stream) in enumerate(zip(("fixed", "scarcity", "random"), streams)):
            while cursors[j] < len(stream) and int(stream[cursors[j]]) in sources:
                cursors[j] += 1
            if cursors[j] < len(stream) and len(selected) < max(1, max_targets - 3):
                t = int(stream[cursors[j]]); cursors[j] += 1
                selected.append(t); sources[t] = source
    initial_hash = schedule_hash(parent)
    operations, records, seen = [], [], set()

    def add(kind, removed, additions, source):
        related = tuple(sorted(set(selected) | {int(parent.target[i]) for i in removed}))
        op = Operation(kind, initial_hash, tuple(removed), tuple(additions), station, lo, hi, related, source)
        if op.operation_id in seen:
            return
        seen.add(op.operation_id)
        # Generation does not score candidate outcomes. Legality is checked at
        # execution so every ranking receives exactly the same proposals.
        reason = "queued" if len(related) <= max_targets else "related_target_cap"
        if len(operations) - target_start >= per_target_limit:
            reason = "deferred_per_target_cap"
        if len(operations) >= limit:
            reason = "deferred_candidate_cap"
        records.append({"operation": op.to_dict(), "reason": reason})
        if reason == "queued":
            operations.append(op)

    per_target_limit = max(2, int(np.ceil(limit / max(1, len(selected)))))
    visited_targets, visited_windows = 0, 0
    for target in selected:
        if clock() >= deadline or len(operations) >= limit:
            break
        visited_targets += 1
        target_start = len(operations)
        windows = w[pb.w_tg[w] == target]
        if len(windows):
            windows = np.roll(windows, -offset % len(windows))
        for window in windows[:4]:
            if clock() >= deadline or len(operations) >= limit or len(operations) - target_start >= per_target_limit:
                break
            visited_windows += 1
            for duration in (pb.cfg.obs_min_s, pb.cfg.obs_max_s):
                for start in _starts(pb, parent, int(window), duration, lo, hi, offset=offset):
                    if clock() >= deadline or len(operations) >= limit or len(operations) - target_start >= per_target_limit:
                        break
                    new = _task(pb, window, start, duration)
                    blockers = _conflicts(pb, parent, new)
                    if not blockers:
                        add("insert", (), [new], sources[target]); continue
                    if len(blockers) > 3:
                        records.append({"target": target, "window": int(window), "start_s": start,
                                        "reason": "more_than_three_blockers"}); continue
                    if any(parent.start_s[i] < lo or parent.start_s[i] + parent.dur_s[i] > hi for i in blockers):
                        records.append({"target": target, "window": int(window), "start_s": start,
                                        "reason": "blocker_outside_region"}); continue
                    if all(counts[parent.target[i]] > 1 for i in blockers):
                        add("replace", blockers, [new], sources[target])
                    trimmed = []
                    for i in blockers:
                        bw = _window_for_task(pb, parent, i)
                        options = [(int(parent.start_s[i]), min(int(parent.dur_s[i]), start - pb.cfg.switch_s - int(parent.start_s[i]))),
                                   (start + duration + pb.cfg.switch_s, int(parent.start_s[i] + parent.dur_s[i]) - start - duration - pb.cfg.switch_s)]
                        options = [(a, min(d, int(parent.dur_s[i]))) for a, d in options if d >= pb.cfg.obs_min_s]
                        if bw < 0 or not options:
                            break
                        a, d = max(options, key=lambda x: x[1]); trimmed.append(_task(pb, bw, a, d))
                    if len(trimmed) == len(blockers):
                        add("trim", blockers, [new] + trimmed, sources[target])
                    # Relocate a single blocker, possibly to another station.
                    if len(blockers) == 1:
                        i = blockers[0]; bt = int(parent.target[i])
                        alternatives = pb.t_win[pb.t_ptr[bt]:pb.t_ptr[bt + 1]]
                        if len(alternatives):
                            alternatives = np.roll(alternatives, -offset % len(alternatives))
                        for alt in alternatives[:16]:
                            if clock() >= deadline or len(operations) >= limit or len(operations) - target_start >= per_target_limit:
                                break
                            for a in _starts(pb, parent, int(alt), int(parent.dur_s[i]), limit=3, offset=offset):
                                moved = _task(pb, alt, a, int(parent.dur_s[i]))
                                if not _conflicts(pb, parent, moved, blockers):
                                    add("migrate", blockers, [new, moved], sources[target]); break
    return operations, {"round_index": round_index, "region": [station, lo, hi],
                        "related_targets": selected, "raw_candidates": records,
                        "candidate_ids": [o.operation_id for o in operations],
                        "enumeration": {"complete": False, "region_targets": len(targets),
                            "selected_targets": len(selected), "visited_targets": visited_targets,
                            "unvisited_selected_targets": len(selected) - visited_targets,
                            "region_windows": len(w), "visited_windows": visited_windows,
                            "per_target_candidate_limit": per_target_limit, "windows_per_target_limit": 4,
                            "alternative_windows_limit": 16, "starts_per_window_limit": 8,
                            "region_visit_offset": offset,
                            "deferred_policy": "cycle station/block; on revisit rotate target/window/start/alternative lists by visit index; no completeness guarantee under finite budget",
                            "target_sources": sources},
                        "timed_out": clock() >= deadline}


def rule_score(pb, parent, operation):
    before = task_score(pb.cfg, pb.P[parent.target[list(operation.remove_indices)]], parent.dur_s[list(operation.remove_indices)]).sum()
    after = sum(float(task_score(pb.cfg, np.asarray([pb.P[t.target]]), np.asarray([t.dur_s]))[0]) for t in operation.additions)
    covered = set(parent.target.tolist())
    gain_c = sum(t not in covered for t in set(a.target for a in operation.additions))
    return pb.vc * gain_c + pb.vs * (after - before)


def search(pb, initial, *, deadline, mode="rule", seed=2026, max_rounds=256,
           ranker=None, feature_context=None, clock=perf_counter, on_state: Callable | None = None):
    if mode not in ("rule", "random", "neural", "static_only"):
        raise ValueError("unknown ordering mode")
    if check_schedule(pb, initial):
        raise ValueError("initial plan failed public structural checks")
    best = copy_schedule(initial); best_metrics = metrics(pb, best)
    trace = []; timings = {"candidate_generation": 0., "features": 0., "inference": 0., "candidate_execution": 0.}
    for rd in range(max_rounds):
        if clock() >= deadline:
            break
        t0 = clock(); ops, generation = generate_operations(pb, best, round_index=rd, seed=seed, deadline=deadline, clock=clock)
        timings["candidate_generation"] += clock() - t0
        if not ops:
            trace.append({"generation": generation, "evaluations": []}); continue
        if mode == "rule":
            order = sorted(range(len(ops)), key=lambda i: (-rule_score(pb, best, ops[i]), i))
        elif mode == "random":
            order = np.random.default_rng(seed + rd).permutation(len(ops)).tolist()
        else:
            if ranker is None or feature_context is None:
                raise ValueError("neural ordering requires ranker and feature context")
            t0 = clock(); static, dynamic = feature_context.rows(best, ops); timings["features"] += clock() - t0
            if clock() >= deadline:
                trace.append({"generation": generation, "reason": "feature_timeout", "evaluations": []}); break
            t0 = clock(); values = ranker.score(static, None if mode == "static_only" else dynamic); timings["inference"] += clock() - t0
            if len(values) != len(ops) or not np.isfinite(values).all():
                raise ValueError("ranker returned invalid candidate scores")
            # Every fourth round is a deterministic rule fallback; no hard prune.
            order = sorted(range(len(ops)), key=lambda i: (-float(values[i]), i)) if rd % 4 != 3 else sorted(range(len(ops)), key=lambda i: (-rule_score(pb, best, ops[i]), i))
        evaluations = []; accepted = None
        for idx in order:
            t0 = clock(); candidate, reason = apply_operation(pb, best, ops[idx], deadline=deadline, clock=clock)
            timings["candidate_execution"] += clock() - t0
            row = {"operation_id": ops[idx].operation_id, "status": reason}
            if candidate is not None:
                after = metrics(pb, candidate); row["after"] = after
                if clock() < deadline and after["C"] >= best_metrics["C"] and after["Q"] > best_metrics["Q"] + 1e-12:
                    best, best_metrics = candidate, after; row["accepted"] = True; accepted = ops[idx].operation_id
            evaluations.append(row)
            if accepted or clock() >= deadline:
                break
        trace.append({"generation": generation, "order": [ops[i].operation_id for i in order], "evaluations": evaluations})
        if accepted and on_state is not None:
            on_state(copy_schedule(best), rd)
    return best, {"mode": mode, "seed": seed, "initial": metrics(pb, initial), "final": best_metrics,
                  "initial_plan_hash": schedule_hash(initial), "final_plan_hash": schedule_hash(best),
                  "stage_timings_s": timings, "rounds": trace, "deadline_reached": clock() >= deadline,
                  "independent_geometry_checked": False}
