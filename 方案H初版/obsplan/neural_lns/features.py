"""One feature implementation shared by collection and inference."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from ..famou_h_eval import FEATURES, feature_rows
from ..schedule import task_score

FEATURE_VERSION = "neural-lns-features-v1"
POOL_VERSION = "full-windows-v1"
PRICE_VERSION = "hplus-fixed-mu-v1"
DYNAMIC_FEATURE_NAMES = [
    "covered", "observation_count", "previous_exists", "previous_gap_margin_s",
    "next_exists", "next_gap_margin_s", "station_occupancy_fraction",
    "free_gap_before_s", "free_gap_after_s", "removed_tasks", "removed_score",
    "replacement_tasks", "replacement_score", "alternative_stations", "alternative_windows",
    "op_insert", "op_migrate", "op_trim", "op_replace", "local_targets", "local_task_count",
    "insertion_start_fraction", "insertion_duration_s", "covered_fraction",
]


def arrays_hash(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        arr = np.ascontiguousarray(value)
        h.update(str(arr.dtype).encode()); h.update(json.dumps(arr.shape).encode()); h.update(arr.tobytes())
    return h.hexdigest()


class FeatureContext:
    def __init__(self, pb, mu):
        self.pb = pb
        self.mu = np.asarray(mu, dtype=np.float64).copy()
        if (self.mu.shape not in ((pb.n_stations, pb.n_samples), (pb.n_stations, pb.n_samples + 2))
                or not np.isfinite(self.mu).all()):
            raise ValueError("price matrix must have finite station x sample entries (public H+ uses two padding cells)")
        self.pool_sha256 = arrays_hash(pb.w_st, pb.w_tg, pb.w_k0, pb.w_k1, pb.P,
                                       np.asarray([pb.cfg.cell_s, pb.n_samples, pb.gap]))
        self.price_sha256 = arrays_hash(self.mu)
        self._static = {}

    @property
    def fingerprint(self):
        return {"feature_version": FEATURE_VERSION, "pool_version": POOL_VERSION,
                "price_version": PRICE_VERSION, "pool_sha256": self.pool_sha256,
                "price_sha256": self.price_sha256, "static_feature_names": list(FEATURES),
                "dynamic_feature_names": list(DYNAMIC_FEATURE_NAMES), "price_shape": list(self.mu.shape)}

    def rows(self, parent, operations):
        pb = self.pb
        keys = [(op.additions[0].window, op.additions[0].dur_s // pb.cfg.cell_s) for op in operations]
        missing = list(dict.fromkeys(k for k in keys if k not in self._static))
        if missing:
            cw, cm = (np.asarray(x, np.int64) for x in zip(*missing))
            # Crucially: FULL pool statistics and the same fixed H+ prices at
            # both collection and inference, never released 12-window features.
            rows = feature_rows(pb, self.mu, cw, cm).astype(np.float32)
            self._static.update(zip(missing, rows))
        static = np.asarray([self._static[k] for k in keys], np.float32).reshape(-1, 20)
        dynamic = np.asarray([dynamic_row(pb, parent, op) for op in operations], np.float32).reshape(-1, 24)
        if not np.isfinite(static).all() or not np.isfinite(dynamic).all():
            raise ValueError("non-finite operation features")
        return static, dynamic


def dynamic_row(pb, parent, operation):
    task = operation.additions[0]
    same = np.nonzero(parent.target == task.target)[0]
    previous = same[parent.start_s[same] <= task.start_s]
    following = same[parent.start_s[same] > task.start_s]
    gap = pb.cfg.repeat_gap_h * 3600
    previous_margin = (task.start_s - np.max(parent.start_s[previous] + parent.dur_s[previous]) - gap) if len(previous) else 0
    next_margin = (np.min(parent.start_s[following]) - task.start_s - task.dur_s - gap) if len(following) else 0
    station = np.nonzero(parent.station == task.station)[0]
    ends = parent.start_s + parent.dur_s
    before = station[ends[station] <= task.start_s]
    after = station[parent.start_s[station] >= task.start_s + task.dur_s]
    free_before = task.start_s - (np.max(ends[before]) + pb.cfg.switch_s if len(before) else 0)
    free_after = (np.min(parent.start_s[after]) - pb.cfg.switch_s if len(after) else pb.cfg.days * 86400) - task.start_s - task.dur_s
    lo, hi = operation.region_start_s, operation.region_end_s
    local = station[(parent.start_s[station] < hi) & (ends[station] > lo)]
    busy = np.maximum(0, np.minimum(ends[local] + pb.cfg.switch_s, hi) - np.maximum(parent.start_s[local], lo)).sum()
    removed = list(operation.remove_indices)
    removed_score = float(task_score(pb.cfg, pb.P[parent.target[removed]], parent.dur_s[removed]).sum())
    replacements = operation.additions[1:]
    replacement_score = sum(float(task_score(pb.cfg, np.asarray([pb.P[t.target]]), np.asarray([t.dur_s]))[0]) for t in replacements)
    windows = pb.t_win[pb.t_ptr[task.target]:pb.t_ptr[task.target + 1]]
    return [float(len(same) > 0), len(same), float(len(previous) > 0), previous_margin,
            float(len(following) > 0), next_margin, busy / max(1, hi - lo), max(0, free_before), max(0, free_after),
            len(removed), removed_score, len(replacements), replacement_score,
            len(np.unique(pb.w_st[windows])), len(windows),
            *[float(operation.kind == kind) for kind in ("insert", "migrate", "trim", "replace")],
            len(operation.related_targets), len(local), task.start_s / (pb.cfg.days * 86400), task.dur_s,
            len(np.unique(parent.target)) / pb.n_targets]
