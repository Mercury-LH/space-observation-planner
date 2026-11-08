"""Train-only, scene-balanced statistics and deterministic bounded-memory batches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..famou_h_eval import FEATURES
from ..training_data.loader import DatasetReader


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def select_records(reader: DatasetReader, catalogs: list[str], split: str) -> list[dict]:
    if not catalogs or len(set(catalogs)) != len(catalogs):
        raise ValueError('catalog list must be nonempty and unique')
    wanted = set(catalogs)
    records = [r for r in reader.records(split) if r['catalog_id'] in wanted]
    if {r['catalog_id'] for r in records} != wanted:
        raise ValueError(f'catalog missing or outside required {split} split')
    if len(records) != len(catalogs):
        raise ValueError('v0.2 requires one scene per catalog')
    return sorted(records, key=lambda r: r['catalog_id'])


def fit_scene_normalizer(reader: DatasetReader, records: list[dict]) -> dict:
    """Equal scene weight, with full natural row distribution inside each scene.

    This differs intentionally from the release loader's all-row statistics.
    Memory holds one compressed scene, never the complete corpus.
    """
    if not records or any(r['split'] != 'train' for r in records):
        raise ValueError('normalizer may only use nonempty train records')
    means, variances, positive_rates, count = [], [], [], 0
    for record in records:
        arrays = reader._load_actions(record)
        x = arrays['features']
        if not len(x):
            raise ValueError('empty scene cannot enter balanced training')
        # Chunked moment accumulation avoids a second float64 scene allocation.
        n, mean, m2 = 0, np.zeros(20), np.zeros(20)
        for offset in range(0, len(x), 8192):
            part = x[offset:offset + 8192].astype(np.float64)
            delta = part.mean(0) - mean
            total = n + len(part)
            m2 += ((part - part.mean(0)) ** 2).sum(0) + delta ** 2 * n * len(part) / total
            mean += delta * len(part) / total
            n = total
        means.append(mean)
        variances.append(m2 / n)
        positive_rates.append(float(arrays['grid_selected'].mean()))
        count += n
    mean = np.mean(means, axis=0)
    var = np.mean([v + (m - mean) ** 2 for m, v in zip(means, variances)], axis=0)
    scale = np.sqrt(np.maximum(var, 0))
    scale[scale < 1e-12] = 1
    rate = float(np.mean(positive_rates))
    if not 0 < rate < 1:
        raise ValueError('train subset must contain positive and negative teacher choices')
    return {'version': 1, 'source_split': 'train', 'feature_names': list(FEATURES),
            'weighting': 'equal_scene_then_uniform_candidate', 'mean': mean.tolist(),
            'scale': scale.tolist(), 'row_count': count,
            'catalog_ids': [r['catalog_id'] for r in records],
            'scene_ids': [r['scene_id'] for r in records],
            'records_sha256': digest(records), 'positive_rate': rate,
            'pos_weight': (1 - rate) / rate}


def transform(x: np.ndarray, normalizer: dict, *, dim: int = 20) -> np.ndarray:
    x = np.asarray(x)
    mean, scale = np.asarray(normalizer['mean']), np.asarray(normalizer['scale'])
    if (x.ndim != 2 or x.shape[1] != dim or mean.shape != (dim,) or scale.shape != (dim,)
            or not np.isfinite(x).all() or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError('invalid features or normalization statistics')
    return np.ascontiguousarray((x - mean) / scale, dtype=np.float32)


class BalancedStream:
    """Fixed batches per scene per cycle, sampling rows with replacement.

    A step identifies scene, cycle, and batch exactly. Resume does not replay all
    prior compressed shards. Candidate IDs remain original release action rows.
    """
    def __init__(self, reader: DatasetReader, records: list[dict], normalizer: dict,
                 *, seed=2026, batch_size=8192, batches_per_scene=16):
        if not records or batch_size <= 0 or batches_per_scene <= 0:
            raise ValueError('nonempty records and positive batch sizes required')
        if any(r['split'] != 'train' for r in records):
            raise ValueError('training stream cannot use holdouts')
        if digest(records) != normalizer['records_sha256']:
            raise ValueError('normalizer does not match train subset')
        self.reader, self.records, self.normalizer = reader, records, normalizer
        self.seed, self.batch_size, self.batches_per_scene = seed, batch_size, batches_per_scene
        self.cached_id, self.arrays = None, None

    def batch(self, step: int) -> dict:
        if step < 0:
            raise ValueError('step must be nonnegative')
        cycle, position = divmod(step, len(self.records) * self.batches_per_scene)
        order = np.random.default_rng(np.random.SeedSequence([self.seed, cycle])).permutation(len(self.records))
        record = self.records[int(order[position // self.batches_per_scene])]
        if self.cached_id != record['scene_id']:
            self.arrays = self.reader._load_actions(record)
            self.cached_id = record['scene_id']
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, 7301]))
        ids = rng.integers(0, len(self.arrays['features']), size=self.batch_size, dtype=np.int64)
        return {'features': transform(self.arrays['features'][ids], self.normalizer),
                'labels': self.arrays['grid_selected'][ids].astype(np.float32),
                'candidate_id': ids, 'scene_id': record['scene_id'],
                'catalog_id': record['catalog_id'], 'step': step}
