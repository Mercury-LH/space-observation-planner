"""Explicit known competition application loader, separate from synthetic holdouts."""
from pathlib import Path
from time import perf_counter

from obsplan.benchmark_worker import dataset_hash
from obsplan.config import Config
from obsplan.data import load
from obsplan.problem import Problem
from obsplan.visibility import get_windows
from .runtime import canonical_hash

CATALOG_ID = 'known-competition-10000-10-6d'


def load_application(data_dir, cache_dir):
    started = perf_counter()
    data, cache = Path(data_dir).resolve(), Path(cache_dir).resolve()
    if cache.is_relative_to(data):
        raise ValueError('application cache must be outside read-only original inputs')
    cache.mkdir(parents=True, exist_ok=True)
    cfg = Config(data_dir=str(data), cache_dir=str(cache), start_utc='2026-08-01T00:00:00', days=6.)
    ds = load(cfg)
    if ds.n_targets != 10000 or ds.n_stations != 10:
        raise ValueError('known application must contain exactly 10000 targets and 10 stations')
    windows_started = perf_counter()
    win = get_windows(cfg, ds)
    window_s = perf_counter()-windows_started
    pb = Problem.build(cfg, ds, win)
    return pb, {'catalog_id': CATALOG_ID, 'scene_id': CATALOG_ID, 'split': 'application',
                'stress_type': None, 'independent_generalization_test': False,
                'data_prepare_s': perf_counter()-started, 'full_windows_s': window_s,
                'windows_mode': 'common geometry with new isolated cache',
                'input_hash': dataset_hash(ds), 'config_hash': canonical_hash(cfg.to_json())}
