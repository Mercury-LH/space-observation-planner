"""合成场景：侯治民的 synthetic-v0.2（600 个六天场景，训练 / 验证 / 测试 = 420 / 90 / 90），
加上我们补的 v0.3"像赛题"场景（180 个，120 / 24 / 36，见 training_data/v3_sampling.py）。

不下载 8.9 GB 的数据包：生成器（obsplan/training_data/，取自他的数据集分支 GENERATOR_COMMIT，
加了 v0.3 需要的几处可选参数，v0.2 结果不变）用同一组种子在本机重建场景输入，可见窗口用我们自己的程序算。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import Config
from .data import load
from .problem import Problem
from .visibility import Windows, compute_windows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "judge"
GENERATOR_COMMIT = "f87515208ce4d26b67ee9dc3c39bdfcae8726cc1"      # feat/neural-training-dataset-v2
DATASETS = ("v2", "v3")


@lru_cache(maxsize=None)
def _specs(dataset: str) -> tuple:
    if dataset == "v2":
        from .training_data import v2_sampling as m
    elif dataset == "v3":
        from .training_data import v3_sampling as m
    else:
        raise ValueError(f"未知数据集 {dataset}")
    return tuple(m.specs())


def specs(dataset: str | None = None) -> list[dict]:
    """场景参数（确定性生成，v0.2 与数据集里冻结的 catalog-specs.json 完全一致）。dataset=None 时两套都给。"""
    return [s for d in ((dataset,) if dataset else DATASETS) for s in _specs(d)]


def select(split: str, n: int | None = None, stress: bool = True, dataset: str = "v2") -> list[str]:
    """某个划分里的场景编号。n 给定时均匀抽 n 个（按编号排序后等距取，保证各种规模都有）；
    测试集和验证集按"主体 + 每类压力"等比例抽。"""
    sp = sorted((s for s in specs(dataset) if s["split"] == split), key=lambda s: s["catalog_id"])
    if not stress:
        sp = [s for s in sp if s["sampling_group"] != "stress"]
    ids = [s["catalog_id"] for s in sp]
    if n is None or n >= len(ids):
        return ids
    idx = np.unique(np.linspace(0, len(ids) - 1, n).round().astype(int))
    return [ids[i] for i in idx]


def spec_of(cid: str) -> dict:
    return next(s for s in specs(cid.split("-")[0]) if s["catalog_id"] == cid)


def scene_dir(cid: str, out: Path = OUT) -> Path:
    return out / "scenes" / cid


def generate(cid: str, out: Path = OUT) -> Path:
    """重建一个场景的输入（已有就跳过）。"""
    from .training_data import catalog, v2_sampling
    d = scene_dir(cid, out)
    if (d / "spec.json").exists():
        return d
    spec = spec_of(cid)
    ds, _ = v2_sampling.generate_catalog(spec)
    catalog.write_native_input(d, ds)
    (d / "spec.json").write_text(json.dumps({**spec, "generator_commit": GENERATOR_COMMIT}, ensure_ascii=False,
                                            indent=1), encoding="utf-8")
    return d


def scene_config(cid: str, out: Path = OUT) -> Config:
    spec = json.loads((scene_dir(cid, out) / "spec.json").read_text(encoding="utf-8"))
    return Config(start_utc=spec["start_utc"], data_dir=str(scene_dir(cid, out)), cache_dir=str(out / "cache" / cid),
                  seed=int(spec["seed"]))


def load_scene(cid: str, out: Path = OUT, workers: int = 8):
    """(cfg, ds, pb, spec)。窗口与短窗口缓存在 outputs/judge/cache/<场景>/（与 get_windows 同格式）。"""
    generate(cid, out)
    cfg = scene_config(cid, out)
    ds = load(cfg)
    if ds.issues:
        raise ValueError(f"{cid} 输入有问题：{ds.issues}")
    path = Path(cfg.cache_dir) / f"windows_{cfg.scenario_key()}.npz"
    if path.exists():
        win = Windows.load(path, ds.n_stations)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        win = compute_windows(cfg, ds, workers=workers, verbose=False, keep_short=True)
        win.save(path)
        st, tg, k0, k1 = win.short
        np.savez_compressed(path.with_name(f"short_{cfg.scenario_key()}.npz"), station=st, target=tg, k0=k0, k1=k1)
    spec = json.loads((scene_dir(cid, out) / "spec.json").read_text(encoding="utf-8"))
    return cfg, ds, Problem.build(cfg, ds, win), spec
