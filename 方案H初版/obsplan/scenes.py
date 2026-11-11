"""合成场景（侯治民的 synthetic-v0.2，600 个六天场景，训练 / 验证 / 测试 = 420 / 90 / 90）。

不下载 8.9 GB 的数据包：用数据集分支里同一个生成器、同一组种子在本机重建场景输入（与数据包里的 input/
同格式），可见窗口用我们自己的程序算。生成器从 git 分支取出，放在 outputs/judge/generator/（不提交）。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

from .config import Config
from .data import load
from .problem import Problem
from .visibility import Windows, compute_windows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "judge"
GENERATOR_COMMIT = "f87515208ce4d26b67ee9dc3c39bdfcae8726cc1"      # feat/neural-training-dataset-v2
GENERATOR_FILES = {"v2_sampling.py": "obsplan/training_data/v2_sampling.py",
                   "catalog.py": "obsplan/training_data/catalog.py",
                   "catalog-specs.json": "datasets/synthetic-v0.2/catalog-specs.json"}


def generator_dir(out: Path = OUT) -> Path:
    """生成器三个文件；没有就从 git 取（需要先 git fetch origin feat/neural-training-dataset-v2）。"""
    d = out / "generator"
    if all((d / f).exists() for f in GENERATOR_FILES):
        return d
    d.mkdir(parents=True, exist_ok=True)
    for name, path in GENERATOR_FILES.items():
        data = subprocess.run(["git", "show", f"{GENERATOR_COMMIT}:./{path}"], cwd=ROOT, check=True,
                              capture_output=True).stdout
        (d / name).write_bytes(data)
    (d / "COMMIT").write_text(GENERATOR_COMMIT)
    return d


def specs(out: Path = OUT) -> list[dict]:
    return json.loads((generator_dir(out) / "catalog-specs.json").read_text(encoding="utf-8"))


def select(split: str, n: int | None = None, out: Path = OUT, stress: bool = True) -> list[str]:
    """某个划分里的场景编号。n 给定时均匀抽 n 个（按编号排序后等距取，保证各种规模都有）；
    测试集和验证集按"主体 + 每类压力"等比例抽。"""
    sp = sorted((s for s in specs(out) if s["split"] == split), key=lambda s: s["catalog_id"])
    if not stress:
        sp = [s for s in sp if s["sampling_group"] == "core"]
    ids = [s["catalog_id"] for s in sp]
    if n is None or n >= len(ids):
        return ids
    idx = np.unique(np.linspace(0, len(ids) - 1, n).round().astype(int))
    return [ids[i] for i in idx]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scene_dir(cid: str, out: Path = OUT) -> Path:
    return out / "scenes" / cid


def generate(cid: str, out: Path = OUT) -> Path:
    """重建一个场景的输入（已有就跳过）。"""
    d = scene_dir(cid, out)
    if (d / "spec.json").exists():
        return d
    g = generator_dir(out)
    sampling = _module(g / "v2_sampling.py", "hou_v2_sampling")
    catalog = _module(g / "catalog.py", "hou_catalog")
    spec = next(s for s in specs(out) if s["catalog_id"] == cid)
    ds, _ = sampling.generate_catalog(spec)
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
