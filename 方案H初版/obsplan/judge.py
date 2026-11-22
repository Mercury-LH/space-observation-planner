"""神经网络放置规则：一个很小的多层感知机，给 H 的每个候选任务（窗口, 时长）打分，代替第一遍贪心的排序。

- 输入：放置规则的 20 个特征（famou_h_eval.FEATURES），加上人工规则和伐谋规则各自的两个打分，共 24 维；
- 输出：一个分数（老师的计划选中这个候选的对数几率）；
- 接口与伐谋规则相同：rule(X) -> (未覆盖时的分, 已覆盖时的分)，第一遍只放分数为正的候选，
  所以 u = 分数 − t_u、c = 分数 − t_c，两个阈值在验证场景上按端到端结果选；
- 推理只用 numpy（权重存在 .npz 里，约 6000 个参数），评委机器上不需要 torch；
- 网络只决定"先试哪个"，放不放得下、合不合法仍由贪心和规则引擎检查，打分错了也不会产生违规计划。

训练（scripts/judge_pipeline.py train）也只用 numpy：带样本权重的二分类交叉熵 + Adam。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .famou_h_eval import FEATURES, baseline_rule

MODEL = Path(__file__).resolve().parent / "models" / "judge_mlp.npz"
INPUTS = FEATURES + ["manual_u", "manual_c", "famou_u", "famou_c"]


def inputs(X: np.ndarray, famou=None) -> np.ndarray:
    """20 个特征 → 24 维网络输入（加上人工规则、伐谋规则的打分）。"""
    X = np.asarray(X, np.float64)
    mu_, mc = baseline_rule(X)
    if famou is None:
        from .evolved import load_rule
        famou = load_rule()
    fu, fc = famou(X) if famou is not None else (mu_, mc)
    return np.column_stack([X, mu_, mc, fu, fc]).astype(np.float32)


def labels(pb, cw: np.ndarray, cm: np.ndarray, sch) -> np.ndarray:
    """候选 (窗口 cw, 时长 cm) 是否出现在计划 sch 里（同站同目标、开始与结束落在该窗口内、时长相同）。"""
    from .fusion_merge import _cells, find_window
    key = cw.astype(np.int64) * 64 + cm
    order = np.argsort(key)
    y = np.zeros(len(cw), np.uint8)
    for s, t, a, m in zip(*_cells(pb, sch)):
        w = find_window(pb, int(s), int(t), int(a), int(m))
        if w < 0:
            continue
        k = w * 64 + m
        i = np.searchsorted(key[order], k)
        if i < len(order) and key[order[i]] == k:
            y[order[i]] = 1
    return y


# ---------------------------------------------------------------- 网络
def init(n_in: int, hidden=(64, 64), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    sizes = [n_in, *hidden, 1]
    p = {}
    for i in range(len(sizes) - 1):
        p[f"W{i}"] = (rng.standard_normal((sizes[i], sizes[i + 1])) * np.sqrt(2.0 / sizes[i])).astype(np.float32)
        p[f"b{i}"] = np.zeros(sizes[i + 1], np.float32)
    return p


def _layers(p: dict) -> int:
    return sum(1 for k in p if k.startswith("W"))


def forward(p: dict, Z: np.ndarray, keep: bool = False):
    """Z 已标准化。返回对数几率（keep=True 时同时返回各层激活，反向传播用）。"""
    acts = [Z]
    h = Z
    L = _layers(p)
    for i in range(L):
        h = h @ p[f"W{i}"] + p[f"b{i}"]
        if i < L - 1:
            h = np.maximum(h, 0.0)
        acts.append(h)
    out = h[:, 0]
    return (out, acts) if keep else out


def loss_grad(p: dict, Z: np.ndarray, y: np.ndarray, w: np.ndarray):
    """带样本权重的二分类交叉熵（按权重和平均）及梯度。"""
    z, acts = forward(p, Z, keep=True)
    wsum = float(w.sum())
    loss = float((w * (np.logaddexp(0.0, z) - y * z)).sum() / wsum)
    g = ((1.0 / (1.0 + np.exp(-z)) - y) * w / wsum).astype(np.float32)[:, None]
    grads = {}
    L = _layers(p)
    for i in reversed(range(L)):
        grads[f"W{i}"] = acts[i].T @ g
        grads[f"b{i}"] = g.sum(axis=0)
        if i > 0:
            g = (g @ p[f"W{i}"].T) * (acts[i] > 0)
    return loss, grads


def standardize(Z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip((Z - mean) / std, -8.0, 8.0).astype(np.float32)


def fit(Z: np.ndarray, y: np.ndarray, w: np.ndarray, Zv: np.ndarray, yv: np.ndarray, wv: np.ndarray,
        hidden=(64, 64), epochs: int = 8, batch: int = 8192, lr: float = 2e-3, wd: float = 1e-5, seed: int = 0,
        log=print) -> tuple[dict, list]:
    """Adam 训练，按留出场景的损失选最好的一轮。Z / Zv 已标准化。"""
    rng = np.random.default_rng(seed)
    p = init(Z.shape[1], hidden, seed)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v2 = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
    best, best_loss, hist = None, np.inf, []
    n = len(Z)
    for ep in range(epochs):
        perm = rng.permutation(n)
        rate = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs)))          # 余弦退火
        tr = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            loss, g = loss_grad(p, Z[idx], y[idx], w[idx])
            tr += loss * len(idx)
            step += 1
            for k in p:
                gk = g[k] + wd * p[k]
                m[k] = b1 * m[k] + (1 - b1) * gk
                v2[k] = b2 * v2[k] + (1 - b2) * gk * gk
                p[k] -= rate * (m[k] / (1 - b1 ** step)) / (np.sqrt(v2[k] / (1 - b2 ** step)) + eps)
        val = evaluate_logits(forward(p, Zv), yv, wv)
        hist.append({"epoch": ep, "train_loss": tr / n, **val})
        log(f"  第 {ep + 1}/{epochs} 轮：训练损失 {tr / n:.5f}  留出损失 {val['loss']:.5f}  AUC {val['auc']:.4f}"
            f"  前 2% 命中率 {val['prec_at_pos']:.3f}")
        if val["loss"] < best_loss:
            best_loss, best = val["loss"], {k: v.copy() for k, v in p.items()}
    return best, hist


def evaluate_logits(z: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    """损失、AUC（按权重）、"打分最高、个数等于正例数的候选里有多少是正例"。"""
    loss = float((w * (np.logaddexp(0.0, z) - y * z)).sum() / w.sum())
    order = np.argsort(z)
    ys, ws = y[order], w[order]
    neg_before = np.cumsum(ws * (1 - ys)) - ws * (1 - ys)
    pos_w, neg_w = float((ws * ys).sum()), float((ws * (1 - ys)).sum())
    auc = float((ws * ys * neg_before).sum() / max(pos_w * neg_w, 1e-12))
    k = int(round(pos_w))
    top = order[::-1]
    cum = np.cumsum(w[top])
    cut = np.searchsorted(cum, k)
    prec = float((w[top[:cut + 1]] * y[top[:cut + 1]]).sum() / max(cum[min(cut, len(cum) - 1)], 1e-12))
    return {"loss": loss, "auc": auc, "prec_at_pos": prec}


# ---------------------------------------------------------------- 保存、加载、接入 H
def save(path: Path, p: dict, mean: np.ndarray, std: np.ndarray, t_u: float, t_c: float, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean.astype(np.float32), std=std.astype(np.float32), t_u=np.float64(t_u),
             t_c=np.float64(t_c), inputs=np.array(INPUTS), meta=np.array(json.dumps(meta, ensure_ascii=False)),
             **p)


def load_model(path: Path | str | None = None) -> dict | None:
    path = Path(path) if path else MODEL
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    if list(z["inputs"]) != INPUTS:
        raise ValueError(f"{path} 的输入特征与当前代码不一致")
    p = {k: z[k] for k in z.files if k[0] in "Wb" and k[1:].isdigit()}
    return {"p": p, "mean": z["mean"], "std": z["std"], "t_u": float(z["t_u"]), "t_c": float(z["t_c"]),
            "meta": json.loads(str(z["meta"]))}


def scores(model: dict, X: np.ndarray, famou=None) -> np.ndarray:
    """候选的对数几率（20 个特征 → 24 维输入 → 网络）。"""
    return forward(model["p"], standardize(inputs(X, famou), model["mean"], model["std"])).astype(np.float64)


def make_rule(model: dict | None = None, t_u: float | None = None, t_c: float | None = None, famou=None):
    """lagrange.recover / solve 用的放置规则 rule(X) -> (u, c)；没有模型文件时返回 None。"""
    model = model or load_model()
    if model is None:
        return None
    tu = model["t_u"] if t_u is None else t_u
    tc = model["t_c"] if t_c is None else t_c
    if famou is None:
        from .evolved import load_rule
        famou = load_rule()

    def rule(X: np.ndarray):
        z = scores(model, X, famou)
        return z - tu, z - tc
    rule.judge = True
    return rule
