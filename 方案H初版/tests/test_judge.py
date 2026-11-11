"""神经网络放置规则：标签、网络的梯度与训练、模型存取、接入 H（含多规则稳健模式）。"""
import numpy as np
import pytest

from obsplan import judge, lagrange
from obsplan.checks import structural_violations
from obsplan.evolved import load_rule
from obsplan.famou_h_eval import FEATURES, feature_rows
from obsplan.greedy import candidates


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    p = {k: v.astype(np.float64) for k, v in judge.init(5, (6, 4), seed=1).items()}
    for k in p:                                     # 偏置初始为 0 时会有样本正好落在 ReLU 拐点上，差分不准
        if k[0] == "b":
            p[k] = rng.normal(0, 0.3, p[k].shape)
    Z = rng.normal(size=(80, 5))
    y = (rng.random(80) < 0.3).astype(float)
    w = rng.random(80) + 0.5
    loss, g = judge.loss_grad(p, Z, y, w)
    for k, idx in (("W0", (2, 1)), ("b1", (3,)), ("W2", (1, 0))):
        q = {kk: vv.copy() for kk, vv in p.items()}
        q[k][idx] += 1e-6
        assert (judge.loss_grad(q, Z, y, w)[0] - loss) / 1e-6 == pytest.approx(g[k][idx], rel=1e-3, abs=1e-8)


def test_fit_learns_a_simple_rule():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(20000, 4)).astype(np.float32)
    y = ((Z[:, 0] + 0.5 * Z[:, 1] ** 2) > 1.2).astype(np.float32)
    w = np.ones(len(y), np.float32)
    p, hist = judge.fit(Z[:15000], y[:15000], w[:15000], Z[15000:], y[15000:], w[15000:], hidden=(16, 16),
                        epochs=20, batch=512, lr=5e-3, log=lambda *_: None)
    best = min(hist, key=lambda h: h["loss"])
    assert best["auc"] > 0.97
    m = judge.evaluate_logits(judge.forward(p, Z[15000:]).astype(float), y[15000:], w[15000:])
    assert m["loss"] == pytest.approx(best["loss"], rel=1e-5)           # 返回的是留出损失最低的一轮


def test_evaluate_logits_reference_values():
    y = np.array([0, 0, 1, 1, 0, 1], float)
    w = np.ones(6)
    assert judge.evaluate_logits(np.array([0, 1, 2, 3, -1, 4.0]), y, w)["auc"] == pytest.approx(1.0)
    assert judge.evaluate_logits(np.array([3, 2, 1, 0, 4, -1.0]), y, w)["auc"] == pytest.approx(0.0)


def test_labels_mark_every_task_of_a_plan(sub):
    sch, _ = lagrange.solve(sub, iters=30, verbose=False)
    cw, cm = candidates(sub)
    y = judge.labels(sub, cw, cm, sch)
    # 每个任务都对应恰好一个候选（同一窗口同一时长的两个任务会被 48 小时间隔排除）
    assert y.sum() == len(sch)
    assert set(np.unique(sub.w_tg[cw[y == 1]])) == set(np.unique(sch.target))


def test_model_roundtrip_and_rule(sub, tmp_path):
    rng = np.random.default_rng(2)
    p = judge.init(len(judge.INPUTS), (8, 8), seed=3)
    mean, std = rng.normal(size=len(judge.INPUTS)), rng.random(len(judge.INPUTS)) + 0.5
    path = tmp_path / "m.npz"
    judge.save(path, p, mean, std, 0.5, 1.5, {"note": "测试"})
    m = judge.load_model(path)
    assert m["t_u"] == 0.5 and m["t_c"] == 1.5 and m["meta"]["note"] == "测试"
    cw, cm = candidates(sub)
    _, info = lagrange.solve(sub, iters=30, local_search=False, verbose=False)
    X = feature_rows(sub, info["mu"], cw, cm)
    assert X.shape[1] == len(FEATURES)
    u, c = judge.make_rule(m)(X)
    z = judge.scores(m, X)
    assert np.allclose(u, z - 0.5) and np.allclose(c, z - 1.5)


def test_portfolio_with_several_rules_picks_the_best(sub, tmp_path):
    famou = load_rule()
    p = judge.init(len(judge.INPUTS), (8, 8), seed=4)
    path = tmp_path / "m.npz"
    judge.save(path, p, np.zeros(len(judge.INPUTS)), np.ones(len(judge.INPUTS)), 0.0, 0.0, {})
    net = judge.make_rule(judge.load_model(path), famou=famou)
    single, i1 = lagrange.solve(sub, iters=40, verbose=False, rule=famou, portfolio=True)
    one_list, i2 = lagrange.solve(sub, iters=40, verbose=False, rule=[famou], portfolio=True)
    assert i1["portfolio"]["objective"] == i2["portfolio"]["objective"]          # 单条规则写成列表，结果不变
    sch, info = lagrange.solve(sub, iters=40, verbose=False, rule=[famou, net], portfolio=True)
    obj = info["portfolio"]["objective"]
    assert set(obj) == {"rule0", "rule1", "manual"}
    assert obj[info["portfolio"]["picked"]] == max(obj.values())
    assert obj["rule0"] == pytest.approx(i1["portfolio"]["objective"]["rule"])
    assert not structural_violations(sub, sch)
    with pytest.raises(ValueError):
        lagrange.solve(sub, iters=5, verbose=False, rule=[famou, net], portfolio=False)
