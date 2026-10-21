"""加分项：天气回放与月光配置（不联网，用构造的云量）。"""
import numpy as np

from obsplan.environment import clear_probability, evaluate_under_weather
from obsplan.greedy import MANUAL_WEIGHTS, algo_weighted
from obsplan.schedule import evaluate


def test_weather_replay_extremes(cfg, ds, sub):
    sch = algo_weighted(sub, MANUAL_WEIGHTS, "C")
    hours = int(cfg.days * 24)
    clear = np.zeros((2, ds.n_stations, hours))
    cloudy = np.full((2, ds.n_stations, hours), 100.0)
    nominal = evaluate(cfg, ds, sch)
    ok = evaluate_under_weather(cfg, ds, sch, clear, 30)
    assert ok["coverage_mean"] == nominal["covered"]
    assert abs(ok["score_mean"] - nominal["score"]) < 1e-6
    bad = evaluate_under_weather(cfg, ds, sch, cloudy, 30)
    assert bad["coverage_mean"] == 0 and bad["score_mean"] == 0


def test_clear_probability_shape(cfg, ds):
    hours = int(cfg.days * 24)
    cloud = np.zeros((3, ds.n_stations, hours))
    cloud[0] = 100.0                                    # 三年里有一年全阴
    q = clear_probability(cfg, cloud, 30)
    assert q.shape[0] == ds.n_stations and q.shape[1] >= cfg.n_cells + 2
    assert np.allclose(q, 2 / 3)


def test_moon_changes_scenario_key(cfg):
    assert cfg.scenario_key() != cfg.with_(moon_sep_deg=30).scenario_key()
    assert "moon" not in cfg.scenario_key()
