"""Streamlit 交互回归：隔离计算与磁盘目录，检验配置、失败恢复和校验状态。

这些 AppTest 测试不启动浏览器，也不代表真实数据规划或浏览器视觉验证。
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from obsplan import app_backend as ab  # noqa: E402
from obsplan import llm_assistant  # noqa: E402
from obsplan.config import Config  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture
def ui(monkeypatch, tmp_path):
    """只把 CPU/网络工作换成桩；执行原始 UI 文件、真实 Run 保存和加载。"""
    st.cache_resource.clear()
    st.cache_data.clear()
    loaded, planned, subsets = [], [], []
    ds = SimpleNamespace(n_targets=3, n_stations=10, priority=np.array([5., 3., 2.]),
                         priority_label=["高", "中", "低"])

    def load_problem(cfg):
        loaded.append(cfg)
        pb = SimpleNamespace(cfg=cfg, ds=ds, n_stations=10)
        return ds, pb, {"load_s": 0., "windows_s": 0., "windows_cached": True, "build_s": 0.}

    def subset_problem(pb, frac, stations, seed=0):
        subsets.append((frac, list(stations), seed))
        return pb

    def unexpected_plan(pb, algo, **kwargs):
        planned.append((pb.cfg, algo, kwargs))
        raise RuntimeError("测试计划计算失败")

    real_save = ab.Run.save
    monkeypatch.setattr(ab, "list_runs", lambda folder: sorted(tmp_path.glob("*.npz")))
    monkeypatch.setattr(ab.Run, "save", lambda self, folder: real_save(self, tmp_path))
    monkeypatch.setattr(ab, "load_problem", load_problem)
    monkeypatch.setattr(ab, "orbit_class", lambda dataset: np.array(["低轨"] * dataset.n_targets))
    monkeypatch.setattr(ab, "subset_problem", subset_problem)
    monkeypatch.setattr(ab, "run_planning", unexpected_plan)
    monkeypatch.setattr(ab, "task_rows", lambda *args: [])
    monkeypatch.setattr(ab, "status_rows", lambda *args: [])
    monkeypatch.setattr(ab, "csv_bytes", lambda *args: b"test-only")
    monkeypatch.setattr(ab, "status_csv_bytes", lambda *args: b"test-only")
    monkeypatch.setattr(llm_assistant, "api_key", lambda: None)

    def forbid_chat(*args, **kwargs):
        raise AssertionError("UI 回归不得调用付费模型")

    monkeypatch.setattr(llm_assistant, "chat", forbid_chat)
    yield SimpleNamespace(folder=tmp_path, loaded=loaded, planned=planned, subsets=subsets,
                          app=lambda: AppTest.from_file(str(APP), default_timeout=15))
    st.cache_resource.clear()
    st.cache_data.clear()


def saved_run(ui, *, tasks=1, validation=None):
    cfg = Config()
    values = np.zeros(tasks, dtype=np.int64)
    run = ab.Run("ui-test", "H", json.loads(cfg.to_json()), {}, values, values.copy(), values.copy(),
                 np.full(tasks, 120, dtype=np.int64),
                 {"tasks": tasks, "covered": int(tasks > 0), "score": 3. * tasks, "combined": .1 * tasks},
                 1., validation=validation)
    run.save(ui.folder)
    return run


@pytest.mark.parametrize("entry,days,subset,chain", [
    ("welcome_demo", 1., (0.05, [0, 1], 0), 0.),
    ("welcome_full", 7., None, 0.5),
])
def test_first_visit_presets_reach_planner(ui, entry, days, subset, chain):
    at = ui.app().run()
    assert not at.exception and not ui.loaded
    at.button(key=entry).click().run()
    if entry == "welcome_full":
        assert at.radio(key="plan_scope").value == "全量规划"
        assert not ui.planned, "选择配置不应立即运行全量计算"
        at.button(key="run_planning").click().run()
    assert not at.exception
    cfg, algo, options = ui.planned[-1]
    assert cfg.days == days and cfg.start_utc == "2026-07-31T00:00:00"
    assert (cfg.elev_min, cfg.elev_max, cfg.sun_max, cfg.use_sunlit) == (10., 85., -6., True)
    assert algo == "H" and options["fast"] and options["evolved_rule"] and options["refine_seconds"]
    assert options["chain_minutes"] == chain and options["full_problem"] == (subset is None)
    assert ui.subsets == ([] if subset is None else [subset])
    assert any("生成计划失败" in e.value for e in at.error)


def test_empty_custom_stations_disable_start(ui):
    at = ui.app().run()
    at.radio(key="plan_scope").set_value("自定义范围").run()
    at.multiselect(key="plan_stations").set_value([]).run()
    assert not at.exception and at.button(key="run_planning").disabled
    assert any("至少选择一个站点" in w.value for w in at.sidebar.warning)
    assert not ui.loaded and not ui.planned


def test_unreadable_history_is_reported_and_preserved(ui):
    path = ui.folder / "broken.npz"
    path.write_bytes(b"incomplete result")
    at = ui.app().run()
    assert not at.exception
    assert any("1 份历史结果无法读取" in w.value for w in at.sidebar.warning)
    assert any("broken.npz" in text.value for text in at.sidebar.text)
    assert not at.button(key="welcome_demo").disabled
    assert path.read_bytes() == b"incomplete result"


def test_failed_planning_can_retry_without_losing_configuration(ui, monkeypatch):
    at = ui.app().run()
    at.button(key="run_planning").click().run()
    assert not at.exception and any("生成计划失败" in e.value for e in at.error)
    assert not at.button(key="run_planning").disabled
    attempts = []

    def succeed(pb, algo, **kwargs):
        attempts.append((pb.cfg, algo, kwargs))
        run = saved_run(ui)
        run.cfg = json.loads(pb.cfg.to_json())
        return run

    # 此测试只检查重试后保存和选中结果；结果展示在下两个测试中独立覆盖。
    monkeypatch.setattr(ab, "list_runs", lambda folder: [])
    monkeypatch.setattr(ab, "run_planning", succeed)
    at.button(key="run_planning").click().run()
    assert not at.exception and len(attempts) == 1
    assert attempts[0][0] == ui.planned[0][0]
    assert at.session_state["selected"] == "ui-test"
    restored = ab.Run.load(ui.folder / "ui-test.npz")
    assert restored.validation is None and restored.subset["stations"] == [0, 1]


def test_failed_independent_validation_keeps_plan_and_allows_retry(ui, monkeypatch):
    saved_run(ui)
    attempts = []

    def validate(pb, run):
        attempts.append(run.run_id)
        if len(attempts) == 1:
            raise OSError("测试星历暂时不可读")
        run.validation = {"legal": True, "violations": {}, "margins": [1., 2., 3.], "seconds": .1}
        return run

    monkeypatch.setattr(ab, "validate_run", validate)
    at = ui.app()
    at.session_state["page"] = "校验与导出"
    at.run()
    assert not at.exception
    at.button(key="validate_export").click().run()
    assert not at.exception and any("未获得新的校验结论" in e.value for e in at.error)
    assert ab.Run.load(ui.folder / "ui-test.npz").validation is None
    assert not at.button(key="validate_export").disabled
    at.button(key="validate_export").click().run()
    assert not at.exception and len(attempts) == 2
    assert ab.Run.load(ui.folder / "ui-test.npz").validation["legal"]
    assert any("校验通过" in s.value for s in at.success)


def test_zero_task_validation_with_missing_margins_is_not_success(ui):
    saved_run(ui, tasks=0, validation={"legal": True, "violations": {}, "margins": [None, None, None], "seconds": 0.})
    at = ui.app()
    at.session_state["page"] = "校验与导出"
    at.run()
    assert not at.exception and at.button(key="validate_export").disabled
    assert any("没有观测任务" in w.value for w in at.warning)
    assert any("无可用测量" in m.value for m in at.markdown)
    assert not any("校验通过" in s.value for s in at.success)


def test_one_day_price_page_renders_without_single_value_slider(ui):
    run = saved_run(ui)
    cfg = Config(days=1.)
    run.cfg = json.loads(cfg.to_json())
    run.mu = np.zeros((10, cfg.n_cells), np.float32)
    run.save(ui.folder)
    at = ui.app()
    at.session_state["page"] = "时间价格"
    at.run()
    assert not at.exception
    assert any("当前规划仅 1 天" in c.value for c in at.caption)
    assert not any(s.key == "price_day" for s in at.slider)
    assert len(at.get("plotly_chart")) == 1


def test_empty_gantt_page_offers_recovery(ui):
    saved_run(ui, tasks=0)
    at = ui.app()
    at.session_state["page"] = "甘特图"
    at.run()
    assert not at.exception
    assert any("当前计划没有观测任务" in i.value for i in at.info)


def test_validation_save_failure_does_not_change_exported_conclusion(ui, monkeypatch):
    saved_run(ui)
    exported_conclusions = []
    real_summary = ab.summary_json

    def summary(run):
        exported_conclusions.append(run.validation)
        return real_summary(run)

    def validate(pb, run):
        run.validation = {"legal": True, "violations": {}, "margins": [1., 2., 3.], "seconds": .1}
        return run

    def refuse_save(run, folder):
        raise OSError("测试磁盘暂不可写")

    monkeypatch.setattr(ab, "summary_json", summary)
    monkeypatch.setattr(ab, "validate_run", validate)
    monkeypatch.setattr(ab.Run, "save", refuse_save)
    at = ui.app()
    at.session_state["page"] = "校验与导出"
    at.run()
    at.button(key="validate_export").click().run()
    assert not at.exception and any("新结论未保存" in e.value for e in at.error)
    assert exported_conclusions[-1] is None
    assert ab.Run.load(ui.folder / "ui-test.npz").validation is None
    assert not at.button(key="validate_export").disabled
