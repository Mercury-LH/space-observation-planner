"""Exercise slider reruns and dispatch without launching expensive optimizers."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest

from obsplan import app_backend as backend, data, visibility, problem
from obsplan.effort import EFFORT_ORDER

APP = Path(__file__).resolve().parents[1] / 'app/streamlit_app.py'


@pytest.fixture
def ui(monkeypatch):
    monkeypatch.setattr(backend, 'list_runs', lambda folder: [])
    monkeypatch.setattr(data, 'load', lambda cfg: pytest.fail('Changing a slider must not load/solve the problem'))
    return AppTest.from_file(str(APP), default_timeout=15)


def test_six_discrete_choices_update_description_without_solving(ui, monkeypatch):
    monkeypatch.setattr(backend, 'run_effort', lambda *a, **kw: pytest.fail('Slider must not start planning'))
    ui.run()
    assert not ui.exception
    assert ui.select_slider[0].value == 'instant'
    assert tuple(ui.select_slider[0].options) == EFFORT_ORDER
    for name in EFFORT_ORDER:
        ui.select_slider[0].set_value(name).run()
        assert not ui.exception
        assert any(name in item.value for item in ui.markdown)
    ui.select_slider[0].set_value('high').run()
    assert any('尚待同条件校准' in item.value for item in ui.info)


def test_manual_mode_and_empty_station_guard(ui):
    ui.run()
    ui.multiselect[0].set_value([]).run()
    assert ui.button[0].disabled
    ui.radio[0].set_value('自定义算法').run()
    assert not ui.exception
    assert not ui.select_slider
    assert tuple(ui.radio[1].options) == tuple(backend.ALGOS[k] for k in ('H', 'E', 'B'))


def test_run_button_dispatches_selected_effort_and_persists_subset(ui, monkeypatch):
    ui.run()
    ui.select_slider[0].set_value('ultra').run()
    ui.multiselect[0].set_value([1, 3]).run()
    called = {}
    monkeypatch.setattr(data, 'load', lambda cfg: SimpleNamespace())
    monkeypatch.setattr(visibility, 'get_windows', lambda cfg, ds: object())
    monkeypatch.setattr(problem.Problem, 'build', lambda *args: SimpleNamespace(cfg=args[0]))
    monkeypatch.setattr(backend, 'orbit_class', lambda ds: np.array([]))
    monkeypatch.setattr(backend, 'subset_problem', lambda pb, frac, stations, seed: pb)
    fake = SimpleNamespace(run_id='test-effort', runtime_s=2, validation={'legal': True}, subset=None, notes={})
    def save(folder):
        called['saved_subset'] = fake.subset
    fake.save = save
    def solve(pb, effort, **kwargs):
        called.update(effort=effort, **kwargs)
        return fake
    monkeypatch.setattr(backend, 'run_effort', solve)
    monkeypatch.setattr(backend, 'run_planning', lambda *a, **kw: pytest.fail('Automatic mode bypassed effort adapter'))
    ui.button[0].click().run()
    assert not ui.exception
    assert called['effort'] == 'ultra'
    assert called['full_problem'] is False
    assert called['saved_subset']['stations'] == [0, 2]
    assert called['workers'] >= 1
    assert fake.notes['ui_total_time_s'] >= fake.notes['upstream_preprocessing_time_s'] >= 0
    assert ui.session_state['selected'] == 'test-effort'
