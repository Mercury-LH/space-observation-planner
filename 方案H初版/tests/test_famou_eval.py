"""伐谋候选在沙箱中运行：正常程序得 0.5，越权或格式错误的程序判无效。"""
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="候选隔离依赖 macOS sandbox-exec")

INIT = Path(__file__).resolve().parents[1] / "integrations" / "famou" / "init.py"


@pytest.fixture(scope="module")
def cases(cfg):
    from obsplan.famou_eval import load_cases
    return load_cases(cfg)


def test_initial_program_scores_half(cases):
    from obsplan.famou_eval import evaluate_source
    r = evaluate_source(INIT.read_bytes(), cases)
    assert r["validity"] == 1
    assert r["combined_score"] == pytest.approx(0.5, abs=1e-5)


@pytest.mark.parametrize("src", [
    b"import os\ndef score(names, rows):\n    os.listdir('/Users')\n    return [], []\n",
    b"def score(names, rows):\n    open('/etc/hosts').read()\n    return [], []\n",
    b"def score(names, rows):\n    return [1.0], [1.0]\n",
    b"def score(names, rows):\n    return [float('nan')]*len(rows), [0.0]*len(rows)\n",
])
def test_bad_candidates_rejected(cases, src):
    from obsplan.famou_eval import evaluate_source
    r = evaluate_source(src, cases)
    assert r["validity"] == 0 and r["combined_score"] == 0.0
