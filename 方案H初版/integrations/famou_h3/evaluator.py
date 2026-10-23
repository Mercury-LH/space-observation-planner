"""伐谋混合云的本地评测入口（方案 H 价格引导放置规则，第三次实验：直接在全量实例上评测）。
本文件、数据和评测逻辑都只在本机，不上传云端。"""
import sys
from pathlib import Path

SCHEME_DIR = Path(__file__).resolve().parents[2]
if str(SCHEME_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEME_DIR))

from obsplan.famou_h_eval import evaluate_source  # noqa: E402


def evaluate(program_path):
    return evaluate_source(Path(program_path).read_bytes(), version="full", timeout_s=240)
