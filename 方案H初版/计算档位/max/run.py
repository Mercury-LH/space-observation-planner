"""max 档位入口；算法与规则由公共模块提供。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parent.name))
