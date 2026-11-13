"""界面冒烟检查：用 Streamlit 自带的 AppTest 在本进程里跑一遍界面，不开浏览器。

    uv run python scripts/check_app.py

换一个专用的规则场景（规划 1 天、最低仰角 11°），先删掉它的窗口缓存，让界面现算可见窗口，
再用演示模式（5% 目标 × 2 站）运行一次规划，确认运行和结果各页都没有报错。
界面里现算窗口会开进程池：spawn 方式（Windows、macOS）的子进程会重新执行主模块，
而 Streamlit 把界面脚本装成主模块，这一步最容易在评委机器上出问题，所以单独检查。
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from obsplan.config import Config  # noqa: E402

DAYS, ELEV_MIN = 1.0, 11.0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):          # Windows 上输出到管道（如 CI）时默认编码印不出中文
        sys.stdout.reconfigure(encoding="utf-8")
    key = Config(days=DAYS, elev_min=ELEV_MIN).scenario_key()
    for name in (f"windows_{key}.npz", f"short_{key}.npz", f"near_{key}.npz"):
        (ROOT / "cache" / name).unlink(missing_ok=True)
    t0 = time.time()
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=900).run()
    assert not at.exception, at.exception
    sb = at.sidebar
    sb.date_input[0].set_value(date(2026, 8, 1))
    sb.number_input[0].set_value(DAYS)              # 规划天数
    sb.number_input[1].set_value(ELEV_MIN)          # 最低仰角
    sb.slider[2].set_value(0.05)                    # 演示模式：目标比例
    sb.multiselect[0].set_value([1, 2])             # 演示模式：站点
    sb.button[0].click()
    at.run()
    assert not at.exception, at.exception
    assert (ROOT / "cache" / f"windows_{key}.npz").exists(), "界面没有算出可见窗口"
    assert len(at.tabs) >= 9, "结果页没有显示"
    print(f"界面检查通过：现算可见窗口并完成一次演示规划，结果页 {len(at.tabs)} 个标签，用时 {time.time() - t0:.0f} 秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
