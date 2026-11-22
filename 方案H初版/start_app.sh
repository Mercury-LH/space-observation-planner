#!/usr/bin/env bash
# 方案 H 一键启动（Linux）：安装依赖 → 准备可见窗口 → 打开网页界面。
#   bash start_app.sh                          # 在任意目录运行都可以
#   bash start_app.sh --server.port 8502       # 8501 端口被占用时换一个
set -euo pipefail
cd "$(dirname "$0")"

# uv 默认装在 ~/.local/bin，刚装好时当前终端的 PATH 里可能还没有它
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "没有找到 uv（用来安装 Python 3.12 和依赖的工具）。请先安装："
  echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "  或者（已有 Python 时）：pip install uv"
  echo "装好后重新运行本脚本。详见 README.md 的「评委运行指南」。"
  exit 1
fi

echo "[1/3] 安装 Python 3.12 和依赖（首次需联网，约 1–5 分钟）……"
uv sync --extra app
echo "[2/3] 准备可见窗口（首次约 0.5–3 分钟，之后直接读缓存）……"
uv run --no-sync python -m obsplan windows
echo "[3/3] 启动界面：浏览器会自动打开 http://localhost:8501 ，按 Ctrl+C 结束。"
exec uv run --no-sync streamlit run app/streamlit_app.py --server.headless false --server.showEmailPrompt false "$@"
