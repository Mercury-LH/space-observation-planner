#!/usr/bin/env bash
# 方案 H 一键启动（macOS）：在访达里双击即可，会打开终端窗口运行。
# 安装依赖 → 准备可见窗口 → 打开网页界面；关闭终端窗口即结束。
set -euo pipefail
cd "$(dirname "$0")"

if [ "$(uname -m)" != "arm64" ]; then
  echo "这台 Mac 是 Intel 芯片（或终端在 Rosetta 下运行）。程序用到的 numba 0.68、llvmlite 0.50"
  echo "没有 Intel Mac 的安装包，请改用 Docker 运行，见 README.md 的「评委运行指南」。"
  exit 1
fi

# uv 默认装在 ~/.local/bin，刚装好时当前终端的 PATH 里可能还没有它
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "没有找到 uv（用来安装 Python 3.12 和依赖的工具）。请先在「终端」里运行："
  echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "  或者（已有 Homebrew 时）：brew install uv"
  echo "装好后重新双击本文件。详见 README.md 的「评委运行指南」。"
  exit 1
fi

echo "[1/3] 安装 Python 3.12 和依赖（首次需联网，约 1–5 分钟）……"
uv sync --extra app
echo "[2/3] 准备可见窗口（首次约 0.5–3 分钟，之后直接读缓存）……"
uv run --no-sync python -m obsplan windows
echo "[3/3] 启动界面：浏览器会自动打开 http://localhost:8501 ，关闭本窗口即结束。"
exec uv run --no-sync streamlit run app/streamlit_app.py --server.headless false --server.showEmailPrompt false "$@"
