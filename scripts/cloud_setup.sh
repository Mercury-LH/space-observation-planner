#!/bin/bash
# Claude Code 云端会话启动时安装方案H初版的依赖；在本机直接退出，不改动本地环境。

if [ "$CLAUDE_CODE_REMOTE" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR/方案H初版" || exit 0

if uv sync --frozen > /tmp/cloud_setup.log 2>&1; then
  echo "云端环境：方案H初版 依赖已安装（uv sync），可直接 uv run python run_compare.py"
else
  echo "云端环境：方案H初版 依赖安装失败，日志见 /tmp/cloud_setup.log"
fi
exit 0
