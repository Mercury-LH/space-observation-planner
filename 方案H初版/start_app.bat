@echo off
chcp 65001 >nul
setlocal
rem 方案 H 一键启动（Windows）：双击运行。安装依赖 → 准备可见窗口 → 打开网页界面；关闭本窗口即结束。
rem 8501 端口被占用时，在命令行里运行：start_app.bat --server.port 8502
cd /d "%~dp0"
set PYTHONUTF8=1

rem uv 默认装在 用户目录\.local\bin，刚装好时当前窗口的 PATH 里可能还没有它
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul || goto :no_uv

echo [1/3] 安装 Python 3.12 和依赖（首次需联网，约 1–5 分钟）……
uv sync --extra app || goto :failed
echo [2/3] 准备可见窗口（首次约 0.5–3 分钟，之后直接读缓存）……
uv run --no-sync python -m obsplan windows || goto :failed
echo [3/3] 启动界面：浏览器会自动打开 http://localhost:8501 ，关闭本窗口即结束。
uv run --no-sync streamlit run app/streamlit_app.py --server.headless false --server.showEmailPrompt false %* || goto :failed
goto :eof

:no_uv
echo 没有找到 uv（用来安装 Python 3.12 和依赖的工具）。请先在 PowerShell 里运行：
echo     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
echo   或者（已有 Python 时）：pip install uv
echo 装好后重新双击 start_app.bat。详见 README.md 的“评委运行指南”。
pause
exit /b 1

:failed
echo.
echo 上一步出错，错误信息见上方。常见问题见 README.md 的“评委运行指南”。
pause
exit /b 1
