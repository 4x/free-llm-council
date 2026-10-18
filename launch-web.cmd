@echo off
setlocal
title llm-council - Web UI
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo Error: uv was not found on PATH. Install uv ^(https://docs.astral.sh/uv/^) and try again.
    pause
    exit /b 1
)

uv run llm-council --web --open-browser %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" if not "%EXIT_CODE%"=="130" (
    echo.
    echo llm-council exited with status %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
