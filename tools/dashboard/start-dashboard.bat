@echo off
rem Starts the web application of Dense-Armor (the dashboard of the finance system) and opens it in the browser.
rem Double-click it, or run it from any folder:  tools\dashboard\start-dashboard.bat
rem Options are passed on:  start-dashboard.bat --port 9000   or   start-dashboard.bat --enable-playground
setlocal
cd /d "%~dp0..\.."
set "PYTHONPATH=%CD%\tools;%CD%"
set "RUNDIR=%CD%\tools\dashboard\run"
if not exist "%RUNDIR%" mkdir "%RUNDIR%"
if not exist "%RUNDIR%\demo.toml" copy /y "test\fixtures\finance\demo.toml" "%RUNDIR%\demo.toml" >nul

set "PY=python"
if exist "%CD%\.venv\Scripts\python.exe" set "PY=%CD%\.venv\Scripts\python.exe"

"%PY%" -c "import flask" 2>nul
if errorlevel 1 (
  echo Flask is not installed: installing it...
  "%PY%" -m pip install flask
  if errorlevel 1 goto :fail
)

echo.
echo Dense-Armor dashboard on http://127.0.0.1:8080/   (Ctrl+C stops it)
echo The log, the feed and the stop file are in %RUNDIR%
echo.
start "" "http://127.0.0.1:8080/"
"%PY%" -m dashboard.app --config "%RUNDIR%\demo.toml" %*
if errorlevel 1 goto :fail
goto :eof

:fail
echo.
echo The dashboard did not start. Python 3.11 or later and a working "python" in the PATH are needed.
pause
exit /b 1
