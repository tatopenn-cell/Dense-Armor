@echo off
rem Installs the Dense-Armor dashboard and puts the icon "Dense-Armor Dashboard" on the desktop.
rem Double-click it. Python 3.12 is installed with winget if the computer has none.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-dashboard.ps1" %*
if errorlevel 1 (
  echo.
  echo The installation did not finish. Read the message above.
)
pause
