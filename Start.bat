@echo off
REM Double-click this to start Vehicle Counter on Windows.
title Vehicle Counter

cd /d "%~dp0"

echo.
echo   Vehicle Counter
echo   ===============
echo.

python -c "import flask, cv2, ultralytics" 2>nul
if errorlevel 1 (
  echo   First run - installing dependencies. This takes a few minutes.
  echo.
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo   Install failed. Check that Python 3.9+ is installed and on PATH.
    pause
    exit /b 1
  )
  echo.
)

echo   Starting on http://127.0.0.1:8000
echo   Close this window to stop.
echo.

REM Give the server a moment to bind before opening the browser
start "" /b cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8000/"

python -u app.py
pause
