@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found in PATH.
    echo Please install Python 3 and restart this launcher.
    pause
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo Node.js or npm was not found in PATH.
    echo Please install Node.js and restart this launcher.
    pause
    exit /b 1
)

REM --- Backend setup ---
set "BACKEND_DIR=%~dp0app\backend"
set "VENV_DIR=%BACKEND_DIR%\.venv"
set "BACKEND_DEPS_FILE=%BACKEND_DIR%\.deps_ok"
set "BACKEND_NEEDS_INSTALL=0"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating Python virtual environment for backend...
    python -m venv "%VENV_DIR%"
    set "BACKEND_NEEDS_INSTALL=1"
)

if not exist "%BACKEND_DEPS_FILE%" (
    set "BACKEND_NEEDS_INSTALL=1"
)

if exist "%BACKEND_DIR%\requirements.txt" if exist "%BACKEND_DEPS_FILE%" (
    powershell -NoProfile -Command "$req = Get-Item '%BACKEND_DIR%\requirements.txt'; $ok = Get-Item '%BACKEND_DEPS_FILE%'; if ($req.LastWriteTimeUtc -gt $ok.LastWriteTimeUtc) { exit 1 } else { exit 0 }" >nul 2>nul
    if errorlevel 1 set "BACKEND_NEEDS_INSTALL=1"
)

if "%BACKEND_NEEDS_INSTALL%"=="1" (
    echo Installing backend dependencies only if required...
    call "%VENV_DIR%\Scripts\activate.bat"
    cd /d "%BACKEND_DIR%"
    python -m pip install --disable-pip-version-check -r requirements.txt
    type nul > "%BACKEND_DEPS_FILE%"
) else (
    echo Backend dependencies are already up to date.
)

REM --- Speech model for the meeting features (downloaded once, not in git; the app also runs without it) ---
"%VENV_DIR%\Scripts\python.exe" "%~dp0tools\setup\download_asr_model.py"
if errorlevel 1 echo Speech model download failed: meeting captions stay off until tools\setup\download_asr_model.py succeeds.

REM --- Frontend setup ---
set "FRONTEND_DIR=%~dp0app\frontend"
set "FRONTEND_DEPS_FILE=%FRONTEND_DIR%\.deps_ok"
set "FRONTEND_NEEDS_INSTALL=0"

if not exist "%FRONTEND_DIR%\node_modules" (
    set "FRONTEND_NEEDS_INSTALL=1"
)

if not exist "%FRONTEND_DEPS_FILE%" (
    set "FRONTEND_NEEDS_INSTALL=1"
)

if exist "%FRONTEND_DIR%\package.json" if exist "%FRONTEND_DEPS_FILE%" (
    powershell -NoProfile -Command "$pkg = Get-Item '%FRONTEND_DIR%\package.json'; $ok = Get-Item '%FRONTEND_DEPS_FILE%'; if ($pkg.LastWriteTimeUtc -gt $ok.LastWriteTimeUtc) { exit 1 } else { exit 0 }" >nul 2>nul
    if errorlevel 1 set "FRONTEND_NEEDS_INSTALL=1"
)

if "%FRONTEND_NEEDS_INSTALL%"=="1" (
    echo Installing frontend dependencies only if required...
    cd /d "%FRONTEND_DIR%"
    npm install --no-fund --no-audit
    type nul > "%FRONTEND_DEPS_FILE%"
) else (
    echo Frontend dependencies are already up to date.
)

REM --- Launch services ---
start "MUDRA Backend" cmd /k "cd /d ""%BACKEND_DIR%"" && ""%VENV_DIR%\Scripts\python.exe"" main.py"

cd /d "%FRONTEND_DIR%"
start "MUDRA Frontend" cmd /k "cd /d ""%FRONTEND_DIR%"" && npm run dev -- --host"

echo.
echo MUDRA project started.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
pause
