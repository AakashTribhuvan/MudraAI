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

REM --- Backend setup ---
set "BACKEND_DIR=%~dp0app\backend"
set "VENV_DIR=%BACKEND_DIR%\.venv-runtime"
set "BACKEND_DEPS_FILE=%BACKEND_DIR%\.deps_ok"
set "BACKEND_NEEDS_INSTALL=0"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating Python virtual environment for backend...
    where py >nul 2>nul
    if errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,12) else 1)"
        if errorlevel 1 (
            echo ERROR: Python 3.12 or newer is required. Install Python 3.12+ and run this launcher again.
            pause
            exit /b 1
        )
        python -m venv "%VENV_DIR%"
    ) else (
        py -3.14 -m venv "%VENV_DIR%" 2>nul
        if errorlevel 1 py -3.13 -m venv "%VENV_DIR%" 2>nul
        if errorlevel 1 py -3.12 -m venv "%VENV_DIR%" 2>nul
        if errorlevel 1 (
            echo ERROR: Python 3.12 or newer is required. Install Python 3.12+ and run this launcher again.
            pause
            exit /b 1
        )
    )
    set "BACKEND_NEEDS_INSTALL=1"
)

if not exist "%BACKEND_DEPS_FILE%" (
    set "BACKEND_NEEDS_INSTALL=1"
)

if exist "%BACKEND_DIR%\requirements-runtime.txt" if exist "%BACKEND_DEPS_FILE%" (
    powershell -NoProfile -Command "$req = Get-Item '%BACKEND_DIR%\requirements-runtime.txt'; $ok = Get-Item '%BACKEND_DEPS_FILE%'; if ($req.LastWriteTimeUtc -gt $ok.LastWriteTimeUtc) { exit 1 } else { exit 0 }" >nul 2>nul
    if errorlevel 1 set "BACKEND_NEEDS_INSTALL=1"
)

REM A stale .deps_ok marker must not hide a broken/missing app runtime (e.g. FastAPI absent).
if exist "%VENV_DIR%\Scripts\python.exe" (
    "%VENV_DIR%\Scripts\python.exe" -c "import fastapi, uvicorn, pydantic, multipart, sqlalchemy, alembic, numpy, sklearn, joblib, yaml, torch, mudra_ml, google.auth, dotenv" >nul 2>nul
    if errorlevel 1 set "BACKEND_NEEDS_INSTALL=1"
)

if "%BACKEND_NEEDS_INSTALL%"=="1" (
    echo Installing SignLine runtime dependencies for CPU inference; first setup can take several minutes...
    call "%VENV_DIR%\Scripts\activate.bat"
    cd /d "%BACKEND_DIR%"
    python -m pip install --disable-pip-version-check -r requirements-runtime.txt
    if errorlevel 1 (
        echo ERROR: Could not install SignLine runtime dependencies. Check the network and Python installation.
        pause
        exit /b 1
    )
    python -m pip install --disable-pip-version-check torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo ERROR: Could not install the CPU PyTorch runtime. Check the network and Python version.
        pause
        exit /b 1
    )
    "%VENV_DIR%\Scripts\python.exe" -c "import fastapi, uvicorn, pydantic, multipart, sqlalchemy, alembic, numpy, sklearn, joblib, yaml, torch, mudra_ml, google.auth, dotenv" >nul 2>nul
    if errorlevel 1 (
        echo ERROR: Runtime dependencies are still incomplete; see the setup output above.
        pause
        exit /b 1
    )
    type nul > "%BACKEND_DEPS_FILE%"
) else (
    echo Backend dependencies are already up to date.
)

REM Backend tunnel mode intentionally skips all Node/Electron/frontend setup.
if /i "%~1"=="backend-tunnel" goto BACKEND_TUNNEL

where npm >nul 2>nul
if errorlevel 1 (
    echo Node.js or npm was not found in PATH.
    echo Please install Node.js and restart this launcher.
    pause
    exit /b 1
)

REM --- Speech model for the meeting features (downloaded once, not in git; the app also runs without it) ---
"%VENV_DIR%\Scripts\python.exe" "%~dp0tools\setup\download_asr_model.py"
if errorlevel 1 echo Speech model download failed: meeting captions stay off until tools\setup\download_asr_model.py succeeds.

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

REM Verify the Electron development toolchain too; node_modules can be partially installed.
if exist "%FRONTEND_DIR%\node_modules" (
    cd /d "%FRONTEND_DIR%"
    node -e "for (const name of ['vite','electron','electron-builder','concurrently','cross-env','wait-on']) require.resolve(name)" >nul 2>nul
    if errorlevel 1 set "FRONTEND_NEEDS_INSTALL=1"
    if not exist "%FRONTEND_DIR%\node_modules\electron\dist\electron.exe" set "FRONTEND_NEEDS_INSTALL=1"
)

if "%FRONTEND_NEEDS_INSTALL%"=="1" (
    echo Installing frontend dependencies only if required...
    cd /d "%FRONTEND_DIR%"
    npm install --no-fund --no-audit
    if errorlevel 1 (
        echo ERROR: Could not install frontend dependencies. Check the network and Node.js installation.
        pause
        exit /b 1
    )
    if not exist "%FRONTEND_DIR%\node_modules\electron\dist\electron.exe" (
        echo ERROR: Electron's desktop runtime was not downloaded. Check the network and rerun this launcher.
        pause
        exit /b 1
    )
    type nul > "%FRONTEND_DEPS_FILE%"
) else (
    echo Frontend dependencies are already up to date.
)

if /i "%~1"=="electron" (
    goto ELECTRON_APP
)

REM The normal launcher starts the full judges-ready web app and tunnel.
goto JUDGES_TUNNEL

:JUDGES_TUNNEL
echo.
echo Starting SignLine: backend + web frontend + public tunnel.
echo.
powershell -NoProfile -Command "try { $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2; if ($health.model_loaded -and $health.model -eq 'isl_mudra_combined_v15b_bilstm') { exit 0 } else { exit 2 } } catch { exit 1 }" >nul 2>nul
if errorlevel 2 (
    echo ERROR: A backend is already running on port 8000 with a different model.
    echo Close its MUDRA Backend window, then run this launcher again to load v15b.
    pause
    exit /b 1
)
if errorlevel 1 (
    start "MUDRA Backend" cmd /k "cd /d ""%BACKEND_DIR%"" && ""%VENV_DIR%\Scripts\python.exe"" main.py"
    echo Waiting for the backend and v15b model to become ready...
    powershell -NoProfile -Command "$ready = $false; for ($i = 0; $i -lt 180; $i++) { try { $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2; if ($health.status -eq 'ok' -and $health.model_loaded -and $health.model -eq 'isl_mudra_combined_v15b_bilstm') { $ready = $true; break } } catch { }; Start-Sleep -Seconds 1 }; if (-not $ready) { exit 1 }"
    if errorlevel 1 (
        echo ERROR: Backend did not become ready on port 8000 within 180 seconds.
        echo Check the MUDRA Backend window for startup errors.
        pause
        exit /b 1
    )
) else (
    echo Reusing the backend already running on port 8000.
)

powershell -NoProfile -Command "$listener = Get-NetTCPConnection -State Listen -LocalPort 3000 -ErrorAction SilentlyContinue | Select-Object -First 1; if ($listener) { exit 1 } else { exit 0 }" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Port 3000 is already in use.
    echo Close the existing frontend window before starting another tunnel session.
    pause
    exit /b 1
)

set "MUDRA_PUBLIC_TUNNEL=1"
set "VITE_API_URL="
start "MUDRA Frontend" cmd /k "cd /d ""%FRONTEND_DIR%"" && npm run dev -- --host 0.0.0.0 --port 3000 --strictPort"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_judges_tunnel.ps1"
if errorlevel 1 (
    echo.
    echo ERROR: The public tunnel stopped. Review the diagnostics above before retrying.
    pause
    exit /b 1
)
exit /b 0

:ELECTRON_APP
start "MUDRA Backend" cmd /k "cd /d ""%BACKEND_DIR%"" && ""%VENV_DIR%\Scripts\python.exe"" main.py"
cd /d "%FRONTEND_DIR%"
echo Starting SignLine desktop app. Keep the backend window open while you use it.
set "ELECTRON_RUN_AS_NODE="
npm run electron:dev
exit /b

:BACKEND_TUNNEL
echo.
echo Starting the backend-only launcher. The public tunnel will expose the API on port 8000.
powershell -NoProfile -Command "try { Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>nul
if errorlevel 1 (
    echo Starting FastAPI backend...
    start "MUDRA Backend" cmd /k "cd /d ""%BACKEND_DIR%"" && ""%VENV_DIR%\Scripts\python.exe"" main.py"
    echo Waiting for the backend to respond...
    powershell -NoProfile -Command "$ready = $false; for ($i = 0; $i -lt 60; $i++) { try { Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2 | Out-Null; $ready = $true; break } catch { Start-Sleep -Seconds 1 } }; if (-not $ready) { exit 1 }"
    if errorlevel 1 (
        echo ERROR: Backend did not become ready on port 8000 within 60 seconds.
        echo Check the MUDRA Backend window for startup errors.
        pause
        exit /b 1
    )
) else (
    echo FastAPI backend is already responding on port 8000; reusing it.
)

where cloudflared >nul 2>nul
if errorlevel 1 (
    if exist "%ProgramFiles(x86)%\cloudflared\cloudflared.exe" (
        set "CLOUDFLARED=%ProgramFiles(x86)%\cloudflared\cloudflared.exe"
    ) else if exist "%ProgramFiles%\cloudflared\cloudflared.exe" (
        set "CLOUDFLARED=%ProgramFiles%\cloudflared\cloudflared.exe"
    ) else (
        echo.
        echo Backend is running, but cloudflared is not installed.
        echo Install it with: winget install --id Cloudflare.cloudflared
        pause
        exit /b 1
    )
) else (
    set "CLOUDFLARED=cloudflared"
)

echo.
echo Opening Cloudflare Quick Tunnel to http://127.0.0.1:8000
echo Keep this window open while you use the tunnel. Press Ctrl+C to stop it.
"%CLOUDFLARED%" tunnel --url http://127.0.0.1:8000
exit /b %errorlevel%
