@echo off
setlocal
cd /d "%~dp0"

for %%I in ("%~dp0..\..") do set "PROJECT_ROOT=%%~fI"
set "PYTHON=%PROJECT_ROOT%\..\.venv\Scripts\python.exe"

if /I "%~1"=="--server" goto RUN_SERVER

title SignLine - local camera dashboard
echo SignLine camera capture and inference run on this computer.
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "try{$null=Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/status' -TimeoutSec 3; exit 0}catch{exit 1}"
if errorlevel 1 (
    echo Starting the camera server in a separate window...
    start "SignLine - camera server" /D "%~dp0" "%ComSpec%" /k call "%~f0" --server
) else (
    echo Reusing the running camera server at http://127.0.0.1:8765.
)

echo Waiting for the local dashboard at http://127.0.0.1:8765 ...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ready=$false; for($i=0; $i -lt 120; $i++){ try{$r=Invoke-WebRequest -Uri 'http://127.0.0.1:8765/api/status' -UseBasicParsing -TimeoutSec 2; if($r.StatusCode -eq 200){$ready=$true; break}}catch{}; Start-Sleep -Seconds 1 }; if(-not $ready){Write-Error 'The local dashboard did not start within 120 seconds.'; exit 1}"
if errorlevel 1 (
    echo ERROR: Camera server did not start. Check its terminal window.
    pause
    exit /b 1
)

if /I "%~1"=="--public-tunnel" (
    echo WARNING: Anyone with the tunnel URL can view and control this camera dashboard.
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0cloudflare\start-tunnel.ps1"
    set "EXIT_CODE=%ERRORLEVEL%"
    if not "%EXIT_CODE%"=="0" echo ERROR: Cloudflare Tunnel stopped with code %EXIT_CODE%.
    echo The server keeps running in its separate window until you close it with Ctrl+C.
    pause
    exit /b %EXIT_CODE%
)

echo Local dashboard: http://127.0.0.1:8765
echo Add --public-tunnel only when you intend to expose the camera dashboard.
echo Keep the camera server window open; close it with Ctrl+C when finished.
pause
exit /b 0

:RUN_SERVER
title SignLine - camera and model server
if not exist "%PYTHON%" (
    echo ERROR: The shared Python environment was not found at:
    echo   %PYTHON%
    echo Install Python 3.12+ and the packages in tools\phone-camera-prototype\requirements.txt.
    pause
    exit /b 1
)

"%PYTHON%" -c "import cv2, mediapipe" >nul 2>nul
if errorlevel 1 (
    echo ERROR: OpenCV and MediaPipe are missing from the shared Python environment.
    echo Install tools\phone-camera-prototype\requirements.txt into that environment.
    pause
    exit /b 1
)

if not exist "%~dp0models\pose_landmarker_lite.task" (
    echo ERROR: Missing models\pose_landmarker_lite.task
    pause
    exit /b 1
)
if not exist "%~dp0models\hand_landmarker.task" (
    echo ERROR: Missing models\hand_landmarker.task
    pause
    exit /b 1
)
if not exist "%~dp0training_data\models\static_pose_medoids.npz" (
    echo ERROR: Missing training_data\models\static_pose_medoids.npz
    echo Build it with: python static_sign_prototype.py prepare
    echo                python static_sign_prototype.py train
    pause
    exit /b 1
)

echo Camera: index 0 / Media Foundation on the original workstation.
echo Camera: negotiated native resolution, up to 960 pixels for inference.
echo Dashboard: http://127.0.0.1:8765
echo Press Ctrl+C here to stop.
echo.
"%PYTHON%" "%~dp0test.py" --camera 0 --backend msmf --camera-setup --static-demo --speak-recognized --max-width=960 --preview-width=1280 --open-web-ui
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo SignLine camera session stopped with error code %EXIT_CODE%.
) else (
    echo SignLine camera session has stopped.
)
pause
exit /b %EXIT_CODE%
