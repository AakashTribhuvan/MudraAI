# Makes Windows give Google Chrome the high-performance GPU (the NVIDIA / AMD card on a laptop that also has on-chip
# graphics). A web page cannot choose the GPU itself; on the development laptop hand tracking went from 46 ms to 17 ms
# per frame with it. Same as Windows Settings > System > Display > Graphics > Google Chrome > High performance.
# Takes effect after Chrome is fully restarted (chrome://restart). Uses more battery when unplugged.
#
#   powershell -ExecutionPolicy Bypass -File tools\setup\chrome_gpu_high_performance.ps1          # set
#   powershell -ExecutionPolicy Bypass -File tools\setup\chrome_gpu_high_performance.ps1 -Undo    # back to Windows' choice
param([switch]$Undo, [string]$Chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe")

$key = "HKCU:\Software\Microsoft\DirectX\UserGpuPreferences"
if (-not (Test-Path $Chrome)) { Write-Error "Chrome not found at $Chrome (pass -Chrome <path to chrome.exe>)"; exit 1 }
if ($Undo) {
    Remove-ItemProperty -Path $key -Name $Chrome -ErrorAction SilentlyContinue
    "Chrome GPU preference removed (Windows decides again). Restart Chrome: chrome://restart"
    exit 0
}
if (-not (Test-Path $key)) { New-Item -Path $key -Force | Out-Null }
New-ItemProperty -Path $key -Name $Chrome -Value "GpuPreference=2;" -PropertyType String -Force | Out-Null
"Chrome set to the high-performance GPU. Restart Chrome to apply it: chrome://restart"
