$ErrorActionPreference = 'Stop'
$cloudflareDir = Join-Path $PSScriptRoot 'phone-camera-prototype\cloudflare'
$tokenPath = Join-Path $cloudflareDir 'update-token.txt'
$workerUrl = 'https://mudra-camera-gateway.aakashtribhuvan2006.workers.dev'

if (-not (Test-Path -LiteralPath $tokenPath)) {
    throw 'The Worker is not configured. Run tools\phone-camera-prototype\cloudflare\configure-worker.ps1 once, then start the judges launcher again.'
}
$workerToken = [IO.File]::ReadAllText($tokenPath).Trim()
if (-not $workerToken) { throw 'The local Worker update token file is empty.' }

$command = Get-Command cloudflared.exe, cloudflared -ErrorAction SilentlyContinue | Select-Object -First 1
$cloudflaredPath = if ($command) { $command.Source } else { Join-Path ${env:ProgramFiles(x86)} 'cloudflared\cloudflared.exe' }
if (-not (Test-Path -LiteralPath $cloudflaredPath -PathType Leaf)) {
    throw 'cloudflared is not installed. Install it with: winget install --id Cloudflare.cloudflared'
}

$ready = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    try {
        $null = Invoke-RestMethod -Uri 'http://127.0.0.1:3000/health' -TimeoutSec 2
        $ready = $true
        break
    } catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) { throw 'The web app did not become ready at http://127.0.0.1:3000. Check the backend and frontend windows.' }

$stdoutPath = Join-Path $env:TEMP ("signline-judges-tunnel-{0}.out" -f [guid]::NewGuid())
$stderrPath = Join-Path $env:TEMP ("signline-judges-tunnel-{0}.err" -f [guid]::NewGuid())
$process = $null
$registered = $false
$failed = $false
try {
    $process = Start-Process -FilePath $cloudflaredPath `
        -ArgumentList @('tunnel', '--url', 'http://127.0.0.1:3000') `
        -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -NoNewWindow

    Write-Host 'Opening a temporary tunnel to the SignLine web app.'
    Write-Host 'The permanent Worker URL will stay the same; keep this window open.'
    while ($true) {
        $process.Refresh()
        if ($process.HasExited) {
            throw "cloudflared exited with code $($process.ExitCode)."
        }
        if ($registered) {
            Start-Sleep -Seconds 2
            continue
        }

        foreach ($logPath in @($stdoutPath, $stderrPath)) {
            if (-not (Test-Path -LiteralPath $logPath)) { continue }
            $logText = Get-Content -Raw -LiteralPath $logPath -ErrorAction SilentlyContinue
            if (-not $logText) { continue }
            $matches = [regex]::Matches($logText, 'https://[a-z0-9-]+\.trycloudflare\.com', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
            if ($matches.Count -eq 0) { continue }
            $tunnelUrl = $matches[$matches.Count - 1].Value
            $tunnelReady = $false
            $lastHealthError = $null
            for ($probe = 0; $probe -lt 30; $probe++) {
                try {
                    $null = Invoke-RestMethod -Uri "$tunnelUrl/health" -TimeoutSec 5
                    $tunnelReady = $true
                    break
                } catch {
                    $lastHealthError = $_.Exception.Message
                    Start-Sleep -Seconds 2
                }
            }
            if (-not $tunnelReady) {
                throw "Quick Tunnel did not reach the SignLine app at $tunnelUrl. Last health-check error: $lastHealthError"
            }

            $body = @{ target = $tunnelUrl } | ConvertTo-Json -Compress
            $updated = Invoke-RestMethod -Uri "$workerUrl/__update" -Method Post `
                -Headers @{ Authorization = "Bearer $workerToken" } `
                -ContentType 'application/json' -Body $body -TimeoutSec 20
            if (-not $updated.updated) { throw 'The Worker did not confirm the tunnel update.' }
            $registered = $true
            Write-Host ''
            Write-Host "Temporary upstream tunnel: $tunnelUrl"
            Write-Host "SignLine judge URL: $workerUrl"
            try { Set-Clipboard -Value $workerUrl; Write-Host 'The judge URL was copied to the clipboard.' } catch { }
            break
        }
        Start-Sleep -Seconds 1
    }
} catch {
    $failed = $true
    Write-Host ''
    Write-Host "Tunnel startup failed: $($_.Exception.Message)"
    if (Test-Path -LiteralPath $stderrPath) {
        Write-Host 'Recent cloudflared diagnostics:'
        Get-Content -LiteralPath $stderrPath -Tail 30 -ErrorAction SilentlyContinue
    }
    Write-Host "Diagnostic logs retained at: $stdoutPath and $stderrPath"
    throw
} finally {
    if ($process -and -not $process.HasExited) { Stop-Process -Id $process.Id; $process.WaitForExit() }
    if (-not $failed) {
        foreach ($logPath in @($stdoutPath, $stderrPath)) {
            if (Test-Path -LiteralPath $logPath) { Remove-Item -LiteralPath $logPath -Force }
        }
    }
}
