$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

$currentTunnelPath = Join-Path $PSScriptRoot 'current-tunnel.txt'
$tokenPath = Join-Path $PSScriptRoot 'update-token.txt'
$workerUrl = 'https://mudra-camera-gateway.aakashtribhuvan2006.workers.dev'
$workerToken = if (Test-Path -LiteralPath $tokenPath) {
    [IO.File]::ReadAllText($tokenPath).Trim()
} else {
    $null
}
if (-not $workerToken) {
    throw 'The Worker is not configured. Run cloudflare\configure-worker.ps1 once before starting SignLine.'
}

$dashboardReady = $false
for ($attempt = 0; $attempt -lt 15; $attempt++) {
    try {
        $null = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/status' -TimeoutSec 3
        $dashboardReady = $true
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $dashboardReady) {
    throw 'The local SignLine dashboard is not responding at http://127.0.0.1:8765.'
}

$cloudflaredCommand = Get-Command cloudflared.exe, cloudflared -ErrorAction SilentlyContinue |
    Select-Object -First 1
$cloudflaredPath = if ($cloudflaredCommand) {
    $cloudflaredCommand.Source
} else {
    Join-Path ${env:ProgramFiles(x86)} 'cloudflared\cloudflared.exe'
}
if (-not (Test-Path -LiteralPath $cloudflaredPath -PathType Leaf)) {
    throw 'cloudflared is not installed. Install it with: winget install --id Cloudflare.cloudflared'
}

$stdoutPath = Join-Path $env:TEMP ("mudra-cloudflared-{0}.out" -f [guid]::NewGuid())
$stderrPath = Join-Path $env:TEMP ("mudra-cloudflared-{0}.err" -f [guid]::NewGuid())
$process = $null
$registeredUrl = $null

try {
    $process = Start-Process `
        -FilePath $cloudflaredPath `
        -ArgumentList @('tunnel', '--url', 'http://127.0.0.1:8765') `
        -PassThru `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -NoNewWindow

    Write-Host 'SignLine is starting a temporary Cloudflare Quick Tunnel.'
    Write-Host 'Anyone who has the tunnel URL can open the site while this computer is running.'
    Write-Host 'Keep this window open; closing it stops remote access.'
    while (-not $process.HasExited) {
        foreach ($logPath in @($stdoutPath, $stderrPath)) {
            if (-not (Test-Path -LiteralPath $logPath)) { continue }
            $logText = Get-Content -Raw -LiteralPath $logPath -ErrorAction SilentlyContinue
            if (-not $logText) { continue }
            $urlMatches = [regex]::Matches(
                $logText,
                'https://[a-z0-9-]+\.trycloudflare\.com',
                [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
            )
            if ($urlMatches.Count -eq 0) { continue }
            $tunnelUrl = $urlMatches[$urlMatches.Count - 1].Value
            if ($tunnelUrl -eq $registeredUrl) { continue }

            $tunnelReady = $false
            for ($probe = 0; $probe -lt 30; $probe++) {
                try {
                    $null = Invoke-RestMethod -Uri "$tunnelUrl/api/status" -TimeoutSec 5
                    $tunnelReady = $true
                    break
                } catch {
                    Start-Sleep -Seconds 2
                }
            }
            if (-not $tunnelReady) {
                throw "The Quick Tunnel did not reach the SignLine dashboard at $tunnelUrl."
            }

            $registeredUrl = $tunnelUrl
            [IO.File]::WriteAllText(
                $currentTunnelPath,
                $tunnelUrl,
                (New-Object Text.UTF8Encoding($false))
            )
            Write-Host ''
            $updateBody = @{ target = $tunnelUrl } | ConvertTo-Json -Compress
            $updateResponse = Invoke-RestMethod `
                -Uri "$workerUrl/__update" `
                -Method Post `
                -Headers @{ Authorization = "Bearer $workerToken" } `
                -ContentType 'application/json' `
                -Body $updateBody `
                -TimeoutSec 20
            if (-not $updateResponse.updated) {
                throw 'The Worker did not confirm that its tunnel destination was updated.'
            }
            Write-Host "Temporary tunnel URL: $tunnelUrl"
            Write-Host "Permanent website URL: $workerUrl/model-test"
            try {
                Set-Clipboard -Value "$workerUrl/model-test"
                Write-Host 'The permanent Worker URL was copied to the clipboard.'
            } catch {
                Write-Warning "Could not copy the website URL to the clipboard: $($_.Exception.Message)"
            }
        }
        Start-Sleep -Seconds 1
        $process.Refresh()
    }

    if ($process.ExitCode -ne 0) {
        throw "cloudflared exited with code $($process.ExitCode)."
    }
    if (-not $registeredUrl) {
        throw 'cloudflared stopped before reporting a Quick Tunnel URL.'
    }
} finally {
    if ($process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id
        $process.WaitForExit()
    }
    foreach ($logPath in @($stdoutPath, $stderrPath)) {
        if (Test-Path -LiteralPath $logPath) {
            Remove-Item -LiteralPath $logPath -Force
        }
    }
}
