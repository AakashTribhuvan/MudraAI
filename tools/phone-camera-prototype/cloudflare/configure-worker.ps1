$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot

try {
    $npx = Get-Command npx.cmd -ErrorAction SilentlyContinue
    if (-not $npx) {
        throw 'Node.js and npm are required to configure the Cloudflare Worker.'
    }

    & $npx.Source wrangler deploy --config wrangler.jsonc
    if ($LASTEXITCODE -ne 0) {
        throw 'Worker deployment failed. Check Wrangler login and Cloudflare account access.'
    }

    $tokenPath = Join-Path $PSScriptRoot 'update-token.txt'
    if (-not (Test-Path -LiteralPath $tokenPath)) {
        $bytes = New-Object byte[] 32
        $generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
        try {
            $generator.GetBytes($bytes)
        } finally {
            $generator.Dispose()
        }
        $token = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
        [IO.File]::WriteAllText(
            $tokenPath,
            $token,
            (New-Object Text.UTF8Encoding($false))
        )
    }

    $token = [IO.File]::ReadAllText($tokenPath).Trim()
    if (-not $token) {
        throw 'The local Worker update token file is empty.'
    }
    $token | & $npx.Source wrangler secret put UPDATE_TOKEN --config wrangler.jsonc
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not configure the Worker update secret.'
    }

    Write-Host ''
    Write-Host 'Password-free visitor URL: https://mudra-camera-gateway.aakashtribhuvan2006.workers.dev'
    Write-Host 'The Worker update token is stored locally in cloudflare\update-token.txt and is not needed by visitors.'
} finally {
    Pop-Location
}
