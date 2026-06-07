param(
    [string]$CondaEnv = "graph-rag",
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$frontendRoot = Join-Path $projectRoot "frontend"
$logDir = Join-Path $projectRoot "output\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$backendLog = Join-Path $logDir "backend.log"
$frontendLog = Join-Path $logDir "frontend.log"

function Test-Http {
    param(
        [string]$Uri,
        [int]$TimeoutSeconds = 2
    )
    try {
        Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec $TimeoutSeconds | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Test-Port {
    param([int]$Port)
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Wait-ForHttp {
    param(
        [string]$Uri,
        [int]$Seconds = 30
    )
    for ($i = 0; $i -lt $Seconds; $i++) {
        if (Test-Http $Uri 2) {
            return $true
        }
        Start-Sleep -Seconds 1
    }
    return $false
}

Set-Location $projectRoot

Write-Host "Starting ScholarScope"
Write-Host "Project: $projectRoot"
Write-Host "Logs: $logDir"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda was not found. Run scripts/check-env.ps1 for details."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm was not found. Install Node.js first."
}

if (-not (Test-Path -LiteralPath (Join-Path $frontendRoot "node_modules"))) {
    Write-Host "frontend/node_modules is missing. Running npm install..."
    Push-Location $frontendRoot
    npm install
    Pop-Location
}

$backendHealth = "http://127.0.0.1:$BackendPort/health"
$frontendUrl = "http://127.0.0.1:$FrontendPort/"

if (Test-Http $backendHealth 2) {
    Write-Host "Backend is already healthy on port $BackendPort."
} elseif (Test-Port $BackendPort) {
    throw "Port $BackendPort is in use, but backend health check failed. Run scripts/stop.ps1 or free the port."
} else {
    Write-Host "Starting backend on $backendHealth ..."
    $backendCommand = "Set-Location -LiteralPath '$projectRoot'; conda run -n '$CondaEnv' python -m uvicorn backend.app.main:app --host 127.0.0.1 --port $BackendPort *>> '$backendLog'"
    Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $backendCommand) -WindowStyle Hidden
    if (-not (Wait-ForHttp $backendHealth 45)) {
        throw "Backend did not become healthy. See $backendLog"
    }
}

if (Test-Http $frontendUrl 2) {
    Write-Host "Frontend is already available on port $FrontendPort."
} elseif (Test-Port $FrontendPort) {
    throw "Port $FrontendPort is in use, but frontend did not respond. Run scripts/stop.ps1 or free the port."
} else {
    Write-Host "Starting frontend on $frontendUrl ..."
    $frontendCommand = "Set-Location -LiteralPath '$frontendRoot'; npm run dev -- --host 127.0.0.1 --port $FrontendPort *>> '$frontendLog'"
    Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $frontendCommand) -WindowStyle Hidden
    if (-not (Wait-ForHttp $frontendUrl 45)) {
        throw "Frontend did not become available. See $frontendLog"
    }
}

Write-Host ""
Write-Host "ScholarScope is running:"
Write-Host "Frontend: $frontendUrl"
Write-Host "Backend:  $backendHealth"
Write-Host "Logs:"
Write-Host "  $backendLog"
Write-Host "  $frontendLog"

if (-not $NoBrowser) {
    Start-Process $frontendUrl
}
