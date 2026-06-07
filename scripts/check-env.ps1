param(
    [string]$CondaEnv = "graph-rag",
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$frontendRoot = Join-Path $projectRoot "frontend"
$envPath = Join-Path $projectRoot ".env"

function Write-Check {
    param(
        [string]$Name,
        [bool]$Ok,
        [string]$Detail = ""
    )
    $prefix = if ($Ok) { "[OK]" } else { "[WARN]" }
    $message = "$prefix $Name"
    if ($Detail) {
        $message = "$message - $Detail"
    }
    Write-Host $message
}

function Command-Exists {
    param([string]$Command)
    return [bool](Get-Command $Command -ErrorAction SilentlyContinue)
}

function Test-Port {
    param([int]$Port)
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

Write-Host "ScholarScope environment check"
Write-Host "Project: $projectRoot"
Write-Host ""

$hasConda = Command-Exists "conda"
Write-Check "conda" $hasConda $(if ($hasConda) { (conda --version) } else { "install Anaconda or Miniconda first" })

if ($hasConda) {
    $envList = conda env list | Out-String
    Write-Check "conda env $CondaEnv" ($envList -match "(^|\s)$([regex]::Escape($CondaEnv))\s") "required for backend"
}

$hasNode = Command-Exists "node"
Write-Check "node" $hasNode $(if ($hasNode) { (node --version) } else { "install Node.js first" })

$hasNpm = Command-Exists "npm"
Write-Check "npm" $hasNpm $(if ($hasNpm) { (npm --version) } else { "npm is required for frontend" })

Write-Check ".env" (Test-Path -LiteralPath $envPath) $(if (Test-Path -LiteralPath $envPath) { "found" } else { "optional, copy .env.example if you use an LLM" })

$envConfig = @{}
if (Test-Path -LiteralPath $envPath) {
    Get-Content -LiteralPath $envPath | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }
        $key, $value = $line.Split("=", 2)
        $envConfig[$key.Trim()] = $value.Trim()
    }
}

$hasKey = [bool]($envConfig["OPENAI_API_KEY"] -or $env:OPENAI_API_KEY)
$baseUrl = $envConfig["RLA_OPENAI_BASE_URL"]
if (-not $baseUrl) { $baseUrl = $envConfig["OPENAI_BASE_URL"] }
if (-not $baseUrl) { $baseUrl = $env:RLA_OPENAI_BASE_URL }
if (-not $baseUrl) { $baseUrl = $env:OPENAI_BASE_URL }
$model = $envConfig["RLA_LLM_MODEL"]
if (-not $model) { $model = $env:RLA_LLM_MODEL }

Write-Check "OPENAI_API_KEY" $hasKey $(if ($hasKey) { "configured, value hidden" } else { "LLM features will fall back or be unavailable" })
Write-Check "LLM base URL" ([bool]$baseUrl) $(if ($baseUrl) { $baseUrl } else { "using SDK default if key is configured" })
Write-Check "LLM model" ([bool]$model) $(if ($model) { $model } else { "backend default will be used" })

$hasNodeModules = Test-Path -LiteralPath (Join-Path $frontendRoot "node_modules")
Write-Check "frontend node_modules" $hasNodeModules $(if ($hasNodeModules) { "installed" } else { "run npm install in frontend" })
Write-Check "backend port $BackendPort" (-not (Test-Port $BackendPort)) $(if (Test-Port $BackendPort) { "already in use" } else { "free" })
Write-Check "frontend port $FrontendPort" (-not (Test-Port $FrontendPort)) $(if (Test-Port $FrontendPort) { "already in use" } else { "free" })

Write-Host ""
Write-Host "Run scripts/start.ps1 to launch the app."
