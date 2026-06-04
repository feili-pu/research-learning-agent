$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

$envPath = Join-Path $projectRoot ".env"
if (-not (Test-Path -LiteralPath $envPath)) {
    Write-Error ".env file was not found."
}

$config = @{}
Get-Content -LiteralPath $envPath | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
        return
    }
    $key, $value = $line.Split("=", 2)
    $config[$key.Trim()] = $value.Trim()
}

$apiKey = $config["OPENAI_API_KEY"]
$baseUrl = ($config["RLA_OPENAI_BASE_URL"] ?? $config["OPENAI_BASE_URL"]).TrimEnd("/")
$model = $config["RLA_LLM_MODEL"]
$wireApi = $config["RLA_OPENAI_WIRE_API"]
if (-not $wireApi) {
    $wireApi = "responses"
}

Write-Output @{
    has_key = [bool]$apiKey
    base_url = $baseUrl
    model = $model
    wire_api = $wireApi
}

$headers = @{
    Authorization = "Bearer $apiKey"
    "Content-Type" = "application/json"
}

if ($wireApi -eq "chat") {
    $uri = "$baseUrl/chat/completions"
    $body = @{
        model = $model
        messages = @(
            @{
                role = "user"
                content = "用中文回答：接口测试成功了吗？"
            }
        )
    } | ConvertTo-Json -Depth 10
} else {
    $uri = "$baseUrl/responses"
    $body = @{
        model = $model
        input = @(
            @{
                role = "user"
                content = "用中文回答：接口测试成功了吗？"
            }
        )
    } | ConvertTo-Json -Depth 10
}

try {
    Invoke-RestMethod -Uri $uri -Method Post -Headers $headers -Body $body | ConvertTo-Json -Depth 20
} catch {
    Write-Output "REQUEST_FAILED"
    Write-Output $_.Exception.Message
    if ($_.ErrorDetails.Message) {
        Write-Output $_.ErrorDetails.Message
    }
    exit 1
}

