param(
    [int[]]$Ports = @(8000, 5173)
)

$ErrorActionPreference = "Stop"

foreach ($port in $Ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if (-not $connections) {
        Write-Host "Port $port is not listening."
        continue
    }

    $processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($processId in $processIds) {
        try {
            $process = Get-Process -Id $processId -ErrorAction Stop
            Write-Host "Stopping port $port process $processId ($($process.ProcessName))"
            Stop-Process -Id $processId -Force
        } catch {
            Write-Warning "Could not stop process $processId for port ${port}: $($_.Exception.Message)"
        }
    }
}
