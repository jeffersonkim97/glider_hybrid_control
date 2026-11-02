param()

$ErrorActionPreference = "Stop"

$dqnDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspace = Split-Path -Parent $dqnDirectory
$python = Join-Path $workspace ".venv_p1b\Scripts\python.exe"
$runner = Join-Path $dqnDirectory "phase6_g2_local_sse.py"
$runDirectory = Join-Path $dqnDirectory (
    "runs\phase6_g2_local_sse\final_id_001\dx10m_dpsi5deg_r1"
)
$statusPath = Join-Path $runDirectory "background_process.json"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Phase-6 runner not found: $runner"
}

if (Test-Path -LiteralPath $statusPath) {
    $previous = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    $active = Get-Process -Id ([int]$previous.pid) -ErrorAction SilentlyContinue
    if ($null -ne $active) {
        throw "Phase-6 10 m process is already running with PID $($previous.pid)"
    }
}

New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutLog = Join-Path $runDirectory "phase6_10m_${timestamp}.stdout.log"
$stderrLog = Join-Path $runDirectory "phase6_10m_${timestamp}.stderr.log"

$startArguments = @{
    FilePath = $python
    ArgumentList = @(
        $runner,
        "--terrain-index", "1",
        "--spatial-resolution-m", "10",
        "--heading-spacing-deg", "5",
        "--r-neighbor", "1",
        "--device", "cuda"
    )
    WorkingDirectory = $workspace
    WindowStyle = "Hidden"
    RedirectStandardOutput = $stdoutLog
    RedirectStandardError = $stderrLog
    PassThru = $true
}
$process = Start-Process @startArguments

$status = [ordered]@{
    schema = "p1b-phase6-background-process-v1"
    pid = [int]$process.Id
    started_local = (Get-Date).ToString("o")
    spatial_resolution_m = 10.0
    stdout_log = $stdoutLog
    stderr_log = $stderrLog
    exact_cache = (Join-Path $runDirectory "exact_response_cache.json")
    final_result = (Join-Path $runDirectory "phase6_g2_local_sse_pilot.json")
}
$json = $status | ConvertTo-Json -Depth 4
[System.IO.File]::WriteAllText(
    $statusPath,
    $json + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)

[pscustomobject]$status
