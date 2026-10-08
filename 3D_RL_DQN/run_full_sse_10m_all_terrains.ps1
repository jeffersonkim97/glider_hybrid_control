$ErrorActionPreference = 'Stop'

$workspace = Split-Path -Parent $PSScriptRoot
$python = Join-Path $workspace '.venv_p1b\Scripts\python.exe'
$comparisonScript = Join-Path $PSScriptRoot 'full_sse_dqn_comparison.py'
$aggregateScript = Join-Path $PSScriptRoot 'aggregate_full_sse_results.py'
$outputDir = Join-Path $workspace '3D_0827\figure\phase_4_multi_terrain_dqn\dx10m_dpsi5deg\official_batched_v1\full_sse'
$statusPath = Join-Path $outputDir 'full_sse_all_terrains_status.json'
$terrains = @(
    'centered_cube',
    'centered_cube_half_height',
    'offset_cube_left',
    'offset_cube_right',
    'stepped_pyramid'
)

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$startedAt = (Get-Date).ToString('o')
$completed = [System.Collections.Generic.List[string]]::new()

function Write-Status([string]$stage, [string]$terrain, [int]$exitCode, [string]$message) {
    $payload = [ordered]@{
        stage = $stage
        current_terrain = $terrain
        completed_terrains = @($completed)
        exit_code = $exitCode
        message = $message
        process_id = $PID
        started_at = $startedAt
        updated_at = (Get-Date).ToString('o')
        condition = [ordered]@{
            spatial_resolution_m = 10
            heading_spacing_deg = 5
            r_neighbor = 1
            run_label = 'official_batched_v1'
        }
    }
    $temporary = "$statusPath.tmp"
    $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

Write-Status 'running' $terrains[0] 0 'Starting 10 m full Local-SSE terrain sweep.'

foreach ($terrain in $terrains) {
    $resultPath = Join-Path $outputDir "full_sse_${terrain}_comparison.json"
    if (Test-Path -LiteralPath $resultPath) {
        $completed.Add($terrain)
        Write-Status 'running' $terrain 0 "Existing completed result retained for $terrain."
        continue
    }

    $stdoutPath = Join-Path $outputDir "full_sse_${terrain}.stdout.log"
    $stderrPath = Join-Path $outputDir "full_sse_${terrain}.stderr.log"
    Write-Status 'running' $terrain 0 "Running full Local-SSE for $terrain."
    $arguments = @(
        $comparisonScript,
        '--terrain', $terrain,
        '--spatial-resolution-m', '10',
        '--heading-spacing-deg', '5',
        '--r-neighbor', '1',
        '--run-label', 'official_batched_v1',
        '--device', 'cuda'
    )
    $process = Start-Process -FilePath $python -ArgumentList $arguments -Wait -PassThru `
        -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    if ($process.ExitCode -ne 0) {
        Write-Status 'failed' $terrain $process.ExitCode "Full Local-SSE failed for $terrain."
        exit $process.ExitCode
    }
    $completed.Add($terrain)
    & $python $aggregateScript --full-sse-dir $outputDir --allow-partial
    if ($LASTEXITCODE -ne 0) {
        Write-Status 'failed' $terrain $LASTEXITCODE 'Partial aggregation failed.'
        exit $LASTEXITCODE
    }
}

& $python $aggregateScript --full-sse-dir $outputDir
if ($LASTEXITCODE -ne 0) {
    Write-Status 'failed' '' $LASTEXITCODE 'Final aggregation failed.'
    exit $LASTEXITCODE
}
Write-Status 'complete' '' 0 'All five 10 m terrain full Local-SSE comparisons completed.'
