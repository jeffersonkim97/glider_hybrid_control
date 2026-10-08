$ErrorActionPreference = 'Stop'

$workspace = Split-Path -Parent $PSScriptRoot
$python = Join-Path $workspace '.venv_p1b\Scripts\python.exe'
$reportScript = Join-Path $PSScriptRoot 'full_sse_resolution_report.py'
$pipelineDir = Join-Path $PSScriptRoot 'runs\phase4_multi_terrain\resolution_pipeline_50m_100m'
$pipelineStatus = Join-Path $pipelineDir 'pipeline_status.json'
$statusPath = Join-Path $pipelineDir 'report_status.json'
$pipelineProcessId = 2788

function Write-ReportStatus([string]$stage, [string]$message, [int]$exitCode = 0) {
    [ordered]@{
        stage = $stage
        message = $message
        exit_code = $exitCode
        process_id = $PID
        updated_at = (Get-Date).ToString('o')
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
}

try {
    Write-ReportStatus 'waiting_for_training_pipeline' 'Waiting for 50 m and 100 m training/full-SSE pipeline.'
    Wait-Process -Id $pipelineProcessId -ErrorAction SilentlyContinue
    $pipeline = Get-Content -LiteralPath $pipelineStatus -Raw | ConvertFrom-Json
    if ($pipeline.stage -ne 'complete') {
        throw "Training pipeline ended with stage '$($pipeline.stage)': $($pipeline.message)"
    }

    Write-ReportStatus 'generating' 'Generating 10/25/50/100 m stepped-pyramid 3D HTML and aggregate report.'
    & $python $reportScript --resolutions 10 25 50 100 --run-label official_batched_v1
    if ($LASTEXITCODE -ne 0) {
        throw "Resolution report exited with code $LASTEXITCODE"
    }
    Write-ReportStatus 'complete' 'All resolution HTML files and the aggregate payoff/time report were generated.'
}
catch {
    Write-ReportStatus 'failed' $_.Exception.Message 1
    Write-Error $_
    exit 1
}
