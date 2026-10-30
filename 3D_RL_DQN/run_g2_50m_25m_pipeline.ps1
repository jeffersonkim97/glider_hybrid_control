$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv_p1b\Scripts\python.exe"
$runner = Join-Path $PSScriptRoot "phase4_g2_training.py"
$pipelineRoot = Join-Path $PSScriptRoot "runs\phase4_extended_random_boxes\training_v2\g2_generalization\pipeline_50m_25m_v1"
New-Item -ItemType Directory -Force -Path $pipelineRoot | Out-Null

$conditions = @(50, 25)
foreach ($resolution in $conditions) {
    $label = "dx${resolution}m_dpsi5deg"
    $stdout = Join-Path $pipelineRoot "${label}_stdout.log"
    $stderr = Join-Path $pipelineRoot "${label}_stderr.log"
    $started = Get-Date
    [pscustomobject]@{
        status = "running"
        spatial_resolution_m = $resolution
        heading_spacing_deg = 5
        seed = 0
        episodes = 10000
        started_at = $started.ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $pipelineRoot "${label}_status.json") -Encoding utf8

    & $python $runner `
        --spatial-resolution-m $resolution `
        --heading-spacing-deg 5 `
        --episodes 10000 `
        --evaluation-interval-episodes 1000 `
        1> $stdout 2> $stderr
    if ($LASTEXITCODE -ne 0) {
        [pscustomobject]@{
            status = "failed"
            spatial_resolution_m = $resolution
            exit_code = $LASTEXITCODE
            started_at = $started.ToString("o")
            finished_at = (Get-Date).ToString("o")
            stdout = $stdout
            stderr = $stderr
        } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $pipelineRoot "${label}_status.json") -Encoding utf8
        throw "G2 ${resolution} m training failed with exit code $LASTEXITCODE"
    }

    [pscustomobject]@{
        status = "completed"
        spatial_resolution_m = $resolution
        heading_spacing_deg = 5
        seed = 0
        episodes_maximum = 10000
        started_at = $started.ToString("o")
        finished_at = (Get-Date).ToString("o")
        stdout = $stdout
        stderr = $stderr
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $pipelineRoot "${label}_status.json") -Encoding utf8
}

[pscustomobject]@{
    status = "completed"
    conditions_m = $conditions
    finished_at = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $pipelineRoot "pipeline_complete.json") -Encoding utf8
