$ErrorActionPreference = 'Stop'

$workspace = Split-Path -Parent $PSScriptRoot
$python = Join-Path $workspace '.venv_p1b\Scripts\python.exe'
$trainer = Join-Path $PSScriptRoot 'phase4_multi_terrain.py'
$fullSse = Join-Path $PSScriptRoot 'full_sse_dqn_comparison.py'
$pipelineDir = Join-Path $PSScriptRoot 'runs\phase4_multi_terrain\resolution_pipeline_50m_100m'
$statusPath = Join-Path $pipelineDir 'pipeline_status.json'
$current10mProcessId = 4076
$current10mResult = Join-Path $workspace '3D_0827\figure\phase_4_multi_terrain_dqn\dx10m_dpsi5deg\official_batched_v1\full_sse\full_sse_stepped_pyramid_comparison.json'
$startedAt = (Get-Date).ToString('o')

New-Item -ItemType Directory -Path $pipelineDir -Force | Out-Null

function Write-PipelineStatus {
    param(
        [string]$Stage,
        [string]$Message,
        [double]$Resolution = 0,
        [int]$ExitCode = 0
    )
    $payload = [ordered]@{
        stage = $Stage
        message = $Message
        spatial_resolution_m = if ($Resolution -gt 0) { $Resolution } else { $null }
        exit_code = $ExitCode
        process_id = $PID
        started_at = $startedAt
        updated_at = (Get-Date).ToString('o')
        fixed_controls = [ordered]@{
            heading_spacing_deg = 5
            r_neighbor = 1
            seeds = @(0, 1, 2)
            episodes_per_seed = 60000
            episode_batch_size = 8
            checkpoint_interval_episodes = 1500
            evaluation_interval_episodes = 6000
            device = 'cuda'
            run_label = 'official_batched_v1'
        }
        sequence = @(
            'wait_for_10m_stepped_pyramid_full_sse',
            '50m_preflight',
            '50m_official_training',
            '50m_stepped_pyramid_full_sse',
            '100m_preflight',
            '100m_official_training',
            '100m_stepped_pyramid_full_sse'
        )
    }
    $temporary = "$statusPath.tmp"
    $payload | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

function Invoke-LoggedPython {
    param(
        [string]$Stage,
        [double]$Resolution,
        [string[]]$Arguments
    )
    $safeStage = $Stage -replace '[^A-Za-z0-9_.-]', '_'
    $stdoutPath = Join-Path $pipelineDir "$safeStage.stdout.log"
    $stderrPath = Join-Path $pipelineDir "$safeStage.stderr.log"
    Write-PipelineStatus -Stage $Stage -Resolution $Resolution -Message "Running $Stage."
    $process = Start-Process -FilePath $python -ArgumentList $Arguments `
        -WorkingDirectory $workspace -Wait -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    if ($process.ExitCode -ne 0) {
        throw "$Stage exited with code $($process.ExitCode). See $stderrPath"
    }
}

function Invoke-ResolutionPipeline {
    param([double]$Resolution)

    $tag = "dx$([int]$Resolution)m_dpsi5deg"
    $officialManifest = Join-Path $workspace "3D_0827\figure\phase_4_multi_terrain_dqn\$tag\official_batched_v1\phase4_multi_terrain_manifest.json"
    $fullSseResult = Join-Path $workspace "3D_0827\figure\phase_4_multi_terrain_dqn\$tag\official_batched_v1\full_sse\full_sse_stepped_pyramid_comparison.json"

    if (-not (Test-Path -LiteralPath $officialManifest)) {
        Invoke-LoggedPython -Stage "${Resolution}m_preflight" -Resolution $Resolution -Arguments @(
            $trainer,
            '--smoke',
            '--device', 'cuda',
            '--spatial-resolution-m', "$Resolution",
            '--heading-spacing-deg', '5',
            '--run-label', 'official'
        )
        Invoke-LoggedPython -Stage "${Resolution}m_official_training" -Resolution $Resolution -Arguments @(
            $trainer,
            '--device', 'cuda',
            '--spatial-resolution-m', "$Resolution",
            '--heading-spacing-deg', '5',
            '--run-label', 'official_batched_v1'
        )
    }
    else {
        Write-PipelineStatus -Stage "${Resolution}m_training_already_complete" -Resolution $Resolution -Message "Compatible official manifest already exists; training retained."
    }

    if (-not (Test-Path -LiteralPath $fullSseResult)) {
        Invoke-LoggedPython -Stage "${Resolution}m_stepped_pyramid_full_sse" -Resolution $Resolution -Arguments @(
            $fullSse,
            '--terrain', 'stepped_pyramid',
            '--spatial-resolution-m', "$Resolution",
            '--heading-spacing-deg', '5',
            '--r-neighbor', '1',
            '--run-label', 'official_batched_v1',
            '--device', 'cuda'
        )
    }
    else {
        Write-PipelineStatus -Stage "${Resolution}m_full_sse_already_complete" -Resolution $Resolution -Message "Existing stepped-pyramid full-SSE result retained."
    }
}

try {
    Write-PipelineStatus -Stage 'waiting_for_10m_full_sse' -Message 'Waiting for the active 10 m stepped-pyramid full-SSE process.'
    Wait-Process -Id $current10mProcessId -ErrorAction SilentlyContinue
    if (-not (Test-Path -LiteralPath $current10mResult)) {
        throw 'The 10 m stepped-pyramid process ended without its comparison JSON.'
    }

    Invoke-ResolutionPipeline -Resolution 50
    Invoke-ResolutionPipeline -Resolution 100
    Write-PipelineStatus -Stage 'complete' -Message '50 m and 100 m official training and stepped-pyramid full SSE completed.'
}
catch {
    Write-PipelineStatus -Stage 'failed' -Message $_.Exception.Message -ExitCode 1
    Write-Error $_
    exit 1
}
