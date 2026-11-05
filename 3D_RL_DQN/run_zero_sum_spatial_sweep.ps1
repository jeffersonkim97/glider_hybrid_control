param(
    [double]$MinimumAvailableRamGB = 12.0,
    [double]$MaximumCpuPercent = 25.0,
    [int]$RequiredReadySamples = 2,
    [int]$PollSeconds = 30,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$DqnDir = $PSScriptRoot
$Workspace = Split-Path -Parent $DqnDir
$Python = Join-Path $Workspace ".venv_p1b\Scripts\python.exe"
$Runner = Join-Path $DqnDir "phase6_g2_local_sse.py"
$Aggregator = Join-Path $DqnDir "zero_sum_spatial_sweep.py"
$RunDir = Join-Path $DqnDir "runs\zero_sum_spatial_sweep\orchestration_v1"
$StatusPath = Join-Path $RunDir "status.json"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$Steps = @()
foreach ($Resolution in @(100, 50, 25, 10)) {
    foreach ($Terrain in @(0, 1, 2, 3)) {
        $Name = "terrain_{0:D3}_dx{1}m" -f $Terrain, $Resolution
        $ResultPath = Join-Path $DqnDir (
            "runs\phase6_g2_local_sse\final_id_{0:D3}\dx{1}m_dpsi5deg_r1\zero_sum_g2_local_search.json" -f `
                $Terrain, $Resolution
        )
        $Steps += @{
            Name = $Name
            ResultPath = $ResultPath
            Arguments = @(
                "--terrain-index", [string]$Terrain,
                "--spatial-resolution-m", [string]$Resolution,
                "--heading-spacing-deg", "5",
                "--r-neighbor", "1",
                "--device", "cuda",
                "--leader-objective", "attacker_objective"
            )
        }
    }
}
$Steps += @{
    Name = "aggregate"
    ResultPath = Join-Path $DqnDir (
        "runs\zero_sum_spatial_sweep\closeout_v1\zero_sum_spatial_sweep.json"
    )
    Arguments = @("--aggregate")
    IsAggregate = $true
}

function Test-CompletedResult {
    param([hashtable]$Step)
    if (-not (Test-Path -LiteralPath $Step.ResultPath)) {
        return $false
    }
    if ($Step.IsAggregate) {
        return $true
    }
    try {
        $Payload = Get-Content -LiteralPath $Step.ResultPath -Raw | ConvertFrom-Json
        return (
            $Payload.status -eq "complete" -and
            $Payload.schema -eq "p1b-zero-sum-g2-local-search-v1" -and
            $Payload.condition.leader_objective -eq "attacker_objective"
        )
    }
    catch {
        return $false
    }
}

function Get-SystemLoad {
    $Samples = (Get-Counter -Counter `
        "\Memory\Available MBytes", `
        "\Processor(_Total)\% Processor Time").CounterSamples
    $Available = ($Samples | Where-Object Path -Like "*available mbytes").CookedValue
    $Cpu = ($Samples | Where-Object Path -Like "*processor time").CookedValue
    return @{
        AvailableRamGB = [math]::Round($Available / 1024.0, 3)
        CpuPercent = [math]::Round($Cpu, 2)
    }
}

function Write-RunStatus {
    param(
        [string]$State,
        [string]$CurrentStep,
        [int]$CompletedSteps,
        [double]$AvailableRamGB = 0.0,
        [double]$CpuPercent = 0.0,
        [int]$ReadySamples = 0,
        [string]$Message = ""
    )
    $Payload = @{
        schema = "p1b-zero-sum-spatial-orchestration-v1"
        state = $State
        current_step = $CurrentStep
        completed_steps = $CompletedSteps
        total_steps = $Steps.Count
        process_id = $PID
        readiness = @{
            minimum_available_ram_gb = $MinimumAvailableRamGB
            maximum_cpu_percent = $MaximumCpuPercent
            required_consecutive_samples = $RequiredReadySamples
            current_consecutive_samples = $ReadySamples
            available_ram_gb = $AvailableRamGB
            cpu_percent = $CpuPercent
        }
        condition_contract = @{
            terrain_indices = @(0, 1, 2, 3)
            spatial_resolutions_m = @(100, 50, 25, 10)
            heading_spacing_deg = 5
            r_neighbor = 1
            leader_objective = "attacker_objective"
            frozen_checkpoints = $true
            sequential_execution = $true
            new_training = $false
        }
        updated_at = (Get-Date).ToString("o")
        message = $Message
    }
    $Temporary = $StatusPath + ".tmp"
    $Payload | ConvertTo-Json -Depth 6 | Out-File -LiteralPath $Temporary -Encoding utf8
    Move-Item -LiteralPath $Temporary -Destination $StatusPath -Force
}

if ($PlanOnly) {
    Write-Output ("Zero-sum spatial sweep: {0} simulation steps + aggregate" -f ($Steps.Count - 1))
    foreach ($Step in $Steps) {
        $Disposition = if (Test-CompletedResult $Step) { "skip" } else { "run" }
        Write-Output ("{0}: {1}" -f $Disposition, $Step.Name)
    }
    exit 0
}

$ReadySamples = 0
$CompletedSteps = 0
$Step = $null
try {
    while ($ReadySamples -lt $RequiredReadySamples) {
        $Load = Get-SystemLoad
        $Ready = (
            $Load.AvailableRamGB -ge $MinimumAvailableRamGB -and
            $Load.CpuPercent -le $MaximumCpuPercent
        )
        if ($Ready) {
            $ReadySamples += 1
        }
        else {
            $ReadySamples = 0
        }
        Write-RunStatus `
            -State "waiting_for_idle_resources" `
            -CurrentStep "resource_gate" `
            -CompletedSteps 0 `
            -AvailableRamGB $Load.AvailableRamGB `
            -CpuPercent $Load.CpuPercent `
            -ReadySamples $ReadySamples
        if ($ReadySamples -lt $RequiredReadySamples) {
            Start-Sleep -Seconds $PollSeconds
        }
    }

    for ($Index = 0; $Index -lt $Steps.Count; $Index++) {
        $Step = $Steps[$Index]
        if (Test-CompletedResult $Step) {
            $CompletedSteps = $Index + 1
            Write-RunStatus `
                -State "running" `
                -CurrentStep ("skip_existing_" + $Step.Name) `
                -CompletedSteps $CompletedSteps `
                -Message "Validated existing result; no overwrite."
            continue
        }

        $LogPath = Join-Path $RunDir ($Step.Name + ".log")
        $ErrorLogPath = Join-Path $RunDir ($Step.Name + ".stderr.log")
        $Load = Get-SystemLoad
        Write-RunStatus `
            -State "running" `
            -CurrentStep $Step.Name `
            -CompletedSteps $Index `
            -AvailableRamGB $Load.AvailableRamGB `
            -CpuPercent $Load.CpuPercent

        $CommandArguments = if ($Step.IsAggregate) {
            @($Aggregator) + @($Step.Arguments)
        }
        else {
            @($Runner) + @($Step.Arguments)
        }
        $Child = Start-Process `
            -FilePath $Python `
            -ArgumentList $CommandArguments `
            -WindowStyle Hidden `
            -RedirectStandardOutput $LogPath `
            -RedirectStandardError $ErrorLogPath `
            -Wait `
            -PassThru
        if ($Child.ExitCode -ne 0) {
            $ErrorText = if (Test-Path -LiteralPath $ErrorLogPath) {
                Get-Content -LiteralPath $ErrorLogPath -Raw
            }
            else {
                "No Python stderr log was created."
            }
            throw "Step $($Step.Name) failed with exit code $($Child.ExitCode):`n$ErrorText"
        }
        if (-not (Test-CompletedResult $Step)) {
            throw "Step $($Step.Name) did not create a valid expected result."
        }
        $CompletedSteps = $Index + 1
    }

    $Load = Get-SystemLoad
    Write-RunStatus `
        -State "complete" `
        -CurrentStep "none" `
        -CompletedSteps $Steps.Count `
        -AvailableRamGB $Load.AvailableRamGB `
        -CpuPercent $Load.CpuPercent
}
catch {
    $FailedStep = if ($null -eq $Step) { "resource_gate" } else { $Step.Name }
    $Load = Get-SystemLoad
    Write-RunStatus `
        -State "failed" `
        -CurrentStep $FailedStep `
        -CompletedSteps $CompletedSteps `
        -AvailableRamGB $Load.AvailableRamGB `
        -CpuPercent $Load.CpuPercent `
        -Message $_.Exception.Message
    throw
}

