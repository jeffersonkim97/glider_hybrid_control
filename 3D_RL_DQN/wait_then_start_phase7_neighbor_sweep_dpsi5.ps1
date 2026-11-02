param()

$ErrorActionPreference = "Stop"
$DqnDir = $PSScriptRoot
$Workspace = Split-Path -Parent $DqnDir
$Python = Join-Path $Workspace ".venv_p1b\Scripts\python.exe"
$Runner = Join-Path $DqnDir "phase7_neighbor_sweep.py"
$RunDir = Join-Path $DqnDir "runs\phase7_neighbor_sweep_dpsi5deg\orchestration_v1"
$StatusPath = Join-Path $RunDir "status.json"
$TrialCommandFragments = @(
    "Final_10302026",
    "baselines_fw.py",
    "runs/long10"
)

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$Steps = @(
    @{ Name = "local_sse_r2"; Arguments = @("--heading-spacing-deg", "5", "--run-local-sse", "2") },
    @{ Name = "local_sse_r3"; Arguments = @("--heading-spacing-deg", "5", "--run-local-sse", "3") },
    @{ Name = "local_sse_r4"; Arguments = @("--heading-spacing-deg", "5", "--run-local-sse", "4") },
    @{ Name = "local_sse_r5"; Arguments = @("--heading-spacing-deg", "5", "--run-local-sse", "5") },
    @{ Name = "aggregate"; Arguments = @("--heading-spacing-deg", "5", "--aggregate") }
)

function Get-ActiveExternalTrial {
    return @(
        Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
            Where-Object {
                $Command = [string]$_.CommandLine
                $MatchesAll = $true
                foreach ($Fragment in $TrialCommandFragments) {
                    if (-not $Command.Contains($Fragment)) {
                        $MatchesAll = $false
                        break
                    }
                }
                $MatchesAll
            }
    )
}
function Write-RunStatus {
    param(
        [string]$State,
        [string]$CurrentStep,
        [int]$CompletedSteps,
        [int[]]$WaitingForProcessIds = @(),
        [string]$Message = ""
    )
    @{
        schema = "p1b-phase7-neighbor-dpsi5-orchestration-v1"
        state = $State
        current_step = $CurrentStep
        completed_steps = $CompletedSteps
        total_steps = $Steps.Count
        process_id = $PID
        waiting_for_external_process_ids = @($WaitingForProcessIds)
        external_trial_command_fragments = @($TrialCommandFragments)
        raw_data_policy = "preserve per-radius JSON, Markdown, exact cache, and log"
        updated_at = (Get-Date).ToString("o")
        message = $Message
    } | ConvertTo-Json | Out-File -LiteralPath $StatusPath -Encoding utf8
}

try {
    while ($true) {
        $ActiveTrial = @(Get-ActiveExternalTrial)
        if ($ActiveTrial.Count -eq 0) {
            break
        }
        Write-RunStatus `
            -State "waiting_for_external_trial" `
            -CurrentStep "wait_for_Final_10302026_long10" `
            -CompletedSteps 0 `
            -WaitingForProcessIds @($ActiveTrial.ProcessId)
        Start-Sleep -Seconds 30
    }

    for ($Index = 0; $Index -lt $Steps.Count; $Index++) {
        $Step = $Steps[$Index]
        $LogPath = Join-Path $RunDir ($Step.Name + ".log")
        Write-RunStatus `
            -State "running" `
            -CurrentStep $Step.Name `
            -CompletedSteps $Index
        $StepArguments = @($Step.Arguments)
        & $Python $Runner @StepArguments *> $LogPath
        if ($LASTEXITCODE -ne 0) {
            throw "Step $($Step.Name) failed with exit code $LASTEXITCODE"
        }
    }
    Write-RunStatus `
        -State "complete" `
        -CurrentStep "none" `
        -CompletedSteps $Steps.Count
}
catch {
    $FailedStep = if ($null -eq $Step) { "wait_or_setup" } else { $Step.Name }
    $FailedIndex = if ($null -eq $Index) { 0 } else { $Index }
    Write-RunStatus `
        -State "failed" `
        -CurrentStep $FailedStep `
        -CompletedSteps $FailedIndex `
        -Message $_.Exception.Message
    throw
}

