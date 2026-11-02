param()

$ErrorActionPreference = "Stop"
$DqnDir = $PSScriptRoot
$Workspace = Split-Path -Parent $DqnDir
$Python = Join-Path $Workspace ".venv_p1b\Scripts\python.exe"
$RunDir = Join-Path $DqnDir "runs\phase7_angular_sweep\orchestration_v1"
$StatusPath = Join-Path $RunDir "status.json"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$Steps = @(
    @{
        Name = "train_dpsi10"
        Script = Join-Path $DqnDir "phase4_g2_training.py"
        Arguments = @(
            "--spatial-resolution-m", "25",
            "--heading-spacing-deg", "10",
            "--episodes", "10000",
            "--evaluation-interval-episodes", "1000"
        )
    },
    @{
        Name = "local_sse_dpsi10"
        Script = Join-Path $DqnDir "phase7_angular_sweep.py"
        Arguments = @("--run-local-sse", "10")
    },
    @{
        Name = "train_dpsi2p5"
        Script = Join-Path $DqnDir "phase4_g2_training.py"
        Arguments = @(
            "--spatial-resolution-m", "25",
            "--heading-spacing-deg", "2.5",
            "--episodes", "10000",
            "--evaluation-interval-episodes", "1000"
        )
    },
    @{
        Name = "local_sse_dpsi2p5"
        Script = Join-Path $DqnDir "phase7_angular_sweep.py"
        Arguments = @("--run-local-sse", "2.5")
    },
    @{
        Name = "aggregate"
        Script = Join-Path $DqnDir "phase7_angular_sweep.py"
        Arguments = @("--aggregate")
    }
)

function Write-RunStatus {
    param(
        [string]$State,
        [string]$CurrentStep,
        [int]$CompletedSteps,
        [string]$Message = ""
    )
    @{
        schema = "p1b-phase7-angular-orchestration-v1"
        state = $State
        current_step = $CurrentStep
        completed_steps = $CompletedSteps
        total_steps = $Steps.Count
        process_id = $PID
        updated_at = (Get-Date).ToString("o")
        message = $Message
    } | ConvertTo-Json | Out-File -LiteralPath $StatusPath -Encoding utf8
}
Write-RunStatus -State "running" -CurrentStep $Steps[0].Name -CompletedSteps 0

try {
    for ($Index = 0; $Index -lt $Steps.Count; $Index++) {
        $Step = $Steps[$Index]
        $LogPath = Join-Path $RunDir ($Step.Name + ".log")
        Write-RunStatus `
            -State "running" `
            -CurrentStep $Step.Name `
            -CompletedSteps $Index
        $StepArguments = @($Step.Arguments)
        & $Python $Step.Script @StepArguments *> $LogPath
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
    Write-RunStatus `
        -State "failed" `
        -CurrentStep $Step.Name `
        -CompletedSteps $Index `
        -Message $_.Exception.Message
    throw
}

