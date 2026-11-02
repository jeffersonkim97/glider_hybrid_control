param()

$ErrorActionPreference = "Stop"
$DqnDir = $PSScriptRoot
$Workspace = Split-Path -Parent $DqnDir
$Python = Join-Path $Workspace ".venv_p1b\Scripts\python.exe"
$RunDir = Join-Path $DqnDir "runs\phase7_neighbor_sweep\orchestration_v1"
$StatusPath = Join-Path $RunDir "status.json"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$Steps = @(
    @{ Name = "local_sse_r2"; Arguments = @("--run-local-sse", "2") },
    @{ Name = "local_sse_r3"; Arguments = @("--run-local-sse", "3") },
    @{ Name = "local_sse_r4"; Arguments = @("--run-local-sse", "4") },
    @{ Name = "local_sse_r5"; Arguments = @("--run-local-sse", "5") },
    @{ Name = "aggregate"; Arguments = @("--aggregate") }
)
$Runner = Join-Path $DqnDir "phase7_neighbor_sweep.py"

function Write-RunStatus {
    param(
        [string]$State,
        [string]$CurrentStep,
        [int]$CompletedSteps,
        [string]$Message = ""
    )
    @{
        schema = "p1b-phase7-neighbor-orchestration-v1"
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
    Write-RunStatus `
        -State "failed" `
        -CurrentStep $Step.Name `
        -CompletedSteps $Index `
        -Message $_.Exception.Message
    throw
}

