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
$Runner = Join-Path $DqnDir "zero_sum_multi_axis_sweep.py"
$RunDir = Join-Path $DqnDir "runs\zero_sum_multi_axis_sweep\orchestration_v1"
$StatusPath = Join-Path $RunDir "status.json"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$Steps = @()
foreach ($Heading in @(10, 2.5)) {
    foreach ($Terrain in @(0, 1, 2, 3)) {
        $Root = if ($Terrain -eq 1) { "phase7_angular_sweep" } else { "phase9_multi_axis_generalization\angular" }
        $Tag = ("{0:g}" -f $Heading)
        $Steps += @{
            Name = "angular_terrain_{0:D3}_dpsi{1}" -f $Terrain, $Tag
            ResultPath = Join-Path $DqnDir (
                "runs\{0}\final_id_{1:D3}\dx25m_dpsi{2}deg_r1\zero_sum_angular_local_search.json" -f $Root, $Terrain, $Tag
            )
            Arguments = @("--run-angular", "--terrain-index", [string]$Terrain, "--heading-spacing-deg", [string]$Heading)
            ExpectedSchema = "p1b-zero-sum-angular-local-search-v1"
        }
    }
}
foreach ($Radius in @(3, 5)) {
    foreach ($Terrain in @(0, 1, 2, 3)) {
        $Root = if ($Terrain -eq 1) { "phase7_neighbor_sweep_dpsi5deg" } else { "phase9_multi_axis_generalization\neighbor_dpsi5deg" }
        $Steps += @{
            Name = "neighbor_terrain_{0:D3}_r{1}" -f $Terrain, $Radius
            ResultPath = Join-Path $DqnDir (
                "runs\{0}\final_id_{1:D3}\dx25m_dpsi5deg_r{2}\zero_sum_neighbor_local_search.json" -f $Root, $Terrain, $Radius
            )
            Arguments = @("--run-neighbor", "--terrain-index", [string]$Terrain, "--r-neighbor", [string]$Radius)
            ExpectedSchema = "p1b-zero-sum-neighbor-local-search-v1"
        }
    }
}
$Steps += @{
    Name = "aggregate"
    ResultPath = Join-Path $DqnDir "runs\zero_sum_multi_axis_sweep\closeout_v1\zero_sum_multi_axis_sweep.json"
    Arguments = @("--aggregate")
    ExpectedSchema = "p1b-zero-sum-multi-axis-sweep-v1"
    IsAggregate = $true
}

function Test-Result {
    param([hashtable]$Step)
    if (-not (Test-Path -LiteralPath $Step.ResultPath)) { return $false }
    try {
        $Payload = Get-Content -Raw -LiteralPath $Step.ResultPath | ConvertFrom-Json
        return ($Payload.status -eq "complete" -and $Payload.schema -eq $Step.ExpectedSchema)
    }
    catch { return $false }
}

function Get-SystemLoad {
    $Samples = (Get-Counter -Counter "\Memory\Available MBytes", "\Processor(_Total)\% Processor Time").CounterSamples
    $Available = ($Samples | Where-Object Path -Like "*available mbytes").CookedValue
    $Cpu = ($Samples | Where-Object Path -Like "*processor time").CookedValue
    return @{ AvailableRamGB=[math]::Round($Available/1024.0,3); CpuPercent=[math]::Round($Cpu,2) }
}

function Write-Status {
    param([string]$State,[string]$CurrentStep,[int]$Completed,[string]$Message="",[hashtable]$Load=@{})
    @{
        schema="p1b-zero-sum-multi-axis-orchestration-v1"; state=$State
        current_step=$CurrentStep; completed_steps=$Completed; total_steps=$Steps.Count
        process_id=$PID; message=$Message; updated_at=(Get-Date).ToString("o")
        condition_contract=@{
            terrain_indices=@(0,1,2,3); dx_m=25; angular_dpsi_deg=@(10,5,2.5)
            angular_r=1; neighbor_dpsi_deg=5; neighbor_radii=@(1,3,5)
            leader_objective="attacker_objective"; new_runs=16; full_factorial=$false
        }
        readiness=$Load
    } | ConvertTo-Json -Depth 6 | Out-File -LiteralPath ($StatusPath+".tmp") -Encoding utf8
    Move-Item -LiteralPath ($StatusPath+".tmp") -Destination $StatusPath -Force
}

if ($PlanOnly) {
    foreach ($Step in $Steps) {
        [pscustomobject]@{Name=$Step.Name;Disposition=if(Test-Result $Step){"skip"}else{"run"};Path=$Step.ResultPath}
    }
    exit 0
}

$Ready=0; $Completed=0; $Step=$null
try {
    while ($Ready -lt $RequiredReadySamples) {
        $Load=Get-SystemLoad
        if($Load.AvailableRamGB -ge $MinimumAvailableRamGB -and $Load.CpuPercent -le $MaximumCpuPercent){$Ready++}else{$Ready=0}
        Write-Status "waiting_for_idle_resources" "resource_gate" 0 -Load $Load
        if($Ready -lt $RequiredReadySamples){Start-Sleep -Seconds $PollSeconds}
    }
    for($Index=0;$Index -lt $Steps.Count;$Index++){
        $Step=$Steps[$Index]
        if(Test-Result $Step){$Completed=$Index+1;Write-Status "running" ("skip_existing_"+$Step.Name) $Completed;continue}
        Write-Status "running" $Step.Name $Index -Load (Get-SystemLoad)
        $Stdout=Join-Path $RunDir ($Step.Name+".stdout.log")
        $Stderr=Join-Path $RunDir ($Step.Name+".stderr.log")
        $Child=Start-Process -FilePath $Python -ArgumentList (@($Runner)+@($Step.Arguments)) -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -Wait -PassThru
        if($Child.ExitCode -ne 0){throw "Step $($Step.Name) failed: $(Get-Content -Raw -LiteralPath $Stderr)"}
        if(-not (Test-Result $Step)){throw "Invalid or missing result for $($Step.Name)"}
        $Completed=$Index+1
    }
    Write-Status "complete" "none" $Steps.Count -Load (Get-SystemLoad)
}
catch {
    $Name=if($null -eq $Step){"resource_gate"}else{$Step.Name}
    Write-Status "failed" $Name $Completed -Message $_.Exception.Message -Load (Get-SystemLoad)
    throw
}

