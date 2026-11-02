param()

$ErrorActionPreference = "Stop"

$dqnDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$runDirectory = Join-Path $dqnDirectory (
    "runs\phase6_g2_local_sse\final_id_001\dx10m_dpsi5deg_r1"
)
$backgroundStatusPath = Join-Path $runDirectory "background_process.json"
$alarmStatusPath = Join-Path $runDirectory "completion_alarm_status.json"
$youtubeUrl = "https://www.youtube.com/watch?v=2i-STzGbCl4&t=8s"

if (-not (Test-Path -LiteralPath $backgroundStatusPath)) {
    throw "Background process status not found: $backgroundStatusPath"
}

$background = Get-Content -LiteralPath $backgroundStatusPath -Raw |
    ConvertFrom-Json
$parentPid = [int]$background.pid
$finalResultPath = [string]$background.final_result

while ($true) {
    if (Test-Path -LiteralPath $finalResultPath) {
        $result = Get-Content -LiteralPath $finalResultPath -Raw |
            ConvertFrom-Json
        if ($result.status -eq "complete") {
            $status = [ordered]@{
                schema = "p1b-phase6-completion-alarm-v1"
                status = "triggered"
                triggered_local = (Get-Date).ToString("o")
                result_path = $finalResultPath
                url = $youtubeUrl
            }
            $json = $status | ConvertTo-Json -Depth 4
            [System.IO.File]::WriteAllText(
                $alarmStatusPath,
                $json + [Environment]::NewLine,
                [System.Text.UTF8Encoding]::new($false)
            )
            Start-Process -FilePath $youtubeUrl
            exit 0
        }
    }

    $parent = Get-Process -Id $parentPid -ErrorAction SilentlyContinue
    if ($null -eq $parent) {
        $status = [ordered]@{
            schema = "p1b-phase6-completion-alarm-v1"
            status = "calculation_stopped_before_completion"
            stopped_local = (Get-Date).ToString("o")
            result_path = $finalResultPath
            url = $youtubeUrl
        }
        $json = $status | ConvertTo-Json -Depth 4
        [System.IO.File]::WriteAllText(
            $alarmStatusPath,
            $json + [Environment]::NewLine,
            [System.Text.UTF8Encoding]::new($false)
        )
        exit 1
    }

    Start-Sleep -Seconds 30
}
