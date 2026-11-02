param()

$ErrorActionPreference = "Stop"
$RunDir = Join-Path $PSScriptRoot "runs\phase7_angular_sweep\orchestration_v1"
$StatusPath = Join-Path $RunDir "status.json"
$AlarmStatusPath = Join-Path $RunDir "youtube_alarm_status.json"
$VideoUrl = "https://www.youtube.com/watch?v=2i-STzGbCl4&t=8s"

function Write-AlarmStatus {
    param([string]$State, [string]$Message)
    @{
        schema = "p1b-phase7-youtube-completion-alarm-v1"
        state = $State
        message = $Message
        video_url = $VideoUrl
        watcher_process_id = $PID
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Out-File -LiteralPath $AlarmStatusPath -Encoding utf8
}
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
Write-AlarmStatus -State "watching" -Message "Waiting for angular sweep completion."

while ($true) {
    if (Test-Path -LiteralPath $StatusPath) {
        try {
            $RunStatus = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
            if ($RunStatus.state -eq "complete") {
                Start-Process $VideoUrl
                Write-AlarmStatus `
                    -State "opened" `
                    -Message "Angular sweep completed; the requested video was opened."
                exit 0
            }
            if ($RunStatus.state -eq "failed") {
                Write-AlarmStatus `
                    -State "cancelled" `
                    -Message "Angular sweep failed; the video was not opened."
                exit 1
            }
        }
        catch {
            Write-AlarmStatus `
                -State "watching" `
                -Message "Status file was temporarily unreadable; monitoring continues."
        }
    }
    Start-Sleep -Seconds 20
}

