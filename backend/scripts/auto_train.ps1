# Automated, unattended training for DATORYX's native model - Windows PowerShell.
#
# Runs training toward a target total step count, in chunks, and keeps
# going even if a chunk gets interrupted (laptop sleep, closed window, a
# crash) - just re-run this same script and it picks up exactly where it
# left off, because every chunk uses --resume against the real checkpoint
# on disk.
#
# Usage (from backend/ in PowerShell):
#   .\scripts\auto_train.ps1                          # trains to 20,000 total steps
#   .\scripts\auto_train.ps1 -TargetSteps 50000        # trains to 50,000 total steps
#   .\scripts\auto_train.ps1 -TargetSteps 50000 -ExtraTextDir C:\path\to\notes
#
# To let it run in the background and survive closing the window, start
# it minimized/detached instead:
#   Start-Process powershell -ArgumentList "-File scripts\auto_train.ps1 -TargetSteps 50000" -WindowStyle Hidden
# Check on it later with: Get-Content models\native\checkpoint\auto_train.log -Tail 20 -Wait

param(
    [int]$TargetSteps = 20000,
    [string]$ExtraTextDir = "",
    [int]$ChunkSteps = 2000,
    [int]$MaxRetries = 5
)

Set-Location (Join-Path $PSScriptRoot "..")   # run from backend/

$CheckpointDir = "models\native\checkpoint"
$LogFile = Join-Path $CheckpointDir "auto_train.log"
New-Item -ItemType Directory -Force -Path $CheckpointDir | Out-Null

function Get-CurrentStep {
    $logPath = Join-Path $CheckpointDir "train_log.json"
    if (Test-Path $logPath) {
        try {
            $log = Get-Content $logPath -Raw | ConvertFrom-Json
            if ($log.Count -gt 0) { return $log[-1].step } else { return 0 }
        } catch { return 0 }
    }
    return 0
}

function Write-Log($msg) {
    $line = "$msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Write-Log "=== DATORYX native model - automated training ==="
Write-Log "Target: $TargetSteps total steps | started: $(Get-Date)"

$extraArgs = @()
if ($ExtraTextDir -ne "") {
    $extraArgs += @("--extra-text-dir", $ExtraTextDir)
    Write-Log "Extra text source: $ExtraTextDir"
}

$current = Get-CurrentStep
Write-Log "Currently at step $current"

$retries = 0
while ($current -lt $TargetSteps) {
    $remaining = $TargetSteps - $current
    $thisChunk = [Math]::Min($remaining, $ChunkSteps)

    Write-Log "--- $(Get-Date): training chunk of $thisChunk steps (currently at $current/$TargetSteps) ---"

    $args = @("-m", "models.native.train", "--steps", $thisChunk, "--out-dir", $CheckpointDir, "--resume") + $extraArgs
    $proc = Start-Process -FilePath "python" -ArgumentList $args -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "$LogFile.tmp" -RedirectStandardError "$LogFile.tmp.err"
    Get-Content "$LogFile.tmp" -ErrorAction SilentlyContinue | Add-Content -Path $LogFile
    Get-Content "$LogFile.tmp.err" -ErrorAction SilentlyContinue | Add-Content -Path $LogFile
    Remove-Item "$LogFile.tmp", "$LogFile.tmp.err" -ErrorAction SilentlyContinue

    $newCurrent = Get-CurrentStep

    if ($proc.ExitCode -ne 0 -and $newCurrent -eq $current) {
        $retries++
        Write-Log "Chunk failed with no progress (attempt $retries/$MaxRetries)"
        if ($retries -ge $MaxRetries) {
            Write-Log "Giving up after $MaxRetries failed attempts with no progress. Check $LogFile for the actual error."
            exit 1
        }
        Start-Sleep -Seconds 5
    } else {
        $retries = 0
    }

    $current = Get-CurrentStep
    Write-Log "Now at step $current/$TargetSteps"
}

Write-Log "=== Done: reached $current/$TargetSteps steps at $(Get-Date) ==="
