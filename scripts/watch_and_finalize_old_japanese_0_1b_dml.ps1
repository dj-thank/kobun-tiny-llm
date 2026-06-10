param(
  [Parameter(Mandatory = $true)]
  [int]$ProcessId,
  [Parameter(Mandatory = $true)]
  [string]$RunId
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ($RunId -notmatch '^old_japanese_0_1b_dml_[0-9A-Za-z][0-9A-Za-z_-]{0,63}$') {
  throw "Invalid RunId: $RunId"
}

New-Item -ItemType Directory -Force logs | Out-Null
$WatchLog = "logs\watch_finalize_${RunId}.log"
$FinalLog = "logs\finalize_${RunId}.log"
$ExitSentinel = "logs\train_exit_${RunId}.json"
$ExpectedCheckpoint = "checkpoints\${RunId}.pt"
$ExpectedBestCheckpoint = "checkpoints\${RunId}_best.pt"
$ActiveLock = "logs\active_old_japanese_0_1b_dml.lock"
$WatchStarted = Get-Date

"watch_started=$($WatchStarted.ToString('o')) pid=$ProcessId run_id=$RunId" | Add-Content -Encoding UTF8 $WatchLog
"hf_export_requested=False" | Add-Content -Encoding UTF8 $WatchLog
"hf_export_policy=watcher_never_exports" | Add-Content -Encoding UTF8 $WatchLog

function Test-PidMatchesRun {
  param(
    [int]$PidValue,
    [string]$ExpectedRunId
  )
  if ($PidValue -le 0) {
    return $false
  }
  $Proc = Get-CimInstance Win32_Process -Filter "ProcessId=$PidValue" -ErrorAction SilentlyContinue
  if (-not $Proc) {
    return $false
  }
  $CommandLine = [string]$Proc.CommandLine
  if ($CommandLine -notlike "*$ExpectedRunId*") {
    return $false
  }
  return (
    $CommandLine -like "*kobun_llm.train*" -or
    $CommandLine -like "*train_old_japanese_0_1b_dml.ps1*" -or
    $CommandLine -like "*watch_and_finalize_old_japanese_0_1b_dml.ps1*" -or
    $CommandLine -like "*start_old_japanese_0_1b_dml_and_watch.ps1*"
  )
}

function Complete-ActiveLock {
  param(
    [Parameter(Mandatory = $true)]
    [string]$State,
    [int]$ExitCode = 0
  )
  try {
  if (-not (Test-Path $ActiveLock)) {
    return
  }
    $Lock = Get-Content -Raw -Encoding UTF8 $ActiveLock | ConvertFrom-Json
  } catch {
    "active_lock_complete_skipped=$(Get-Date -Format o) reason=unreadable_lock message=$($_.Exception.Message)" |
      Add-Content -Encoding UTF8 $WatchLog
    return
  }
  try {
  if ($Lock.run_id -ne $RunId) {
    return
  }
  $LockTrainPid = 0
  if ($Lock.train_pid -and -not [int]::TryParse([string]$Lock.train_pid, [ref]$LockTrainPid)) {
    "active_lock_complete_skipped=$(Get-Date -Format o) reason=invalid_train_pid value=$($Lock.train_pid)" |
      Add-Content -Encoding UTF8 $WatchLog
    return
  }
  if ($LockTrainPid -gt 0 -and $LockTrainPid -ne $ProcessId) {
    return
  }
  foreach ($Pair in @(
    @{ Name = "state"; Value = $State },
    @{ Name = "completed_at"; Value = Get-Date -Format o },
    @{ Name = "finalize_exit_code"; Value = $ExitCode },
    @{ Name = "hf_export"; Value = $false }
  )) {
    $Lock | Add-Member -NotePropertyName $Pair.Name -NotePropertyValue $Pair.Value -Force
  }
  $Archive = "logs\active_old_japanese_0_1b_dml.${RunId}.${State}.json"
  $Lock | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $Archive
  if (-not (Test-PidMatchesRun $ProcessId $RunId)) {
    Remove-Item -LiteralPath $ActiveLock -Force -ErrorAction SilentlyContinue
  }
  } catch {
    "active_lock_complete_failed=$(Get-Date -Format o) state=$State message=$($_.Exception.Message)" |
      Add-Content -Encoding UTF8 $WatchLog
    return
  }
}

function Write-NonReleaseRunRecord {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Reason,
    [string]$ExitSentinelPath = $ExitSentinel
  )
  try {
    New-Item -ItemType Directory -Force logs\non_release_runs | Out-Null
    $Archive = Get-ChildItem logs -Filter "active_old_japanese_0_1b_dml.${RunId}.failed*.json" -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 1
    $ArchivePath = ""
    $ArchiveSha256 = ""
    if ($Archive) {
      $ArchivePath = "logs/$($Archive.Name)"
      $ArchiveSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive.FullName).Hash.ToLowerInvariant()
    }
    $Payload = [pscustomobject]@{
      run_id = $RunId
      release_status = "non_release_artifact"
      reason = $Reason
      created_at = Get-Date -Format o
      train_exit_sentinel = $ExitSentinelPath
      active_lock_archive = if ($Archive) { $Archive.Name } else { "" }
      source_archive_path = $ArchivePath
      source_archive_sha256 = $ArchiveSha256
      hf_export = $false
    }
    $Path = "logs\non_release_runs\${RunId}.json"
    $Payload | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $Path
    "non_release_record=$Path reason=$Reason" | Add-Content -Encoding UTF8 $WatchLog
  } catch {
    "non_release_record_failed=$(Get-Date -Format o) reason=$Reason message=$($_.Exception.Message)" |
      Add-Content -Encoding UTF8 $WatchLog
  }
}

function Normalize-RepoPath {
  param(
    [Parameter(Mandatory = $true)]
    [string]$PathText
  )
  $PathText = $PathText -replace '/', '\'
  $PathObject = [IO.Path]::GetFullPath((Join-Path $Root $PathText))
  return $PathObject.TrimEnd('\').ToLowerInvariant()
}

function Assert-SameRepoPath {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Label,
    [Parameter(Mandatory = $true)]
    [string]$Expected,
    [Parameter(Mandatory = $true)]
    [string]$Actual
  )
  if ((Normalize-RepoPath $Expected) -ne (Normalize-RepoPath $Actual)) {
    throw "Training exit sentinel $Label mismatch: expected=$Expected actual=$Actual"
  }
}

try {
  $ExitCode = $null
  $ObservedProcess = $false
  try {
  $Process = Get-Process -Id $ProcessId -ErrorAction Stop
  $ObservedProcess = $true
  $Process.WaitForExit()
  try {
    $ExitCode = $Process.ExitCode
  } catch {
    $ExitCode = $null
  }
  "training_process_exited=$(Get-Date -Format o) observed_exit_code=$ExitCode" | Add-Content -Encoding UTF8 $WatchLog
  if ($null -eq $ExitCode) {
    "training_process_exit_code_unavailable=$(Get-Date -Format o) using_train_exit_sentinel=true" |
      Add-Content -Encoding UTF8 $WatchLog
  }
  elseif ($ExitCode -ne 0) {
    throw "Watched training process exited with code $ExitCode."
  }
} catch {
  if ($ObservedProcess -and $null -ne $ExitCode -and $ExitCode -ne 0) {
    throw
  }
  if ($ObservedProcess -and $null -eq $ExitCode) {
    "training_process_exit_status_unknown=$(Get-Date -Format o) using_train_exit_sentinel=true message=$($_.Exception.Message)" |
      Add-Content -Encoding UTF8 $WatchLog
  } else {
    "training_process_not_found_or_already_exited=$(Get-Date -Format o) message=$($_.Exception.Message)" |
      Add-Content -Encoding UTF8 $WatchLog
  }
  }

  if (-not (Test-Path $ExitSentinel)) {
  throw "Training exit sentinel was not found: $ExitSentinel"
  }
  $Sentinel = Get-Content -Raw -Encoding UTF8 $ExitSentinel | ConvertFrom-Json
  if ($Sentinel.run_id -ne $RunId) {
  throw "Training exit sentinel run id mismatch: expected=$RunId actual=$($Sentinel.run_id)"
  }
  if ([int]$Sentinel.exit_code -ne 0) {
  throw "Training exit sentinel reports failure: exit_code=$($Sentinel.exit_code) message=$($Sentinel.message)"
  }
  $SentinelCompletedAt = [datetimeoffset]::Parse([string]$Sentinel.completed_at)
  $WatchStartedOffset = [datetimeoffset]::new($WatchStarted)
  if ($SentinelCompletedAt -lt $WatchStartedOffset) {
  throw "Training exit sentinel is stale: completed_at=$($Sentinel.completed_at) watch_started=$($WatchStarted.ToString('o'))"
  }
  Assert-SameRepoPath -Label "checkpoint" -Expected $ExpectedCheckpoint -Actual ([string]$Sentinel.checkpoint)
  Assert-SameRepoPath -Label "best checkpoint" -Expected $ExpectedBestCheckpoint -Actual ([string]$Sentinel.best_checkpoint)
  "training_exit_sentinel_ok=$ExitSentinel exit_code=$($Sentinel.exit_code)" | Add-Content -Encoding UTF8 $WatchLog

  $TrainingLog = "logs\${RunId}.out.log"
  if (-not (Test-Path $TrainingLog)) {
  throw "Expected training log for this run was not found: $TrainingLog"
  }
  $RunLine = Select-String -Path $TrainingLog -Pattern "^run_id=$([regex]::Escape($RunId))\s" | Select-Object -First 1
  if (-not $RunLine) {
  throw "Training log does not prove it belongs to run id ${RunId}: $TrainingLog"
  }

  $Checkpoint = $ExpectedBestCheckpoint
  if (-not (Test-Path $Checkpoint)) {
  throw "Expected best checkpoint for this run was not found: $Checkpoint"
  }

  $FinalizeArgs = @(
  "-NoProfile",
  "-ExecutionPolicy",
  "Bypass",
  "-File",
  "scripts\finalize_old_japanese_0_1b_dml.ps1",
  "-Checkpoint",
  $Checkpoint,
  "-RunId",
  $RunId
  )

  $PreviousErrorActionPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
  & powershell @FinalizeArgs *> $FinalLog
  $FinalizeExit = $LASTEXITCODE
  } finally {
  $ErrorActionPreference = $PreviousErrorActionPreference
  }
  "finalize_exited=$(Get-Date -Format o) exit_code=$FinalizeExit final_log=$FinalLog" | Add-Content -Encoding UTF8 $WatchLog
  if ($FinalizeExit -ne 0) {
  throw "Finalize failed with exit code $FinalizeExit. See $FinalLog."
  }
  Complete-ActiveLock -State "completed" -ExitCode 0
} catch {
  $OriginalError = $_
  Complete-ActiveLock -State "failed" -ExitCode 1
  Write-NonReleaseRunRecord -Reason "watcher_finalizer_or_training_failure"
  throw $OriginalError
}
