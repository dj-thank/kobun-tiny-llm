param(
  [ValidateSet("Monitor", "Evaluate", "Improve", "TrainWhenReady", "FullLoop")]
  [string]$Mode = "Monitor",
  [switch]$AllowStartTraining,
  [switch]$ReviewsPassed
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$PreflightGate = "logs\preflight_gate_old_japanese_0_1b.json"
$ReviewGate = "logs\zero_base_review_gate_old_japanese_0_1b.json"
$Python = Join-Path $Root ".venv-dml\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  $Python = Join-Path $Root ".venv\Scripts\python.exe"
}
if (-not (Test-Path $Python)) {
  throw "Missing project Python environment."
}

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
  )
  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
  }
}

function Update-Board {
  param([bool]$RefreshReviewPacket = $true)
  & $Python scripts\update_evaluation_board.py | Out-Null
  if ($LASTEXITCODE -ne 0) {
    throw "Could not update evaluation board."
  }
  if ($RefreshReviewPacket) {
    & $Python scripts\build_llm_review_packet.py | Out-Null
    if ($LASTEXITCODE -ne 0) {
      throw "Could not build LLM review packet."
    }
  }
  $ActionJson = & $Python scripts\select_next_autonomous_action.py
  if ($LASTEXITCODE -ne 0) {
    throw "Could not select next autonomous action."
  }
  return ($ActionJson | ConvertFrom-Json)
}

function Run-PostQuality {
  param([Parameter(Mandatory = $true)][string]$RunId)
  $Checkpoint = "checkpoints\${RunId}_best.pt"
  if (-not (Test-Path $Checkpoint)) {
    throw "Missing exact best checkpoint: $Checkpoint"
  }
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\finalize_old_japanese_0_1b_dml.ps1 -RunId $RunId -Checkpoint $Checkpoint
  if ($LASTEXITCODE -ne 0) {
    throw "Post-run quality checks failed for $RunId."
  }
}

function New-DmlRunId {
  return "old_japanese_0_1b_dml_$(Get-Date -Format "yyyyMMdd_HHmmss")"
}

function Verify-ReleaseCandidateStartGates {
  Invoke-Checked $Python scripts\verify_preflight_gate.py --gate $PreflightGate --max-age-minutes 120
  Invoke-Checked $Python scripts\verify_zero_base_review_gate.py --gate $ReviewGate --preflight-gate $PreflightGate --max-age-minutes 120
}

function Get-FileSha256 {
  param([Parameter(Mandatory = $true)][string]$PathText)
  return (Get-FileHash -Algorithm SHA256 -LiteralPath $PathText).Hash.ToLowerInvariant()
}

function Write-AutonomousLaunchContext {
  param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [Parameter(Mandatory = $true)][object]$Action
  )
  New-Item -ItemType Directory -Force logs | Out-Null
  $Path = "logs\autonomous_launch_context_${RunId}.json"
  $Nonce = [guid]::NewGuid().ToString("N") + [guid]::NewGuid().ToString("N")
  $env:OLD_JAPANESE_AUTONOMOUS_LAUNCH_NONCE = $Nonce
  $Bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($Nonce)
  $Hash = [System.Security.Cryptography.SHA256]::Create().ComputeHash($Bytes)
  $NonceHash = -join ($Hash | ForEach-Object { $_.ToString("x2") })
  $Payload = [pscustomobject]@{
    schema = "old_japanese_0_1b_autonomous_launch_context_v1"
    run_id = $RunId
    generated_at_utc = Get-Date -Format o
    autonomous_pid = $PID
    autonomous_script = "scripts\autonomous_old_japanese_0_1b_loop.ps1"
    selected_action = [string]$Action.action
    selected_reason = [string]$Action.reason
    preflight_gate = $PreflightGate
    preflight_gate_sha256 = Get-FileSha256 $PreflightGate
    review_gate = $ReviewGate
    review_gate_sha256 = Get-FileSha256 $ReviewGate
    launch_nonce_sha256 = $NonceHash
    hf_export = $false
  }
  $Payload | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $Path
  return $Path
}

function Test-ExactDmlTrainingProcess {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Process,
    [Parameter(Mandatory = $true)]
    [string]$RunId
  )
  if ([string]$Process.Name -ne "python.exe") {
    return $false
  }
  $CommandLine = [string]$Process.CommandLine
  if (-not $CommandLine) {
    return $false
  }
  $RunPattern = '(?i)(?:^|\s)--run-id(?:\s+|=)' + [regex]::Escape($RunId) + '(?=\s|$|"|'')'
  return (
    $CommandLine -match '(?i)(?:^|\s)-m\s+kobun_llm\.train(?=\s|$|"|'')' -and
    $CommandLine -match $RunPattern -and
    $CommandLine -match '(?i)(?:^|\s)--device(?:\s+|=)dml(?=\s|$|"|'')'
  )
}

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

function Get-ExactDmlTrainingProcesses {
  param([Parameter(Mandatory = $true)][string]$RunId)
  return @(Get-CimInstance Win32_Process | Where-Object {
    Test-ExactDmlTrainingProcess -Process $_ -RunId $RunId
  })
}

function Get-DmlTrainingWrapperProcesses {
  param([Parameter(Mandatory = $true)][string]$RunId)
  return @(Get-CimInstance Win32_Process | Where-Object {
    $CommandLine = [string]$_.CommandLine
    return (
      $CommandLine -like "*train_old_japanese_0_1b_dml.ps1*" -and
      $CommandLine -like "*$RunId*"
    )
  })
}

function Get-DmlWatcherProcesses {
  param([Parameter(Mandatory = $true)][string]$RunId)
  return @(Get-CimInstance Win32_Process | Where-Object {
    $CommandLine = [string]$_.CommandLine
    return (
      $CommandLine -like "*watch_and_finalize_old_japanese_0_1b_dml.ps1*" -and
      $CommandLine -like "*$RunId*"
    )
  })
}

function Set-ActiveLockRunning {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [Parameter(Mandatory = $true)]
    [int]$TrainPid,
    [Parameter(Mandatory = $true)]
    [int]$WatcherPid,
    [string]$Reason = "autonomous_launch_repair"
  )
  $ActiveLock = "logs\active_old_japanese_0_1b_dml.lock"
  if (-not (Test-Path $ActiveLock)) {
    throw "Cannot repair active lock because it is missing: $ActiveLock"
  }
  $Lock = Get-Content -Raw -Encoding UTF8 $ActiveLock | ConvertFrom-Json
  if ($Lock.run_id -ne $RunId) {
    throw "Cannot repair active lock for ${RunId}; lock run_id=$($Lock.run_id)"
  }
  foreach ($Pair in @(
    @{ Name = "train_pid"; Value = $TrainPid },
    @{ Name = "watcher_pid"; Value = $WatcherPid },
    @{ Name = "state"; Value = "running" },
    @{ Name = "rescued_at"; Value = Get-Date -Format o },
    @{ Name = "rescue_reason"; Value = $Reason },
    @{ Name = "hf_export"; Value = $false }
  )) {
    $Lock | Add-Member -NotePropertyName $Pair.Name -NotePropertyValue $Pair.Value -Force
  }
  $Json = $Lock | ConvertTo-Json -Depth 6
  [System.IO.File]::WriteAllText(
    [IO.Path]::GetFullPath((Join-Path $Root $ActiveLock)),
    $Json + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
  )
}

function Repair-DmlLaunchIfTrainingLive {
  param([Parameter(Mandatory = $true)][string]$RunId)
  $Wrappers = @(Get-DmlTrainingWrapperProcesses -RunId $RunId)
  $Training = @(Get-ExactDmlTrainingProcesses -RunId $RunId)
  if ($Wrappers.Count -eq 0 -or $Training.Count -eq 0) {
    return $false
  }
  $Wrapper = $Wrappers | Sort-Object ProcessId | Select-Object -First 1
  $Watchers = @(Get-DmlWatcherProcesses -RunId $RunId)
  if ($Watchers.Count -eq 0) {
    $WatchOut = "logs\watch_start_${RunId}.out.log"
    $WatchErr = "logs\watch_start_${RunId}.err.log"
    $Watcher = Start-Process powershell `
      -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "scripts\watch_and_finalize_old_japanese_0_1b_dml.ps1",
        "-ProcessId",
        "$($Wrapper.ProcessId)",
        "-RunId",
        $RunId
      ) `
      -RedirectStandardOutput $WatchOut `
      -RedirectStandardError $WatchErr `
      -PassThru `
      -WindowStyle Hidden
    Start-Sleep -Seconds 2
    if (-not (Get-Process -Id $Watcher.Id -ErrorAction SilentlyContinue)) {
      throw "Launch repair watcher exited immediately for $RunId. See $WatchErr."
    }
  } else {
    $Watcher = $Watchers | Sort-Object ProcessId | Select-Object -First 1
  }
  Set-ActiveLockRunning -RunId $RunId -TrainPid ([int]$Wrapper.ProcessId) -WatcherPid ([int]$Watcher.ProcessId)
  Write-Output "launch_repaired=1 run_id=$RunId train_pid=$($Wrapper.ProcessId) watcher_pid=$($Watcher.ProcessId)"
  return $true
}

function Write-NonReleaseRunRecord {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [Parameter(Mandatory = $true)]
    [string]$Reason,
    [Parameter(Mandatory = $true)]
    [string]$ExitSentinel
  )
  New-Item -ItemType Directory -Force logs\non_release_runs | Out-Null
  $ArchivePattern = "active_old_japanese_0_1b_dml.${RunId}.failed*.json"
  $Archive = Get-ChildItem logs -Filter $ArchivePattern -ErrorAction SilentlyContinue |
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
    train_exit_sentinel = $ExitSentinel
    active_lock_archive = if ($Archive) { $Archive.Name } else { "" }
    source_archive_path = $ArchivePath
    source_archive_sha256 = $ArchiveSha256
    hf_export = $false
  }
  $Path = "logs\non_release_runs\${RunId}.json"
  $Payload | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $Path
  return $Path
}

function Stop-NonReleaseRun {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [Parameter(Mandatory = $true)]
    [string]$Reason,
    [Parameter(Mandatory = $true)]
    [string]$Label
  )
  if ($RunId -notmatch '^old_japanese_0_1b_dml_[0-9A-Za-z][0-9A-Za-z_-]{0,63}$') {
    throw "Refusing ${Label} stop for invalid DML RunId: $RunId"
  }
  $ActiveLock = "logs\active_old_japanese_0_1b_dml.lock"
  if (-not (Test-Path $ActiveLock)) {
    throw "Refusing ${Label} stop because active lock is missing: $ActiveLock"
  }
  $Lock = Get-Content -Raw -Encoding UTF8 $ActiveLock | ConvertFrom-Json
  if ($Lock.run_id -ne $RunId) {
    throw "Refusing ${Label} stop: active lock run_id=$($Lock.run_id) requested=$RunId"
  }
  $LockTrainPid = 0
  if ($Lock.train_pid -and -not [int]::TryParse([string]$Lock.train_pid, [ref]$LockTrainPid)) {
    throw "Refusing ${Label} stop: active lock train_pid is not an integer: $($Lock.train_pid)"
  }
  if ($LockTrainPid -gt 0 -and -not (Test-PidMatchesRun $LockTrainPid $RunId)) {
    throw "Refusing ${Label} stop: active lock train_pid=$LockTrainPid does not match $RunId"
  }
  $Candidates = @(Get-ExactDmlTrainingProcesses -RunId $RunId)
  if ($Candidates.Count -eq 0) {
    throw "No live kobun_llm.train process found for ${Label} stop: $RunId"
  }
  foreach ($Process in $Candidates) {
    Stop-Process -Id ([int]$Process.ProcessId) -Force
  }
  $ExitSentinel = "logs\train_exit_${RunId}.json"
  $Deadline = (Get-Date).AddSeconds(60)
  while (-not (Test-Path $ExitSentinel) -and (Get-Date) -lt $Deadline) {
    Start-Sleep -Seconds 2
  }
  if (-not (Test-Path $ExitSentinel)) {
    throw "${Label} stop did not produce train exit sentinel yet: $ExitSentinel"
  }
  $Sentinel = Get-Content -Raw -Encoding UTF8 $ExitSentinel | ConvertFrom-Json
  if ([int]$Sentinel.exit_code -eq 0) {
    throw "${Label} stop expected a nonzero interrupted sentinel but saw exit_code=0 for $RunId."
  }
  $StillLive = @(Get-ExactDmlTrainingProcesses -RunId $RunId)
  if ($StillLive.Count -gt 0) {
    throw "${Label} stop left live DML training processes for ${RunId}: $($StillLive.ProcessId -join ',')"
  }
  $ArchivePattern = "active_old_japanese_0_1b_dml.${RunId}.failed*.json"
  $ArchiveDeadline = (Get-Date).AddSeconds(60)
  while (-not (Get-ChildItem logs -Filter $ArchivePattern -ErrorAction SilentlyContinue) -and (Get-Date) -lt $ArchiveDeadline) {
    Start-Sleep -Seconds 2
  }
  if (-not (Get-ChildItem logs -Filter $ArchivePattern -ErrorAction SilentlyContinue)) {
    throw "${Label} stop did not produce failed active-lock archive for $RunId."
  }
  $RecordPath = Write-NonReleaseRunRecord -RunId $RunId -Reason $Reason -ExitSentinel $ExitSentinel
  Write-Output "${Label}_run_stopped=1 run_id=$RunId exit_code=$($Sentinel.exit_code)"
  Write-Output "non_release_record=$RecordPath"
}

function Stop-OverfitRun {
  param([Parameter(Mandatory = $true)][string]$RunId)
  Stop-NonReleaseRun -RunId $RunId -Reason "autonomous_overfit_stop" -Label "overfit"
}

function Stop-SupersededRun {
  param([Parameter(Mandatory = $true)][string]$RunId)
  Stop-NonReleaseRun -RunId $RunId -Reason "superseded_by_byte_fallback_speed_fix" -Label "superseded"
}

function Format-SelectorBlockers {
  param([object]$Action)
  if ($Action.hard_blockers) {
    try {
      return (($Action.hard_blockers | ConvertTo-Json -Compress -Depth 8) -replace '\r?\n', ' ')
    } catch {
      return [string]$Action.hard_blockers
    }
  }
  if ($Action.reason) {
    return [string]$Action.reason
  }
  return "selector did not provide blocker details"
}

$RefreshReviewPacketForMode = $true
if (($Mode -eq "TrainWhenReady" -or $Mode -eq "FullLoop") -and $AllowStartTraining.IsPresent -and $ReviewsPassed.IsPresent) {
  $RefreshReviewPacketForMode = $false
}

$Action = Update-Board -RefreshReviewPacket $RefreshReviewPacketForMode

if ($Mode -eq "Monitor") {
  $Action | ConvertTo-Json -Depth 8 -Compress | Write-Output
  exit 0
}

if ($Mode -eq "Evaluate" -or $Mode -eq "FullLoop") {
  if ($Action.action -eq "supersede_non_release_run") {
    Stop-SupersededRun -RunId ([string]$Action.run_id)
    $Action = Update-Board -RefreshReviewPacket $RefreshReviewPacketForMode
  }
  if ($Action.action -eq "stop_overfit_run") {
    Stop-OverfitRun -RunId ([string]$Action.run_id)
    $Action = Update-Board -RefreshReviewPacket $RefreshReviewPacketForMode
  }
  if ($Action.action -eq "run_post_run_quality_checks") {
    Run-PostQuality -RunId ([string]$Action.run_id)
    $Action = Update-Board -RefreshReviewPacket $RefreshReviewPacketForMode
  }
  if ($Mode -eq "Evaluate") {
    exit 0
  }
}

if ($Mode -eq "Improve") {
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_static_quality_checks.ps1 -RefreshEvidence
  if ($LASTEXITCODE -ne 0) {
    throw "Static quality checks failed."
  }
  exit 0
}

if ($Mode -eq "TrainWhenReady" -or $Mode -eq "FullLoop") {
  if ($Action.action -eq "supersede_non_release_run") {
    Stop-SupersededRun -RunId ([string]$Action.run_id)
    $Action = Update-Board -RefreshReviewPacket $RefreshReviewPacketForMode
  }
  if ($Action.action -eq "stop_overfit_run") {
    Stop-OverfitRun -RunId ([string]$Action.run_id)
    $Action = Update-Board -RefreshReviewPacket $RefreshReviewPacketForMode
  }
  if ($Action.action -eq "run_post_run_quality_checks") {
    Run-PostQuality -RunId ([string]$Action.run_id)
    $Action = Update-Board -RefreshReviewPacket $RefreshReviewPacketForMode
    Write-Output "post_run_quality_completed=1 run_id=$($Action.run_id)"
    exit 0
  }
  if ($Action.action -eq "monitor") {
    Write-Output "active_training_present=1; not starting another run."
    exit 0
  }
  if ($Action.action -eq "stop_and_report_upload_ready") {
    Write-Output "upload_ready_not_exported=1 run_id=$($Action.run_id)"
    exit 0
  }
  if ($Action.action -eq "fix_blockers") {
    $Blockers = Format-SelectorBlockers -Action $Action
    throw "Refusing to start training because selector requested blocker fixes: $Blockers"
  }
  if ($Action.action -ne "prepare_next_fresh_run_after_static_gate_and_zero_base_reviews") {
    $ActionJson = $Action | ConvertTo-Json -Compress -Depth 8
    throw "Refusing to start training for unsupported selector action: $ActionJson"
  }
  if (-not $ReviewsPassed.IsPresent) {
    throw "Refusing to start training until the configured independent review gate has passed."
  }
  if (-not $AllowStartTraining.IsPresent) {
    throw "Refusing to start training without -AllowStartTraining."
  }
  Verify-ReleaseCandidateStartGates
  $RunId = New-DmlRunId
  Invoke-Checked $Python scripts\assert_run_id_unused.py --run-id $RunId
  $AutonomousLaunchContext = Write-AutonomousLaunchContext -RunId $RunId -Action $Action
  try {
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_old_japanese_0_1b_dml_and_watch.ps1 -RunId $RunId -PreflightGate $PreflightGate -ReviewGate $ReviewGate -AutonomousLaunchContext $AutonomousLaunchContext -AllowStartTraining -ReviewsPassed
    if ($LASTEXITCODE -ne 0) {
      if (Repair-DmlLaunchIfTrainingLive -RunId $RunId) {
        exit 0
      }
      throw "Fresh DirectML launch failed."
    }
  } finally {
    Remove-Item Env:\OLD_JAPANESE_AUTONOMOUS_LAUNCH_NONCE -ErrorAction SilentlyContinue
  }
}
