param(
  [string]$RunId = "",
  [ValidateSet("auto", "dml", "cuda")]
  [string]$Backend = "auto",
  [int]$TotalSteps = 8000,
  [int]$ProcessId = 0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $RunId) {
  throw "RunId is required. Refusing to infer progress from a historical default run."
}

if ($RunId -like "old_japanese_0_1b_dml_*" -or $RunId -like "old_japanese_0_1b_cuda_*") {
  # Full run id already provided.
} elseif ($Backend -eq "cuda") {
  $RunId = "old_japanese_0_1b_cuda_$RunId"
} else {
  $RunId = "old_japanese_0_1b_dml_$RunId"
}
if ($Backend -eq "cuda" -and $RunId -match '^old_japanese_0_1b_dml_') {
  throw "Invalid CUDA RunId: DML-looking ids are not accepted by CUDA progress reporting: $RunId"
}
if ($RunId -match '^old_japanese_0_1b_dml_') {
  if ($RunId -notmatch '^old_japanese_0_1b_dml_[0-9A-Za-z][0-9A-Za-z_-]{0,63}$') {
    throw "Invalid DML RunId: $RunId"
  }
} elseif ($RunId -match '^old_japanese_0_1b_cuda_') {
  if ($RunId -notmatch '^old_japanese_0_1b_cuda_[0-9A-Za-z][0-9A-Za-z_-]{0,63}$') {
    throw "Invalid CUDA RunId: $RunId"
  }
} else {
  throw "Invalid RunId: $RunId"
}

$LogPath = "logs\${RunId}.out.log"
$ErrPath = "logs\${RunId}.err.log"
$ExitSentinelPath = "logs\train_exit_${RunId}.json"
$BestPath = "checkpoints\${RunId}_best.pt"
$EvalResultsPath = "logs\eval_results_${RunId}.json"
$NonReleaseRecordPath = "logs\non_release_runs\${RunId}.json"
$ActiveLockPath = if ($RunId -match '^old_japanese_0_1b_cuda_') {
  "logs\active_old_japanese_0_1b_cuda.lock"
} else {
  "logs\active_old_japanese_0_1b_dml.lock"
}

function Resolve-TrainingProcessId {
  param(
    [Parameter(Mandatory = $true)]
    [string]$TargetRunId
  )
  $Candidates = Get-CimInstance Win32_Process |
    Where-Object {
      $_.CommandLine -and
      $_.CommandLine -match [regex]::Escape($TargetRunId) -and
      $_.CommandLine -match 'kobun_llm\.train' -and
      $_.Name -match '^python(\.exe)?$'
    } |
    ForEach-Object {
      $Process = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
      if ($Process) {
        [pscustomobject]@{
          Id = [int]$_.ProcessId
          Cpu = if ($null -ne $Process.CPU) { [double]$Process.CPU } else { 0.0 }
          StartTime = $Process.StartTime
          CommandLine = [string]$_.CommandLine
        }
      }
    }
  $LiveCandidates = @($Candidates | Sort-Object Cpu, StartTime -Descending)
  if ($LiveCandidates.Count -eq 0) {
    return 0
  }
  return [int]$LiveCandidates[0].Id
}

function Test-ProcessMatchesRun {
  param(
    [Parameter(Mandatory = $true)][int]$PidValue,
    [Parameter(Mandatory = $true)][string]$TargetRunId
  )
  $Proc = Get-CimInstance Win32_Process -Filter "ProcessId=$PidValue" -ErrorAction SilentlyContinue
  if (-not $Proc) {
    return $false
  }
  $CommandLine = [string]$Proc.CommandLine
  if (-not $CommandLine) {
    return $false
  }
  return (
    $CommandLine -match [regex]::Escape($TargetRunId) -and
    $CommandLine -match 'kobun_llm\.train' -and
    $Proc.Name -match '^python(\.exe)?$'
  )
}

function Get-ActiveLockForRun {
  if (-not (Test-Path $ActiveLockPath)) {
    return $null
  }
  try {
    $Payload = Get-Content -Raw -Encoding UTF8 $ActiveLockPath | ConvertFrom-Json
  } catch {
    return [pscustomobject]@{
      run_id = $RunId
      state = "invalid_json"
      invalid_json = $true
    }
  }
  if ([string]$Payload.run_id -ne $RunId) {
    return $null
  }
  return $Payload
}

function Write-EarlyProgressReport {
  param(
    [Parameter(Mandatory = $true)][string]$Status,
    [object]$ProcessObject = $null,
    [object]$ActiveLockObject = $null,
    [string]$Reason = ""
  )
  $ErrBytes = if (Test-Path $ErrPath) { (Get-Item $ErrPath).Length } else { 0 }
  $EvalStatus = if (Test-Path $EvalResultsPath) { "present_not_read_before_first_eval" } else { "missing" }
  $NonReleaseRecorded = Test-Path $NonReleaseRecordPath
  $Report = [pscustomobject]@{
    RunId = $RunId
    LatestStep = $null
    TotalSteps = $TotalSteps
    TrainLoss = $null
    ValLoss = $null
    BestStep = $null
    BestValLoss = $null
    Completed = $false
    LogCompleted = $false
    ProgressStatus = $Status
    StopReason = $Reason
    StartedAt = if ($ProcessObject) { $ProcessObject.StartTime } else { $null }
    LastEvalAt = $null
    StepsPerMinute = 0
    EstimatedFinishAt = $null
    TrainingPid = if ($ProcessObject) { $ProcessObject.Id } else { $null }
    ProcessStatus = if ($ProcessObject) { "running" } elseif ($ActiveLockObject) { "active_lock_present" } else { "not_found" }
    ExitSentinelStatus = if (Test-Path $ExitSentinelPath) { "present_not_checked_before_first_eval" } else { "missing_while_starting" }
    ExitSentinelPath = $ExitSentinelPath
    EvalResultsPath = $EvalResultsPath
    EvalStatus = $EvalStatus
    NonReleaseRecorded = $NonReleaseRecorded
    ActiveLockMatchesRun = [bool]$ActiveLockObject
    ActiveLockState = if ($ActiveLockObject) { [string]$ActiveLockObject.state } else { "" }
    BestCheckpointBytes = if (Test-Path $BestPath) { (Get-Item $BestPath).Length } else { 0 }
    StderrBytes = $ErrBytes
  }
  $Report | Format-List
}

if ($ProcessId -le 0) {
  $ResolvedProcessId = Resolve-TrainingProcessId -TargetRunId $RunId
  if ($ResolvedProcessId -gt 0) {
    $ProcessId = $ResolvedProcessId
  }
}

$InitialProcess = $null
if ($ProcessId -gt 0) {
  if (-not (Test-ProcessMatchesRun -PidValue $ProcessId -TargetRunId $RunId)) {
    throw "Explicit ProcessId $ProcessId does not match live python kobun_llm.train command for RunId $RunId."
  }
  $InitialProcess = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
}
$InitialActiveLock = Get-ActiveLockForRun

if (-not (Test-Path $LogPath)) {
  if ($InitialProcess -or $InitialActiveLock) {
    Write-EarlyProgressReport `
      -Status "starting_waiting_for_log" `
      -ProcessObject $InitialProcess `
      -ActiveLockObject $InitialActiveLock `
      -Reason "training log has not been created yet"
    exit 0
  }
  throw "Missing training log: $LogPath"
}

function Read-ProgressLogText {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )
  $ResolvedPath = (Resolve-Path -LiteralPath $Path).Path
  $Stream = [IO.File]::Open($ResolvedPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
  try {
    $Bytes = New-Object byte[] $Stream.Length
    $Read = $Stream.Read($Bytes, 0, $Bytes.Length)
    if ($Read -lt $Bytes.Length) {
      [Array]::Resize([ref]$Bytes, $Read)
    }
  } finally {
    $Stream.Dispose()
  }
  $Text = if ($Bytes.Length -ge 2 -and $Bytes[0] -eq 0xff -and $Bytes[1] -eq 0xfe) {
    [Text.Encoding]::Unicode.GetString($Bytes)
  } elseif ($Bytes.Length -ge 2 -and $Bytes[0] -eq 0xfe -and $Bytes[1] -eq 0xff) {
    [Text.Encoding]::BigEndianUnicode.GetString($Bytes)
  } else {
    [Text.Encoding]::UTF8.GetString($Bytes)
  }
  $Text -replace "`0", ""
}

$LogText = Read-ProgressLogText -Path $LogPath
$StepPattern = 'step=(\d+)\s+train_loss=([0-9.]+)\s+val_loss=([0-9.]+)'
$Matches = [regex]::Matches($LogText, $StepPattern)
if (-not $Matches) {
  if ($InitialProcess -or $InitialActiveLock) {
    Write-EarlyProgressReport `
      -Status "running_waiting_for_first_eval" `
      -ProcessObject $InitialProcess `
      -ActiveLockObject $InitialActiveLock `
      -Reason "training log exists but no eval step line has been emitted yet"
    exit 0
  }
  throw "No eval step lines found in $LogPath"
}

$Rows = foreach ($Match in $Matches) {
  $Groups = $Match.Groups
  [pscustomobject]@{
    Step = [int]$Groups[1].Value
    TrainLoss = [double]$Groups[2].Value
    ValLoss = [double]$Groups[3].Value
    Line = $Match.Value
  }
}

$Latest = $Rows | Sort-Object Step | Select-Object -Last 1
$Best = $Rows | Sort-Object ValLoss, Step | Select-Object -First 1
$LogItem = Get-Item $LogPath
$BestItem = if (Test-Path $BestPath) { Get-Item $BestPath } else { $null }

$Process = $InitialProcess

$StartedAt = if ($Process) { $Process.StartTime } else { $LogItem.CreationTime }
$Elapsed = $LogItem.LastWriteTime - $StartedAt
$StepsPerMinute = if ($Latest.Step -gt 0 -and $Elapsed.TotalMinutes -gt 0) {
  $Latest.Step / $Elapsed.TotalMinutes
} else {
  0
}
$RemainingSteps = [Math]::Max(0, $TotalSteps - $Latest.Step)
$EarlyStopMatches = [regex]::Matches($LogText, '(?m)^early stopping:.*$')
$EarlyStopLine = if ($EarlyStopMatches.Count -gt 0) { $EarlyStopMatches[$EarlyStopMatches.Count - 1].Value } else { "" }
$LogCompleted = [bool]$EarlyStopLine -or ($Latest.Step -ge $TotalSteps)
$StopReason = if ($EarlyStopLine) {
  $EarlyStopLine
} elseif ($Latest.Step -ge $TotalSteps) {
  "reached_total_steps"
} else {
  ""
}

$ErrBytes = if (Test-Path $ErrPath) { (Get-Item $ErrPath).Length } else { 0 }
$ExitSentinel = $null
$SentinelOk = $false
$SentinelStatus = ""
if (Test-Path $ExitSentinelPath) {
  $ExitSentinel = Get-Content -Raw -Encoding UTF8 $ExitSentinelPath | ConvertFrom-Json
  $ExpectedCheckpoint = "checkpoints\${RunId}.pt"
  $ExpectedBestCheckpoint = "checkpoints\${RunId}_best.pt"
  function Normalize-RepoPath {
    param([Parameter(Mandatory = $true)][string]$PathText)
    $PathText = $PathText -replace '/', '\'
    $PathObject = [IO.Path]::GetFullPath((Join-Path $Root $PathText))
    return $PathObject.TrimEnd('\').ToLowerInvariant()
  }
  $CompletedAtOk = $false
  try {
    $CompletedAt = [datetimeoffset]::Parse([string]$ExitSentinel.completed_at)
    $StartedAtOffset = [datetimeoffset]::new($StartedAt)
    $CompletedAtOk = $CompletedAt -ge $StartedAtOffset
  } catch {
    $CompletedAtOk = $false
  }
  $CheckpointOk = $false
  $BestCheckpointOk = $false
  $CheckpointFresh = $false
  $BestCheckpointFresh = $false
  try {
    $CheckpointOk = (Normalize-RepoPath $ExpectedCheckpoint) -eq (Normalize-RepoPath ([string]$ExitSentinel.checkpoint))
    $BestCheckpointOk = (Normalize-RepoPath $ExpectedBestCheckpoint) -eq (Normalize-RepoPath ([string]$ExitSentinel.best_checkpoint))
    $CheckpointItem = if (Test-Path $ExpectedCheckpoint) { Get-Item $ExpectedCheckpoint } else { $null }
    $ExpectedBestItem = if (Test-Path $ExpectedBestCheckpoint) { Get-Item $ExpectedBestCheckpoint } else { $null }
    if ($CheckpointItem -and $ExpectedBestItem -and $CompletedAtOk) {
      $CompletedAtOk = $CompletedAtOk -and ($CompletedAt -ge ([datetimeoffset]::new($LogItem.LastWriteTime)))
      $CheckpointFresh = $CompletedAt -ge ([datetimeoffset]::new($CheckpointItem.LastWriteTime))
      $BestCheckpointFresh = $CompletedAt -ge ([datetimeoffset]::new($ExpectedBestItem.LastWriteTime))
    }
  } catch {
    $CheckpointOk = $false
    $BestCheckpointOk = $false
  }
  $SentinelOk = (
    $ExitSentinel.run_id -eq $RunId -and
    [int]$ExitSentinel.exit_code -eq 0 -and
    -not [bool]$ExitSentinel.hf_export -and
    $CompletedAtOk -and
    $CheckpointOk -and
    $BestCheckpointOk -and
    $CheckpointFresh -and
    $BestCheckpointFresh
  )
  $SentinelStatus = if ($SentinelOk) {
    "verified_exit_0_fresh_exact_paths"
  } elseif ([bool]$ExitSentinel.hf_export) {
    "present_but_rejected_hf_export_recorded"
  } else {
    "present_but_unverified_or_mismatched"
  }
} elseif ($LogCompleted) {
  $SentinelStatus = "missing_unverified_completed_run"
} else {
  $SentinelStatus = "missing_while_running_or_unknown"
}
$Completed = if ($ExitSentinel) {
  $SentinelOk
} else {
  $false
}
$EvalStatus = "missing"
if (Test-Path $EvalResultsPath) {
  try {
    $EvalPayload = Get-Content -Raw -Encoding UTF8 $EvalResultsPath | ConvertFrom-Json
    $EvalStatus = if ($EvalPayload.status) { [string]$EvalPayload.status } else { "present_unknown" }
  } catch {
    $EvalStatus = "present_invalid_json"
  }
}
$NonReleaseRecorded = Test-Path $NonReleaseRecordPath
$ActiveLockMatchesRun = $false
if (Test-Path $ActiveLockPath) {
  try {
    $ActiveLockPayload = Get-Content -Raw -Encoding UTF8 $ActiveLockPath | ConvertFrom-Json
    $ActiveLockMatchesRun = ([string]$ActiveLockPayload.run_id) -eq $RunId
  } catch {
    $ActiveLockMatchesRun = $true
  }
}
$Eta = if (-not $Completed -and $StepsPerMinute -gt 0) {
  $LogItem.LastWriteTime.AddMinutes($RemainingSteps / $StepsPerMinute)
} else {
  $null
}
$ProgressStatus = if ($ExitSentinel) {
  if (-not $Completed) {
    "training_exit_failed_or_mismatched_sentinel"
  } elseif ($NonReleaseRecorded) {
    "training_exit_verified_but_non_release_recorded"
  } elseif ($EvalStatus -eq "failed") {
    "training_exit_verified_but_quality_failed"
  } elseif ($EvalStatus -eq "passed" -and -not $ActiveLockMatchesRun) {
    "post_run_quality_passed"
  } elseif ($ActiveLockMatchesRun) {
    "training_exit_verified_finalizer_or_supervisor_pending"
  } else {
    "training_exit_verified_quality_pending"
  }
} elseif ($LogCompleted) {
  "log_completed_unverified_missing_sentinel"
} elseif ($ProcessId -gt 0 -and $Process) {
  "running_verified_process"
} elseif ($ProcessId -gt 0) {
  "process_missing_unverified"
} else {
  "unknown_unverified_no_process_or_sentinel"
}
$ProcessStatus = if ($ProcessId -le 0) {
  "not_checked_process_id_required"
} elseif ($Process) {
  "running"
} elseif ($ProgressStatus -eq "post_run_quality_passed") {
  "post_run_quality_passed"
} else {
  "not_found"
}

$Report = [pscustomobject]@{
  RunId = $RunId
  LatestStep = $Latest.Step
  TotalSteps = $TotalSteps
  TrainLoss = $Latest.TrainLoss
  ValLoss = $Latest.ValLoss
  BestStep = $Best.Step
  BestValLoss = $Best.ValLoss
  Completed = $Completed
  LogCompleted = $LogCompleted
  ProgressStatus = $ProgressStatus
  StopReason = $StopReason
  StartedAt = $StartedAt
  LastEvalAt = $LogItem.LastWriteTime
  StepsPerMinute = [Math]::Round($StepsPerMinute, 2)
  EstimatedFinishAt = $Eta
  TrainingPid = if ($Process) { $Process.Id } else { $null }
  ProcessStatus = $ProcessStatus
  ExitSentinelStatus = $SentinelStatus
  ExitSentinelPath = $ExitSentinelPath
  EvalResultsPath = $EvalResultsPath
  EvalStatus = $EvalStatus
  NonReleaseRecorded = $NonReleaseRecorded
  ActiveLockMatchesRun = $ActiveLockMatchesRun
  BestCheckpointBytes = if ($BestItem) { $BestItem.Length } else { 0 }
  StderrBytes = $ErrBytes
}
$Report | Format-List

if ($ProgressStatus -eq "post_run_quality_passed") {
  exit 0
}
if ($ProcessId -gt 0 -and $Process -and -not $LogCompleted) {
  exit 0
}
if ($ExitSentinel -and -not $Completed) {
  exit 2
}
if ($LogCompleted) {
  exit 2
}
if ($ProcessId -gt 0 -and -not $Process) {
  exit 2
}
if ($ProcessId -le 0 -and -not $ExitSentinel) {
  exit 2
}
