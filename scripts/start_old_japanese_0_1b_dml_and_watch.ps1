param(
  [string]$RunId = "",
  [string]$PreflightGate = "logs\preflight_gate_old_japanese_0_1b.json",
  [string]$ReviewGate = "logs\zero_base_review_gate_old_japanese_0_1b.json",
  [string]$AutonomousLaunchContext = "",
  [switch]$AllowStartTraining,
  [switch]$ReviewsPassed
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $RunId) {
  $RunId = "old_japanese_0_1b_dml_$(Get-Date -Format "yyyyMMdd_HHmmss")"
} elseif ($RunId -match '^old_japanese_0_1b_' -and $RunId -notmatch '^old_japanese_0_1b_dml_') {
  throw "Invalid DML RunId: non-DML old_japanese_0_1b_* ids are not accepted: $RunId"
} elseif ($RunId -notmatch '^old_japanese_0_1b_dml_') {
  if ($RunId -match '^dml_') {
    $RunId = $RunId -replace '^dml_', ''
  }
  $RunId = "old_japanese_0_1b_dml_$RunId"
}
if ($RunId -notmatch '^old_japanese_0_1b_dml_[0-9A-Za-z][0-9A-Za-z_-]{0,63}$') {
  throw "Invalid RunId: $RunId"
}
& (Join-Path $Root ".venv\Scripts\python.exe") -c "from kobun_autonomy.release_policy import require_release_candidate_run; import sys; require_release_candidate_run(sys.argv[1], context='DML supervisor launcher')" $RunId
if ($LASTEXITCODE -ne 0) {
  throw "DML supervisor launcher refuses known non-release RunId: $RunId"
}
if (-not $AllowStartTraining.IsPresent) {
  throw "Refusing DirectML launch without -AllowStartTraining."
}
if (-not $ReviewsPassed.IsPresent) {
  throw "Refusing DirectML launch without -ReviewsPassed."
}
& (Join-Path $Root ".venv\Scripts\python.exe") scripts\verify_preflight_gate.py --gate $PreflightGate --max-age-minutes 120
if ($LASTEXITCODE -ne 0) {
  throw "Refusing DirectML launch because the preflight gate is missing, stale, or mismatched: $PreflightGate"
}
& (Join-Path $Root ".venv\Scripts\python.exe") scripts\verify_zero_base_review_gate.py --gate $ReviewGate --preflight-gate $PreflightGate --max-age-minutes 120
if ($LASTEXITCODE -ne 0) {
  throw "Refusing DirectML launch because the zero-base review gate is missing, stale, incomplete, or mismatched: $ReviewGate"
}

New-Item -ItemType Directory -Force logs | Out-Null
$LaunchOut = "logs\launch_${RunId}.out.log"
$LaunchErr = "logs\launch_${RunId}.err.log"
$WatchOut = "logs\watch_start_${RunId}.out.log"
$WatchErr = "logs\watch_start_${RunId}.err.log"
$ActiveLock = "logs\active_old_japanese_0_1b_dml.lock"
$OtherBackendActiveLock = "logs\active_old_japanese_0_1b_cuda.lock"
$StartupMutexLock = "logs\active_old_japanese_0_1b_training.lock"

function Get-Sha256Text {
  param([Parameter(Mandatory = $true)][string]$Text)
  $Bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($Text)
  $Hash = [System.Security.Cryptography.SHA256]::Create().ComputeHash($Bytes)
  return -join ($Hash | ForEach-Object { $_.ToString("x2") })
}

function Get-Sha256File {
  param([Parameter(Mandatory = $true)][string]$PathText)
  return (Get-FileHash -Algorithm SHA256 -LiteralPath $PathText).Hash.ToLowerInvariant()
}

function Normalize-RepoPath {
  param([Parameter(Mandatory = $true)][string]$PathText)
  $PathText = $PathText -replace '/', '\'
  return [IO.Path]::GetFullPath((Join-Path $Root $PathText)).TrimEnd('\').ToLowerInvariant()
}

function Assert-UnderRepo {
  param(
    [Parameter(Mandatory = $true)][string]$Label,
    [Parameter(Mandatory = $true)][string]$PathText
  )
  $Full = [IO.Path]::GetFullPath((Join-Path $Root $PathText))
  $FullRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
  $FullWithSlash = $Full.TrimEnd('\') + '\'
  if (-not $FullWithSlash.StartsWith($FullRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "$Label escapes repository: $PathText"
  }
  return $Full
}

function Test-PidLive {
  param([int]$PidValue)
  if ($PidValue -le 0) {
    return $false
  }
  return [bool](Get-Process -Id $PidValue -ErrorAction SilentlyContinue)
}

function Test-PidMatchesRun {
  param(
    [int]$PidValue,
    [string]$ExpectedRunId,
    [switch]$AllowLauncherWithoutRunId
  )
  if ($PidValue -le 0) {
    return $false
  }
  $Proc = Get-CimInstance Win32_Process -Filter "ProcessId=$PidValue" -ErrorAction SilentlyContinue
  if (-not $Proc) {
    return $false
  }
  $CommandLine = [string]$Proc.CommandLine
  return (
    (
      $CommandLine -like "*$ExpectedRunId*" -and
      (
        $CommandLine -like "*kobun_llm.train*" -or
        $CommandLine -like "*train_old_japanese_0_1b_dml.ps1*" -or
        $CommandLine -like "*watch_and_finalize_old_japanese_0_1b_dml.ps1*" -or
        $CommandLine -like "*start_old_japanese_0_1b_dml_and_watch.ps1*"
      )
    ) -or
    (
      $AllowLauncherWithoutRunId.IsPresent -and
      $CommandLine -like "*start_old_japanese_0_1b_dml_and_watch.ps1*"
    )
  )
}

function Get-DmlRunIdFromCommandLine {
  param([string]$CommandLine)
  if (-not $CommandLine) {
    return ""
  }
  $Patterns = @(
    '(?i)(?:^|\s)--run-id(?:\s+|=)(?<run>old_japanese_0_1b_dml_[0-9A-Za-z][0-9A-Za-z_-]{0,63})(?=\s|$|"|'')',
    '(?i)(?:^|\s)-RunId(?:\s+|=)(?<run>old_japanese_0_1b_dml_[0-9A-Za-z][0-9A-Za-z_-]{0,63})(?=\s|$|"|'')'
  )
  foreach ($Pattern in $Patterns) {
    $Match = [regex]::Match($CommandLine, $Pattern)
    if ($Match.Success) {
      return $Match.Groups["run"].Value
    }
  }
  return ""
}

function Get-ActiveDmlRunProcesses {
  param([string]$ExpectedRunId = "")
  $Processes = @(Get-CimInstance Win32_Process | Where-Object {
    $CommandLine = [string]$_.CommandLine
    if (-not $CommandLine) {
      return $false
    }
    $RunIdInCommand = Get-DmlRunIdFromCommandLine -CommandLine $CommandLine
    if ($ExpectedRunId) {
      if ($RunIdInCommand -ne $ExpectedRunId) {
        return $false
      }
    } elseif (-not $RunIdInCommand) {
      return $false
    }
    return (
      $CommandLine -like "*kobun_llm.train*" -or
      $CommandLine -like "*train_old_japanese_0_1b_dml.ps1*" -or
      $CommandLine -like "*watch_and_finalize_old_japanese_0_1b_dml.ps1*" -or
      $CommandLine -like "*start_old_japanese_0_1b_dml_and_watch.ps1*"
    )
  })
  return $Processes
}

function Test-DmlTrainingCommandLine {
  param([string]$CommandLine)
  if (-not $CommandLine) {
    return $false
  }
  return (
    $CommandLine -like "*train_old_japanese_0_1b_dml.ps1*" -or
    (
      $CommandLine -like "*kobun_llm.train*" -and
      -not ($CommandLine -match '(?i)(?:^|\s)--device(?:\s+|=)(cpu|cuda|hip)(?=\s|$|"|'')')
    )
  )
}

function Get-AnyActiveDmlTrainingProcesses {
  $Processes = @(Get-CimInstance Win32_Process | Where-Object {
    Test-DmlTrainingCommandLine -CommandLine ([string]$_.CommandLine)
  })
  return $Processes
}

function Test-SupervisedOldJapaneseCommandLine {
  param([string]$CommandLine)
  if (-not $CommandLine) {
    return $false
  }
  if ($CommandLine -notmatch 'old_japanese_0_1b_(dml|cuda)_[0-9A-Za-z][0-9A-Za-z_-]{0,63}') {
    return $false
  }
  return (
    $CommandLine -like "*kobun_llm.train*" -or
    $CommandLine -like "*train_old_japanese_0_1b_dml.ps1*" -or
    $CommandLine -like "*watch_and_finalize_old_japanese_0_1b_dml.ps1*" -or
    $CommandLine -like "*finalize_old_japanese_0_1b_dml.ps1*" -or
    $CommandLine -like "*start_old_japanese_0_1b_dml_and_watch.ps1*" -or
    $CommandLine -like "*start_old_japanese_0_1b_cuda_colab_and_watch.py*"
  )
}

function Get-AnySupervisedOldJapaneseProcesses {
  param([string]$RequestedRunId)
  $IgnorePids = @([int]$PID) + @(Get-AncestorProcessIds -PidValue $PID)
  $Processes = @(Get-CimInstance Win32_Process | Where-Object {
    $PidValue = [int]$_.ProcessId
    if ($IgnorePids -contains $PidValue) {
      return $false
    }
    $CommandLine = [string]$_.CommandLine
    if ($RequestedRunId -and $CommandLine -like "*$RequestedRunId*") {
      return $false
    }
    return Test-SupervisedOldJapaneseCommandLine -CommandLine $CommandLine
  })
  return $Processes
}

function Get-AncestorProcessIds {
  param([int]$PidValue = $PID)
  $Ids = @()
  $CurrentPid = $PidValue
  while ($CurrentPid -gt 0) {
    $Proc = Get-CimInstance Win32_Process -Filter "ProcessId=$CurrentPid" -ErrorAction SilentlyContinue
    if (-not $Proc) {
      break
    }
    $ParentPid = [int]$Proc.ParentProcessId
    if ($ParentPid -le 0 -or $Ids -contains $ParentPid) {
      break
    }
    $Ids += $ParentPid
    $CurrentPid = $ParentPid
  }
  return $Ids
}

function Assert-NoOtherDmlRunProcess {
  param([string]$RequestedRunId)
  $IgnorePids = @([int]$PID) + @(Get-AncestorProcessIds -PidValue $PID)
  $LiveProcesses = @(
    @(Get-ActiveDmlRunProcesses) + @(Get-AnyActiveDmlTrainingProcesses) |
      Where-Object { $IgnorePids -notcontains [int]$_.ProcessId } |
      Sort-Object ProcessId -Unique
  )
  if ($LiveProcesses.Count -gt 0) {
    $Preview = ($LiveProcesses | Select-Object -First 5 | ForEach-Object {
      "pid=$($_.ProcessId) command=$($_.CommandLine)"
    }) -join [Environment]::NewLine
    throw "Refusing to start a new DML run; another old_japanese_0_1b_dml training process is live:`n$Preview"
  }
  $AnyBackendProcesses = @(Get-AnySupervisedOldJapaneseProcesses -RequestedRunId $RequestedRunId)
  if ($AnyBackendProcesses.Count -gt 0) {
    $Preview = ($AnyBackendProcesses | Select-Object -First 5 | ForEach-Object {
      "pid=$($_.ProcessId) command=$($_.CommandLine)"
    }) -join [Environment]::NewLine
    throw "Refusing to start a new DML run; another supervised old_japanese_0_1b process is live:`n$Preview"
  }
}

function Stop-ProcessTree {
  param([int]$RootPid)
  if ($RootPid -le 0) {
    return
  }
  $Children = @(Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $RootPid })
  foreach ($Child in $Children) {
    Stop-ProcessTree -RootPid ([int]$Child.ProcessId)
  }
  $Process = Get-Process -Id $RootPid -ErrorAction SilentlyContinue
  if ($Process) {
    Stop-Process -Id $RootPid -Force -ErrorAction SilentlyContinue
  }
}

function Get-LockPid {
  param(
    [object]$Payload,
    [string]$Name
  )
  $Value = $Payload.$Name
  if ($null -eq $Value) {
    return 0
  }
  $Parsed = 0
  if (-not [int]::TryParse([string]$Value, [ref]$Parsed)) {
    return -1
  }
  return $Parsed
}

function Get-ActiveLockSchemaErrors {
  param([object]$Payload)
  $Errors = @()
  $RunIdValue = [string]$Payload.run_id
  if ($RunIdValue -notmatch '^old_japanese_0_1b_dml_[0-9A-Za-z][0-9A-Za-z_-]{0,63}$') {
    $Errors += "invalid_or_missing_run_id"
  }
  if ([string]$Payload.backend -ne "dml") {
    $Errors += "backend_not_dml"
  }
  if ($Payload.hf_export -ne $false) {
    $Errors += "hf_export_not_false"
  }
  if (-not [string]$Payload.state) {
    $Errors += "missing_state"
  }
  try {
    $null = [datetimeoffset]::Parse([string]$Payload.created_at)
  } catch {
    $Errors += "invalid_or_missing_created_at"
  }
  foreach ($PidName in @("launcher_pid", "train_pid", "watcher_pid")) {
    $Value = $Payload.$PidName
    if ($null -eq $Value) {
      continue
    }
    $Parsed = 0
    if (-not [int]::TryParse([string]$Value, [ref]$Parsed) -or $Parsed -lt 0) {
      $Errors += "invalid_$PidName"
    }
  }
  $TokenHash = [string]$Payload.launch_token_sha256
  if (-not $TokenHash) {
    $Errors += "missing_launch_token_sha256"
  } elseif ($TokenHash -notmatch '^[0-9a-f]{64}$') {
    $Errors += "invalid_launch_token_sha256"
  }
  $NonceHash = [string]$Payload.launch_nonce_sha256
  if (-not $NonceHash) {
    $Errors += "missing_launch_nonce_sha256"
  } elseif ($NonceHash -notmatch '^[0-9a-f]{64}$') {
    $Errors += "invalid_launch_nonce_sha256"
  }
  $RequiredEvidenceFields = @(
    @{ Name = "preflight_gate"; Missing = "missing_preflight_gate" },
    @{ Name = "preflight_gate_sha256"; Missing = "missing_preflight_gate_sha256" },
    @{ Name = "review_gate"; Missing = "missing_review_gate" },
    @{ Name = "review_gate_sha256"; Missing = "missing_review_gate_sha256" },
    @{ Name = "autonomous_launch_context"; Missing = "missing_autonomous_launch_context" },
    @{ Name = "autonomous_launch_context_sha256"; Missing = "missing_autonomous_launch_context_sha256" }
  )
  foreach ($Spec in $RequiredEvidenceFields) {
    $FieldName = [string]$Spec.Name
    if (-not [string]$Payload.$FieldName) {
      $Errors += [string]$Spec.Missing
    }
  }
  foreach ($FieldName in @("preflight_gate_sha256", "review_gate_sha256", "autonomous_launch_context_sha256")) {
    $FieldValue = [string]$Payload.$FieldName
    if ($FieldValue -and $FieldValue -notmatch '^[0-9a-f]{64}$') {
      $Errors += "invalid_$FieldName"
    }
  }
  if (-not [string]$Payload.autonomous_script) {
    $Errors += "missing_autonomous_script"
  }
  if (-not [string]$Payload.selected_action) {
    $Errors += "missing_selected_action"
  }
  return $Errors
}

function Move-InvalidActiveLockOrThrow {
  param(
    [string[]]$SchemaErrors = @(),
    [string]$Reason = "invalid"
  )
  $LiveProcesses = @(Get-AnySupervisedOldJapaneseProcesses -RequestedRunId "")
  if ($LiveProcesses.Count -gt 0) {
    $Preview = ($LiveProcesses | Select-Object -First 5 | ForEach-Object {
      "pid=$($_.ProcessId) command=$($_.CommandLine)"
    }) -join [Environment]::NewLine
    throw "Active run lock is $Reason and live supervised old_japanese_0_1b process exists: $ActiveLock errors=$($SchemaErrors -join ',')`n$Preview"
  }
  $StalePath = "logs\active_old_japanese_0_1b_dml.stale.invalid.$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
  Move-Item -LiteralPath $ActiveLock -Destination $StalePath -Force
}

function Move-ActiveLockArchive {
  param(
    [Parameter(Mandatory = $true)][string]$ArchiveRunId,
    [Parameter(Mandatory = $true)][string]$Reason
  )
  if (-not (Test-Path $ActiveLock)) {
    return ""
  }
  $SafeRunId = if ($ArchiveRunId -match '^old_japanese_0_1b_dml_[0-9A-Za-z][0-9A-Za-z_-]{0,63}$') {
    $ArchiveRunId
  } else {
    "unknown_run_id"
  }
  $SafeReason = $Reason -replace '[^0-9A-Za-z_-]', '_'
  $ArchivePath = "logs\active_old_japanese_0_1b_dml.${SafeRunId}.${SafeReason}.$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
  Move-Item -LiteralPath $ActiveLock -Destination $ArchivePath -Force
  return $ArchivePath
}

function Assert-AutonomousLaunchContext {
  param(
    [Parameter(Mandatory = $true)][string]$ContextPath,
    [Parameter(Mandatory = $true)][string]$ExpectedRunId,
    [Parameter(Mandatory = $true)][string]$ExpectedPreflightGate,
    [Parameter(Mandatory = $true)][string]$ExpectedReviewGate
  )
  if (-not $ContextPath) {
    throw "Refusing DirectML launch without autonomous launch context."
  }
  $ResolvedContext = Assert-UnderRepo -Label "Autonomous launch context" -PathText $ContextPath
  if (-not (Test-Path $ResolvedContext)) {
    throw "Autonomous launch context is missing: $ContextPath"
  }
  $Context = Get-Content -Raw -Encoding UTF8 $ResolvedContext | ConvertFrom-Json
  if ($Context.schema -ne "old_japanese_0_1b_autonomous_launch_context_v1") {
    throw "Autonomous launch context schema mismatch: $($Context.schema)"
  }
  if ($Context.run_id -ne $ExpectedRunId) {
    throw "Autonomous launch context RunId mismatch: expected=$ExpectedRunId actual=$($Context.run_id)"
  }
  if ($Context.selected_action -ne "prepare_next_fresh_run_after_static_gate_and_zero_base_reviews") {
    throw "Autonomous launch context selected unsupported action: $($Context.selected_action)"
  }
  if ($Context.hf_export -ne $false) {
    throw "Autonomous launch context must attest hf_export=false."
  }
  try {
    $GeneratedAt = [datetimeoffset]::Parse([string]$Context.generated_at_utc)
  } catch {
    throw "Autonomous launch context generated_at_utc is invalid: $($Context.generated_at_utc)"
  }
  $AgeMinutes = ([datetimeoffset]::Now - $GeneratedAt).TotalMinutes
  if ($AgeMinutes -lt -1 -or $AgeMinutes -gt 10) {
    throw "Autonomous launch context is stale or from the future: age_minutes=$AgeMinutes"
  }
  if ((Normalize-RepoPath ([string]$Context.preflight_gate)) -ne (Normalize-RepoPath $ExpectedPreflightGate)) {
    throw "Autonomous launch context preflight gate path mismatch."
  }
  if ((Normalize-RepoPath ([string]$Context.review_gate)) -ne (Normalize-RepoPath $ExpectedReviewGate)) {
    throw "Autonomous launch context review gate path mismatch."
  }
  $PreflightFull = [IO.Path]::GetFullPath((Join-Path $Root $ExpectedPreflightGate))
  $ReviewFull = [IO.Path]::GetFullPath((Join-Path $Root $ExpectedReviewGate))
  if ([string]$Context.preflight_gate_sha256 -ne (Get-Sha256File $PreflightFull)) {
    throw "Autonomous launch context preflight gate hash mismatch."
  }
  if ([string]$Context.review_gate_sha256 -ne (Get-Sha256File $ReviewFull)) {
    throw "Autonomous launch context review gate hash mismatch."
  }
  $AncestorIds = @(Get-AncestorProcessIds -PidValue $PID)
  $AutonomousPid = 0
  if (-not [int]::TryParse([string]$Context.autonomous_pid, [ref]$AutonomousPid) -or $AutonomousPid -le 0) {
    throw "Autonomous launch context autonomous_pid is invalid: $($Context.autonomous_pid)"
  }
  if ($AncestorIds -notcontains $AutonomousPid) {
    throw "Autonomous launch context was not issued by an ancestor autonomous loop process: autonomous_pid=$AutonomousPid"
  }
  if ([string]$Context.autonomous_script -ne "scripts\autonomous_old_japanese_0_1b_loop.ps1") {
    throw "Autonomous launch context autonomous_script mismatch: $($Context.autonomous_script)"
  }
  $AutonomousProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$AutonomousPid" -ErrorAction SilentlyContinue
  if (-not $AutonomousProcess) {
    throw "Autonomous launch context issuer process is not live: autonomous_pid=$AutonomousPid"
  }
  $AutonomousCommandLine = [string]$AutonomousProcess.CommandLine
  if (
    $AutonomousCommandLine -notlike "*autonomous_old_japanese_0_1b_loop.ps1*" -or
    $AutonomousCommandLine -notmatch '(?i)(?:^|\s)-Mode(?:\s+|=)(TrainWhenReady|FullLoop)(?=\s|$|"|'')' -or
    $AutonomousCommandLine -notlike "*-AllowStartTraining*" -or
    $AutonomousCommandLine -notlike "*-ReviewsPassed*"
  ) {
    throw "Autonomous launch context issuer command line is not an authorized training launch."
  }
  $Nonce = [string]$env:OLD_JAPANESE_AUTONOMOUS_LAUNCH_NONCE
  if (-not $Nonce) {
    throw "Missing autonomous launch nonce environment."
  }
  if ((Get-Sha256Text $Nonce) -ne [string]$Context.launch_nonce_sha256) {
    throw "Autonomous launch context nonce hash mismatch."
  }
  return $true
}

function Get-LaunchEvidence {
  $PreflightFull = [IO.Path]::GetFullPath((Join-Path $Root $PreflightGate))
  $ReviewFull = [IO.Path]::GetFullPath((Join-Path $Root $ReviewGate))
  $ContextFull = [IO.Path]::GetFullPath((Join-Path $Root $AutonomousLaunchContext))
  $Context = Get-Content -Raw -Encoding UTF8 $ContextFull | ConvertFrom-Json
  return [pscustomobject]@{
    preflight_gate = $PreflightGate
    preflight_gate_sha256 = Get-Sha256File $PreflightFull
    review_gate = $ReviewGate
    review_gate_sha256 = Get-Sha256File $ReviewFull
    autonomous_launch_context = $AutonomousLaunchContext
    autonomous_launch_context_sha256 = Get-Sha256File $ContextFull
    autonomous_pid = [int]$Context.autonomous_pid
    autonomous_script = [string]$Context.autonomous_script
    selected_action = [string]$Context.selected_action
    launch_nonce_sha256 = [string]$Context.launch_nonce_sha256
  }
}

function New-ActiveLockPayload {
  param(
    [Parameter(Mandatory = $true)][string]$State,
    [object]$TrainPid = $null,
    [object]$WatcherPid = $null
  )
  return [pscustomobject]@{
    run_id = $RunId
    backend = "dml"
    launcher_pid = $PID
    train_pid = $TrainPid
    watcher_pid = $WatcherPid
    state = $State
    created_at = Get-Date -Format o
    launch_token_sha256 = $LaunchTokenSha256
    launch_nonce_sha256 = $LaunchEvidence.launch_nonce_sha256
    preflight_gate = $LaunchEvidence.preflight_gate
    preflight_gate_sha256 = $LaunchEvidence.preflight_gate_sha256
    review_gate = $LaunchEvidence.review_gate
    review_gate_sha256 = $LaunchEvidence.review_gate_sha256
    autonomous_launch_context = $LaunchEvidence.autonomous_launch_context
    autonomous_launch_context_sha256 = $LaunchEvidence.autonomous_launch_context_sha256
    autonomous_pid = $LaunchEvidence.autonomous_pid
    autonomous_script = $LaunchEvidence.autonomous_script
    selected_action = $LaunchEvidence.selected_action
    hf_export = $false
  }
}

function Read-ActiveLock {
  if (-not (Test-Path $ActiveLock)) {
    return $null
  }
  try {
    $Payload = Get-Content -Raw -Encoding UTF8 $ActiveLock | ConvertFrom-Json
  } catch {
    Move-InvalidActiveLockOrThrow -Reason "invalid JSON"
    return $null
  }
  $SchemaErrors = @(Get-ActiveLockSchemaErrors -Payload $Payload)
  if ($SchemaErrors.Count -gt 0) {
    Move-InvalidActiveLockOrThrow -Reason "invalid schema" -SchemaErrors $SchemaErrors
    return $null
  }
  return $Payload
}

function Write-ActiveLock {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Payload,
    [switch]$CreateNew
  )
  $Json = $Payload | ConvertTo-Json -Depth 5
  $Full = [IO.Path]::GetFullPath((Join-Path $Root $ActiveLock))
  $Bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($Json + [Environment]::NewLine)
  if ($CreateNew) {
    $Stream = [IO.File]::Open($Full, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
      $Stream.Write($Bytes, 0, $Bytes.Length)
    } finally {
      $Stream.Dispose()
    }
  } else {
    $Stream = [IO.File]::Open($Full, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    try {
      $ExistingBytes = New-Object byte[] ([int]$Stream.Length)
      $Read = $Stream.Read($ExistingBytes, 0, $ExistingBytes.Length)
      $ExistingText = [System.Text.UTF8Encoding]::new($false, $true).GetString($ExistingBytes, 0, $Read)
      $Existing = $ExistingText | ConvertFrom-Json
      if (
        [string]$Existing.run_id -ne $RunId -or
        [int]$Existing.launcher_pid -ne [int]$PID -or
        [string]$Existing.launch_token_sha256 -ne [string]$LaunchTokenSha256
      ) {
        throw "Refusing to update active lock not owned by this launcher: existing_run=$($Existing.run_id) existing_launcher=$($Existing.launcher_pid)"
      }
      $Stream.SetLength(0)
      $Stream.Position = 0
      $Stream.Write($Bytes, 0, $Bytes.Length)
    } finally {
      $Stream.Dispose()
    }
  }
}

function Write-StartupMutexLock {
  $Payload = [pscustomobject]@{
    run_id = $RunId
    backend = "dml"
    launcher_pid = $PID
    state = "startup_mutex"
    created_at = Get-Date -Format o
    hf_export = $false
  }
  $Json = $Payload | ConvertTo-Json -Depth 4
  $Full = [IO.Path]::GetFullPath((Join-Path $Root $StartupMutexLock))
  $Bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($Json + [Environment]::NewLine)
  $Stream = [IO.File]::Open($Full, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
  try {
    $Stream.Write($Bytes, 0, $Bytes.Length)
  } finally {
    $Stream.Dispose()
  }
}

function Assert-StartupMutexAvailable {
  $Python = Join-Path $Root ".venv\Scripts\python.exe"
  $Code = @'
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path.cwd() / "scripts"))
from old_japanese_run_intel import startup_mutex_health

health = startup_mutex_health(Path.cwd())
print(json.dumps(health, ensure_ascii=False))
if health.get("hard_blockers") or health.get("exists"):
    sys.exit(1)
'@
  $Output = $Code | & $Python -
  if ($LASTEXITCODE -ne 0) {
    throw "Startup mutex is live, invalid, or unquarantined; refusing DirectML launch: $Output"
  }
  if ($Output -match "quarantined_path") {
    Write-Output "startup_mutex_health=$Output"
  }
}

function Assert-NoActiveColabCudaLease {
  $Python = Join-Path $Root ".venv\Scripts\python.exe"
  $Code = @'
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path.cwd() / "scripts"))
from old_japanese_run_intel import colab_cuda_lease_health

health = colab_cuda_lease_health(Path.cwd())
print(json.dumps(health, ensure_ascii=False))
if health.get("hard_blockers") or health.get("exists"):
    sys.exit(1)
'@
  $Output = $Code | & $Python -
  if ($LASTEXITCODE -ne 0) {
    throw "Colab CUDA active lease exists, is invalid, or is unquarantined; refusing DirectML launch: $Output"
  }
  if ($Output -match "quarantined") {
    Write-Output "colab_cuda_lease_health=$Output"
  }
}

function Remove-StartupMutexLock {
  if (-not (Test-Path $StartupMutexLock)) {
    return
  }
  try {
    $Payload = Get-Content -Raw -Encoding UTF8 $StartupMutexLock | ConvertFrom-Json
    if ([string]$Payload.run_id -eq $RunId -and [int]$Payload.launcher_pid -eq [int]$PID) {
      $ArchivePath = "logs\active_old_japanese_0_1b_training.${RunId}.released.$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
      Move-Item -LiteralPath $StartupMutexLock -Destination $ArchivePath -Force
    }
  } catch {
    Write-Warning "Could not release startup mutex lock for ${RunId}: $($_.Exception.Message)"
  }
}

function Test-ActiveLockOwnedByThisLauncher {
  if (-not (Test-Path $ActiveLock)) {
    return $false
  }
  $Full = [IO.Path]::GetFullPath((Join-Path $Root $ActiveLock))
  $Stream = $null
  try {
    $Stream = [IO.File]::Open($Full, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)
    $Bytes = New-Object byte[] ([int]$Stream.Length)
    $Read = $Stream.Read($Bytes, 0, $Bytes.Length)
    $Text = [System.Text.UTF8Encoding]::new($false, $true).GetString($Bytes, 0, $Read)
    $Payload = $Text | ConvertFrom-Json
    return (
      [string]$Payload.run_id -eq $RunId -and
      [int]$Payload.launcher_pid -eq [int]$PID -and
      [string]$Payload.launch_token_sha256 -eq [string]$LaunchTokenSha256
    )
  } catch {
    return $false
  } finally {
    if ($Stream) {
      $Stream.Dispose()
    }
  }
}

if (-not $AutonomousLaunchContext) {
  throw "Refusing DirectML launch without autonomous launch context."
}
Assert-AutonomousLaunchContext `
  -ContextPath $AutonomousLaunchContext `
  -ExpectedRunId $RunId `
  -ExpectedPreflightGate $PreflightGate `
  -ExpectedReviewGate $ReviewGate | Out-Null
$LaunchEvidence = Get-LaunchEvidence
$LaunchToken = [guid]::NewGuid().ToString("N") + [guid]::NewGuid().ToString("N")
$LaunchTokenSha256 = Get-Sha256Text $LaunchToken
try {
  Assert-NoActiveColabCudaLease
  Assert-StartupMutexAvailable
  Write-StartupMutexLock
  Assert-NoActiveColabCudaLease
  if (Test-Path $OtherBackendActiveLock) {
    throw "Refusing DirectML launch while CUDA active lock exists: $OtherBackendActiveLock"
  }
  $ExistingLock = Read-ActiveLock
  if ($ExistingLock) {
    $ExistingRunProcesses = @(Get-ActiveDmlRunProcesses -ExpectedRunId ([string]$ExistingLock.run_id))
    $CreatedAtRecent = $false
    try {
      $CreatedAt = [datetimeoffset]::Parse([string]$ExistingLock.created_at)
      $CreatedAtRecent = ([datetimeoffset]::Now - $CreatedAt).TotalMinutes -lt 30
    } catch {
      $CreatedAtRecent = $false
    }
    $LockState = [string]$ExistingLock.state
    $LauncherMayOmitRunId = $CreatedAtRecent -and ($LockState -in @("launching", "train_started_watcher_pending"))
    $Live = @(
      Test-PidMatchesRun (Get-LockPid $ExistingLock "train_pid") ([string]$ExistingLock.run_id)
      Test-PidMatchesRun (Get-LockPid $ExistingLock "watcher_pid") ([string]$ExistingLock.run_id)
      Test-PidMatchesRun (Get-LockPid $ExistingLock "launcher_pid") ([string]$ExistingLock.run_id) -AllowLauncherWithoutRunId:$LauncherMayOmitRunId
      ($ExistingRunProcesses.Count -gt 0)
    ) -contains $true
    if ($Live) {
      throw "Refusing to start a duplicate DML run; active lock is live: $ActiveLock run_id=$($ExistingLock.run_id)"
    }
    $null = Move-ActiveLockArchive -ArchiveRunId ([string]$ExistingLock.run_id) -Reason "stale"
  }
  Assert-NoOtherDmlRunProcess -RequestedRunId $RunId
  & (Join-Path $Root ".venv\Scripts\python.exe") scripts\assert_run_id_unused.py --run-id $RunId --allow-supervisor-launch-artifacts
  if ($LASTEXITCODE -ne 0) {
    throw "Refusing to reuse RunId ${RunId}; run-scoped artifact already exists."
  }
  Write-ActiveLock -CreateNew -Payload (New-ActiveLockPayload -State "launching")
} finally {
  Remove-StartupMutexLock
}

try {
  $env:OLD_JAPANESE_SUPERVISOR_RUN_ID = $RunId
  $env:OLD_JAPANESE_SUPERVISOR_TOKEN = $LaunchToken
  $env:OLD_JAPANESE_ACTIVE_LOCK = [IO.Path]::GetFullPath((Join-Path $Root $ActiveLock))
  $env:OLD_JAPANESE_PREFLIGHT_GATE = [IO.Path]::GetFullPath((Join-Path $Root $PreflightGate))
  $env:OLD_JAPANESE_REVIEW_GATE = [IO.Path]::GetFullPath((Join-Path $Root $ReviewGate))
  $env:OLD_JAPANESE_AUTONOMOUS_LAUNCH_CONTEXT = [IO.Path]::GetFullPath((Join-Path $Root $AutonomousLaunchContext))
  $Train = Start-Process powershell `
  -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    "scripts\train_old_japanese_0_1b_dml.ps1",
    "-RunId",
    $RunId,
    "-LaunchedBySupervisor"
  ) `
  -RedirectStandardOutput $LaunchOut `
  -RedirectStandardError $LaunchErr `
  -PassThru `
  -WindowStyle Hidden
  Write-ActiveLock -Payload (New-ActiveLockPayload -State "train_started_watcher_pending" -TrainPid $Train.Id)
  Remove-Item Env:\OLD_JAPANESE_SUPERVISOR_RUN_ID -ErrorAction SilentlyContinue
  Remove-Item Env:\OLD_JAPANESE_SUPERVISOR_TOKEN -ErrorAction SilentlyContinue
  Remove-Item Env:\OLD_JAPANESE_ACTIVE_LOCK -ErrorAction SilentlyContinue
  Remove-Item Env:\OLD_JAPANESE_PREFLIGHT_GATE -ErrorAction SilentlyContinue
  Remove-Item Env:\OLD_JAPANESE_REVIEW_GATE -ErrorAction SilentlyContinue
  Remove-Item Env:\OLD_JAPANESE_AUTONOMOUS_LAUNCH_CONTEXT -ErrorAction SilentlyContinue

  $Watcher = Start-Process powershell `
  -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    "scripts\watch_and_finalize_old_japanese_0_1b_dml.ps1",
    "-ProcessId",
    "$($Train.Id)",
    "-RunId",
    $RunId
  ) `
  -RedirectStandardOutput $WatchOut `
  -RedirectStandardError $WatchErr `
  -PassThru `
  -WindowStyle Hidden

  Write-ActiveLock -Payload (New-ActiveLockPayload -State "running" -TrainPid $Train.Id -WatcherPid $Watcher.Id)
  Start-Sleep -Seconds 2
  if (-not (Test-PidMatchesRun $Train.Id $RunId)) {
    throw "DML train wrapper is not live or does not match RunId after launch: run_id=$RunId train_pid=$($Train.Id)"
  }
  if (-not (Test-PidMatchesRun $Watcher.Id $RunId)) {
    throw "DML watcher is not live or does not match RunId after launch: run_id=$RunId watcher_pid=$($Watcher.Id)"
  }
  $VerifiedLock = Read-ActiveLock
  if (-not $VerifiedLock -or $VerifiedLock.run_id -ne $RunId -or $VerifiedLock.state -ne "running") {
    throw "DML active lock was not persisted in running state after launch: run_id=$RunId"
  }
} catch {
  Remove-Item Env:\OLD_JAPANESE_SUPERVISOR_RUN_ID -ErrorAction SilentlyContinue
  Remove-Item Env:\OLD_JAPANESE_SUPERVISOR_TOKEN -ErrorAction SilentlyContinue
  Remove-Item Env:\OLD_JAPANESE_ACTIVE_LOCK -ErrorAction SilentlyContinue
  Remove-Item Env:\OLD_JAPANESE_PREFLIGHT_GATE -ErrorAction SilentlyContinue
  Remove-Item Env:\OLD_JAPANESE_REVIEW_GATE -ErrorAction SilentlyContinue
  Remove-Item Env:\OLD_JAPANESE_AUTONOMOUS_LAUNCH_CONTEXT -ErrorAction SilentlyContinue
  if ($Watcher -and (Test-PidLive $Watcher.Id)) {
    Stop-ProcessTree -RootPid $Watcher.Id
  }
  if ($Train -and (Test-PidLive $Train.Id)) {
    Stop-ProcessTree -RootPid $Train.Id
  }
  try {
    if (Test-ActiveLockOwnedByThisLauncher) {
      Write-ActiveLock -Payload (New-ActiveLockPayload `
        -State "startup_failed_stopped" `
        -TrainPid $(if ($Train) { $Train.Id } else { $null }) `
        -WatcherPid $(if ($Watcher) { $Watcher.Id } else { $null }))
    } else {
      Write-Warning "Not writing startup failure active lock for ${RunId}; active lock is absent or not owned by this launcher."
    }
  } catch {
    Write-Warning "Could not write startup failure active lock for ${RunId}: $($_.Exception.Message)"
  }
  if (Test-ActiveLockOwnedByThisLauncher) {
    $null = Move-ActiveLockArchive -ArchiveRunId $RunId -Reason "startup_failed_stopped"
  } else {
    Write-Warning "Not archiving active lock for ${RunId}; active lock is absent or not owned by this launcher."
  }
  throw
}

[pscustomobject]@{
  run_id = $RunId
  train_pid = $Train.Id
  watcher_pid = $Watcher.Id
  stdout_log = "logs\${RunId}.out.log"
  stderr_log = "logs\${RunId}.err.log"
  launch_stdout = $LaunchOut
  launch_stderr = $LaunchErr
  watch_log = "logs\watch_finalize_${RunId}.log"
  final_log = "logs\finalize_${RunId}.log"
  train_exit_sentinel = "logs\train_exit_${RunId}.json"
  best_checkpoint = "checkpoints\${RunId}_best.pt"
  hf_export = $false
} | ConvertTo-Json -Depth 3
