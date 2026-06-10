param(
  [Parameter(Mandatory = $true)]
  [string]$RunId,
  [Parameter(Mandatory = $true)]
  [string]$Reason
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ($RunId -notmatch '^old_japanese_0_1b_dml_[0-9A-Za-z][0-9A-Za-z_-]{0,63}$') {
  throw "Refusing to stop invalid DML RunId: $RunId"
}
if (-not $Reason.Trim()) {
  throw "A non-release reason is required."
}

New-Item -ItemType Directory -Force logs | Out-Null
New-Item -ItemType Directory -Force logs\non_release_runs | Out-Null

function Get-RunProcesses {
  param([Parameter(Mandatory = $true)][string]$ExpectedRunId)
  return @(Get-CimInstance Win32_Process | Where-Object {
    $CommandLine = [string]$_.CommandLine
    if (-not $CommandLine -or $CommandLine -notlike "*$ExpectedRunId*") {
      return $false
    }
    return (
      $CommandLine -like "*kobun_llm.train*" -or
      $CommandLine -like "*run_command_capture.py*" -or
      $CommandLine -like "*train_old_japanese_0_1b_dml.ps1*" -or
      $CommandLine -like "*watch_and_finalize_old_japanese_0_1b_dml.ps1*"
    )
  })
}

function Get-ExactTrainingProcesses {
  param([Parameter(Mandatory = $true)][string]$ExpectedRunId)
  $RunPattern = '(?i)(?:^|\s)--run-id(?:\s+|=)' + [regex]::Escape($ExpectedRunId) + '(?=\s|$|"|'')'
  return @(Get-CimInstance Win32_Process | Where-Object {
    $CommandLine = [string]$_.CommandLine
    return (
      [string]$_.Name -eq "python.exe" -and
      $CommandLine -match '(?i)(?:^|\s)-m\s+kobun_llm\.train(?=\s|$|"|'')' -and
      $CommandLine -match $RunPattern -and
      $CommandLine -match '(?i)(?:^|\s)--device(?:\s+|=)dml(?=\s|$|"|'')'
    )
  })
}

function Stop-ProcessTree {
  param([int]$RootPid)
  if ($RootPid -le 0 -or $RootPid -eq $PID) {
    return
  }
  $Children = @(Get-CimInstance Win32_Process | Where-Object { [int]$_.ParentProcessId -eq $RootPid })
  foreach ($Child in $Children) {
    Stop-ProcessTree -RootPid ([int]$Child.ProcessId)
  }
  $Process = Get-Process -Id $RootPid -ErrorAction SilentlyContinue
  if ($Process) {
    Stop-Process -Id $RootPid -Force -ErrorAction SilentlyContinue
  }
}

function Write-JsonFile {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][object]$Payload
  )
  $Json = $Payload | ConvertTo-Json -Depth 8
  [System.IO.File]::WriteAllText(
    [IO.Path]::GetFullPath((Join-Path $Root $Path)),
    $Json + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
  )
}

function Write-ExitSentinelIfMissing {
  param([string]$Message)
  $ExitSentinel = "logs\train_exit_${RunId}.json"
  if (-not (Test-Path $ExitSentinel)) {
    Write-JsonFile -Path $ExitSentinel -Payload ([pscustomobject]@{
      run_id = $RunId
      exit_code = -1
      message = $Message
      completed_at = Get-Date -Format o
      checkpoint = "checkpoints/${RunId}.pt"
      best_checkpoint = "checkpoints/${RunId}_best.pt"
      hf_export = $false
    })
  }
  return $ExitSentinel
}

function Archive-ActiveLockIfOwned {
  $ActiveLock = "logs\active_old_japanese_0_1b_dml.lock"
  if (-not (Test-Path $ActiveLock)) {
    return ""
  }
  try {
    $Lock = Get-Content -Raw -Encoding UTF8 $ActiveLock | ConvertFrom-Json
  } catch {
    $Archive = "logs\active_old_japanese_0_1b_dml.${RunId}.invalid.$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
    Move-Item -LiteralPath $ActiveLock -Destination $Archive -Force
    return $Archive
  }
  if ([string]$Lock.run_id -ne $RunId) {
    throw "Active DML lock belongs to $($Lock.run_id), not $RunId."
  }
  $Lock | Add-Member -NotePropertyName state -NotePropertyValue "non_release_stopped" -Force
  $Lock | Add-Member -NotePropertyName non_release_reason -NotePropertyValue $Reason -Force
  $Lock | Add-Member -NotePropertyName stopped_at -NotePropertyValue (Get-Date -Format o) -Force
  $Lock | Add-Member -NotePropertyName hf_export -NotePropertyValue $false -Force
  $Archive = "logs\active_old_japanese_0_1b_dml.${RunId}.non_release.$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
  Write-JsonFile -Path $Archive -Payload $Lock
  Remove-Item -LiteralPath $ActiveLock -Force
  return $Archive
}

$RunProcesses = @(Get-RunProcesses -ExpectedRunId $RunId)
if ($RunProcesses.Count -eq 0) {
  throw "No live process found for $RunId; refusing to fabricate a non-release stop."
}

$ExactTraining = @(Get-ExactTrainingProcesses -ExpectedRunId $RunId)
foreach ($Process in $ExactTraining) {
  Stop-Process -Id ([int]$Process.ProcessId) -Force -ErrorAction SilentlyContinue
}

$ExitSentinel = "logs\train_exit_${RunId}.json"
$Deadline = (Get-Date).AddSeconds(90)
while (-not (Test-Path $ExitSentinel) -and (Get-Date) -lt $Deadline) {
  Start-Sleep -Seconds 2
}

$Remaining = @(Get-RunProcesses -ExpectedRunId $RunId)
$RemainingIds = @($Remaining | ForEach-Object { [int]$_.ProcessId })
$Roots = @($Remaining | Where-Object { $RemainingIds -notcontains [int]$_.ParentProcessId })
foreach ($RootProcess in $Roots) {
  Stop-ProcessTree -RootPid ([int]$RootProcess.ProcessId)
}
Start-Sleep -Seconds 2

$StillLive = @(Get-RunProcesses -ExpectedRunId $RunId)
if ($StillLive.Count -gt 0) {
  $Preview = ($StillLive | Select-Object -First 5 | ForEach-Object {
    "pid=$($_.ProcessId) command=$($_.CommandLine)"
  }) -join [Environment]::NewLine
  throw "Non-release stop left live processes for ${RunId}:`n$Preview"
}

$ExitSentinel = Write-ExitSentinelIfMissing -Message "non_release_stop: $Reason"
$Sentinel = Get-Content -Raw -Encoding UTF8 $ExitSentinel | ConvertFrom-Json
if ([int]$Sentinel.exit_code -eq 0) {
  throw "Refusing non-release record because train exit sentinel reports success: $ExitSentinel"
}

$ArchivePath = Archive-ActiveLockIfOwned
$ArchiveRelativePath = ""
$ArchiveBasename = ""
$ArchiveSha256 = ""
if ($ArchivePath -and (Test-Path $ArchivePath)) {
  $ArchiveFullPath = [IO.Path]::GetFullPath((Join-Path $Root $ArchivePath))
  $RootFullPath = ([IO.Path]::GetFullPath($Root)).TrimEnd('\') + '\'
  if (-not $ArchiveFullPath.StartsWith($RootFullPath, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Archive path escapes project root: $ArchivePath"
  }
  $ArchiveRelativePath = $ArchiveFullPath.Substring($RootFullPath.Length).Replace('\', '/')
  $ArchiveBasename = Split-Path -Leaf $ArchivePath
  $ArchiveSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ArchivePath).Hash.ToLowerInvariant()
}
$SentinelSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ExitSentinel).Hash.ToLowerInvariant()

$RecordPath = "logs\non_release_runs\${RunId}.json"
Write-JsonFile -Path $RecordPath -Payload ([pscustomobject]@{
  run_id = $RunId
  release_status = "non_release_artifact"
  reason = $Reason
  created_at = Get-Date -Format o
  train_exit_sentinel = $ExitSentinel
  train_exit_sentinel_sha256 = $SentinelSha256
  active_lock_archive = $ArchiveBasename
  active_lock_archive_sha256 = $ArchiveSha256
  source_archive_path = $ArchiveRelativePath
  source_archive_sha256 = $ArchiveSha256
  hf_export = $false
})

Write-Output "non_release_stop_ok=true run_id=$RunId reason=$Reason"
Write-Output "non_release_record=$RecordPath"
