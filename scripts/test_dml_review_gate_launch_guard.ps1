$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$RunId = "old_japanese_0_1b_dml_review_gate_failtest_$PID"
$MissingReviewGate = "logs\missing_zero_base_review_gate_$PID.json"
$ActiveLock = "logs\active_old_japanese_0_1b_dml.lock"
$PreflightGate = "logs\preflight_gate_old_japanese_0_1b.json"
if (-not (Test-Path $PreflightGate)) {
  throw "Missing preflight gate for review-gate launch guard test: $PreflightGate"
}

$ActiveLockBeforeExists = Test-Path $ActiveLock
$ActiveLockBefore = if ($ActiveLockBeforeExists) {
  Get-Content -Raw -Encoding UTF8 $ActiveLock
} else {
  ""
}

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  $Output = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_old_japanese_0_1b_dml_and_watch.ps1 -RunId $RunId -PreflightGate $PreflightGate -ReviewGate $MissingReviewGate -AllowStartTraining -ReviewsPassed 2>&1
  $ExitCode = $LASTEXITCODE
} finally {
  $ErrorActionPreference = $PreviousErrorActionPreference
}
$OutputText = ($Output | Out-String)

if ($ExitCode -eq 0) {
  throw "Direct DML launcher unexpectedly succeeded with a missing zero-base review gate."
}
if ($OutputText -notlike "*Refusing DirectML launch because the zero-base review gate is missing*") {
  throw "Direct DML launcher failed for the wrong review-gate reason. Output: $OutputText"
}

$ActiveLockAfterExists = Test-Path $ActiveLock
$ActiveLockAfter = if ($ActiveLockAfterExists) {
  Get-Content -Raw -Encoding UTF8 $ActiveLock
} else {
  ""
}
if ($ActiveLockBeforeExists -ne $ActiveLockAfterExists -or $ActiveLockBefore -ne $ActiveLockAfter) {
  throw "Missing review-gate launch guard changed the active lock."
}

$Live = @(Get-CimInstance Win32_Process | Where-Object {
  [string]$_.CommandLine -like "*$RunId*"
})
if ($Live.Count -gt 0) {
  $Preview = ($Live | ForEach-Object { "pid=$($_.ProcessId) command=$($_.CommandLine)" }) -join [Environment]::NewLine
  throw "Missing review-gate launch guard left live processes for ${RunId}:`n$Preview"
}

$Artifacts = @(
  "checkpoints\${RunId}.pt",
  "checkpoints\${RunId}_best.pt",
  "logs\train_exit_${RunId}.json",
  "logs\${RunId}.out.log",
  "logs\${RunId}.err.log",
  "logs\launch_${RunId}.out.log",
  "logs\launch_${RunId}.err.log",
  "logs\watch_start_${RunId}.out.log",
  "logs\watch_start_${RunId}.err.log",
  "data\run_snapshots\${RunId}"
) | Where-Object { Test-Path $_ }
if ($Artifacts.Count -gt 0) {
  throw "Missing review-gate launch guard created artifacts: $($Artifacts -join ', ')"
}

Write-Output "dml_review_gate_launch_guard_ok=true"
exit 0
