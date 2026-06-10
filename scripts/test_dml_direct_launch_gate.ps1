$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$RunId = "old_japanese_0_1b_dml_direct_gate_failtest_$PID"
$ActiveLock = "logs\active_old_japanese_0_1b_dml.lock"
$ActiveLockBeforeExists = Test-Path $ActiveLock
$ActiveLockBefore = if ($ActiveLockBeforeExists) {
  Get-Content -Raw -Encoding UTF8 $ActiveLock
} else {
  ""
}

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  $Output = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_old_japanese_0_1b_dml_and_watch.ps1 -RunId $RunId 2>&1
  $ExitCode = $LASTEXITCODE
} finally {
  $ErrorActionPreference = $PreviousErrorActionPreference
}
$OutputText = ($Output | Out-String)

if ($ExitCode -eq 0) {
  throw "Direct DML launcher unexpectedly succeeded without explicit safe flags."
}
if ($OutputText -notlike "*Refusing DirectML launch without -AllowStartTraining.*") {
  throw "Direct DML launcher failed for the wrong reason. Output: $OutputText"
}

$ActiveLockAfterExists = Test-Path $ActiveLock
$ActiveLockAfter = if ($ActiveLockAfterExists) {
  Get-Content -Raw -Encoding UTF8 $ActiveLock
} else {
  ""
}
if ($ActiveLockBeforeExists -ne $ActiveLockAfterExists -or $ActiveLockBefore -ne $ActiveLockAfter) {
  throw "Direct DML launch gate changed the active lock."
}

$Live = @(Get-CimInstance Win32_Process | Where-Object {
  [string]$_.CommandLine -like "*$RunId*"
})
if ($Live.Count -gt 0) {
  $Preview = ($Live | ForEach-Object { "pid=$($_.ProcessId) command=$($_.CommandLine)" }) -join [Environment]::NewLine
  throw "Direct DML launch gate left live processes for ${RunId}:`n$Preview"
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
  throw "Direct DML launch gate created artifacts: $($Artifacts -join ', ')"
}

Write-Output "dml_direct_launch_gate_ok=true"
exit 0
