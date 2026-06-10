$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$RunId = "old_japanese_0_1b_dml_entrypoint_gate_failtest_$PID"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  throw "Missing project Python for entrypoint gate test: $Python"
}
$ActiveLock = "logs\active_old_japanese_0_1b_dml.lock"
$ActiveLockBeforeExists = Test-Path $ActiveLock
$ActiveLockBefore = if ($ActiveLockBeforeExists) {
  Get-Content -Raw -Encoding UTF8 $ActiveLock
} else {
  ""
}

function Get-Sha256Text {
  param([Parameter(Mandatory = $true)][string]$Text)
  $Bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($Text)
  $Hash = [System.Security.Cryptography.SHA256]::Create().ComputeHash($Bytes)
  return -join ($Hash | ForEach-Object { $_.ToString("x2") })
}

function Restore-ActiveLock {
  if ($ActiveLockBeforeExists) {
    [System.IO.File]::WriteAllText(
      [IO.Path]::GetFullPath((Join-Path $Root $ActiveLock)),
      $ActiveLockBefore,
      [System.Text.UTF8Encoding]::new($false)
    )
  } elseif (Test-Path $ActiveLock) {
    Remove-Item -LiteralPath $ActiveLock -Force
  }
}

function Invoke-ExpectedFailure {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Expected,
    [Parameter(Mandatory = $true)]
    [scriptblock]$Command
  )
  $PreviousErrorActionPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    $Output = & $Command 2>&1
    $ExitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
  }
  $OutputText = ($Output | Out-String)
  if ($ExitCode -eq 0) {
    throw "Expected entrypoint failure but command succeeded. Expected: $Expected"
  }
  if ($OutputText -notlike "*$Expected*") {
    throw "Entrypoint failed for the wrong reason. Expected: $Expected Output: $OutputText"
  }
}

Invoke-ExpectedFailure `
  -Expected "Release-candidate training must be started through scripts\start_old_japanese_0_1b_dml_and_watch.ps1" `
  -Command { powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_old_japanese_0_1b_dml.ps1 -RunId $RunId }

Invoke-ExpectedFailure `
  -Expected "Missing or mismatched supervisor RunId context." `
  -Command { powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_old_japanese_0_1b_dml.ps1 -RunId $RunId -LaunchedBySupervisor }

Invoke-ExpectedFailure `
  -Expected "release-shaped training requires --require-supervisor and the supervised launcher context." `
  -Command { & $Python -m kobun_llm.train --run-id $RunId --device cpu --steps 1 }

$SupervisorToken = "entrypoint-gate-test-$PID"
$AutonomousNonce = "entrypoint-gate-nonce-$PID"
$FakeLock = [pscustomobject]@{
  run_id = $RunId
  launcher_pid = $PID
  train_pid = $null
  watcher_pid = $null
  state = "train_started_watcher_pending"
  created_at = Get-Date -Format o
  launch_token_sha256 = Get-Sha256Text $SupervisorToken
  launch_nonce_sha256 = Get-Sha256Text $AutonomousNonce
  preflight_gate = "logs\preflight_gate_old_japanese_0_1b.json"
  preflight_gate_sha256 = if (Test-Path "logs\preflight_gate_old_japanese_0_1b.json") { (Get-FileHash -Algorithm SHA256 -LiteralPath "logs\preflight_gate_old_japanese_0_1b.json").Hash.ToLowerInvariant() } else { "0" * 64 }
  autonomous_launch_context = "logs\missing_autonomous_context_for_entrypoint_gate.json"
  autonomous_launch_context_sha256 = "0" * 64
  autonomous_pid = $PID
  autonomous_script = "scripts\autonomous_old_japanese_0_1b_loop.ps1"
  selected_action = "prepare_next_fresh_run_after_static_gate_and_zero_base_reviews"
  hf_export = $false
} | ConvertTo-Json -Depth 5
[System.IO.File]::WriteAllText(
  [IO.Path]::GetFullPath((Join-Path $Root $ActiveLock)),
  $FakeLock + [Environment]::NewLine,
  [System.Text.UTF8Encoding]::new($false)
)
try {
  $env:OLD_JAPANESE_SUPERVISOR_RUN_ID = $RunId
  $env:OLD_JAPANESE_SUPERVISOR_TOKEN = $SupervisorToken
  $env:OLD_JAPANESE_AUTONOMOUS_LAUNCH_NONCE = $AutonomousNonce
  $env:OLD_JAPANESE_ACTIVE_LOCK = [IO.Path]::GetFullPath((Join-Path $Root $ActiveLock))
  $env:OLD_JAPANESE_PREFLIGHT_GATE = [IO.Path]::GetFullPath((Join-Path $Root "logs\preflight_gate_old_japanese_0_1b.json"))
  Remove-Item Env:\OLD_JAPANESE_REVIEW_GATE -ErrorAction SilentlyContinue
  Invoke-ExpectedFailure `
    -Expected "active-run lock is missing train_pid." `
    -Command { & $Python -m kobun_llm.train --run-id $RunId --require-supervisor --device cpu --steps 1 }
} finally {
  Remove-Item Env:\OLD_JAPANESE_SUPERVISOR_RUN_ID -ErrorAction SilentlyContinue
  Remove-Item Env:\OLD_JAPANESE_SUPERVISOR_TOKEN -ErrorAction SilentlyContinue
  Remove-Item Env:\OLD_JAPANESE_AUTONOMOUS_LAUNCH_NONCE -ErrorAction SilentlyContinue
  Remove-Item Env:\OLD_JAPANESE_ACTIVE_LOCK -ErrorAction SilentlyContinue
  Remove-Item Env:\OLD_JAPANESE_PREFLIGHT_GATE -ErrorAction SilentlyContinue
  Remove-Item Env:\OLD_JAPANESE_REVIEW_GATE -ErrorAction SilentlyContinue
  Remove-Item Env:\OLD_JAPANESE_AUTONOMOUS_LAUNCH_CONTEXT -ErrorAction SilentlyContinue
  Restore-ActiveLock
}

$Live = @(Get-CimInstance Win32_Process | Where-Object {
  [string]$_.CommandLine -like "*$RunId*"
})
if ($Live.Count -gt 0) {
  $Preview = ($Live | ForEach-Object { "pid=$($_.ProcessId) command=$($_.CommandLine)" }) -join [Environment]::NewLine
  throw "Release training entrypoint gate left live processes for ${RunId}:`n$Preview"
}

$Artifacts = @(
  "checkpoints\${RunId}.pt",
  "checkpoints\${RunId}_best.pt",
  "logs\train_exit_${RunId}.json",
  "logs\${RunId}.out.log",
  "logs\${RunId}.err.log",
  "data\run_snapshots\${RunId}"
) | Where-Object { Test-Path $_ }
if ($Artifacts.Count -gt 0) {
  throw "Release training entrypoint gate created artifacts: $($Artifacts -join ', ')"
}

Write-Output "release_training_entrypoint_gates_ok=true"
exit 0
