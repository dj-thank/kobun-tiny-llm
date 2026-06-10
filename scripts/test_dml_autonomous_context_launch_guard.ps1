$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$RunId = "old_japanese_0_1b_dml_context_gate_failtest_$PID"
$ActiveLock = "logs\active_old_japanese_0_1b_dml.lock"
$PreflightGate = "logs\preflight_gate_old_japanese_0_1b.json"
$ReviewGate = "logs\zero_base_review_context_gate_$PID.json"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $PreflightGate)) {
  throw "Missing preflight gate for autonomous-context launch guard test: $PreflightGate"
}

$ArtifactDir = "logs\zero_base_review_artifacts"
$SafetyArtifact = "$ArtifactDir\context_gate_safety_$PID.json"
$DataArtifact = "$ArtifactDir\context_gate_data_$PID.json"
$BackendArtifact = "$ArtifactDir\context_gate_backend_$PID.json"
function Remove-TestReviewArtifacts {
  Remove-Item -LiteralPath $ReviewGate -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $SafetyArtifact -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $DataArtifact -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $BackendArtifact -Force -ErrorAction SilentlyContinue
  try {
    Remove-Item -LiteralPath $ArtifactDir -Force -ErrorAction SilentlyContinue
  } catch {
  }
}
New-Item -ItemType Directory -Force $ArtifactDir | Out-Null
& $Python scripts\write_zero_base_review_artifact.py --scope safety/release --agent-id 019e17e9-b001-7101-9001-000000000001 --decision no_blockers --blocking-findings-count 0 --summary "No blocking findings in context gate safety test." --prompt "zero-base context safety test prompt" --preflight-gate $PreflightGate --out $SafetyArtifact
if ($LASTEXITCODE -ne 0) { throw "Could not write context-gate safety review artifact." }
& $Python scripts\write_zero_base_review_artifact.py --scope data/eval --agent-id 019e17e9-b002-7101-9001-000000000002 --decision no_blockers --blocking-findings-count 0 --summary "No blocking findings in context gate data test." --prompt "zero-base context data test prompt" --preflight-gate $PreflightGate --out $DataArtifact
if ($LASTEXITCODE -ne 0) { throw "Could not write context-gate data review artifact." }
& $Python scripts\write_zero_base_review_artifact.py --scope backend/runtime --agent-id 019e17e9-b003-7101-9001-000000000003 --decision no_blockers --blocking-findings-count 0 --summary "No blocking findings in context gate backend test." --prompt "zero-base context backend test prompt" --preflight-gate $PreflightGate --out $BackendArtifact
if ($LASTEXITCODE -ne 0) { throw "Could not write context-gate backend review artifact." }
& $Python scripts\write_zero_base_review_gate.py --out $ReviewGate --preflight-gate $PreflightGate --safety-artifact $SafetyArtifact --data-artifact $DataArtifact --backend-artifact $BackendArtifact
if ($LASTEXITCODE -ne 0) { throw "Could not write context-gate review gate." }

$ActiveLockBeforeExists = Test-Path $ActiveLock
$ActiveLockBefore = if ($ActiveLockBeforeExists) {
  Get-Content -Raw -Encoding UTF8 $ActiveLock
} else {
  ""
}

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  $Output = & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_old_japanese_0_1b_dml_and_watch.ps1 -RunId $RunId -PreflightGate $PreflightGate -ReviewGate $ReviewGate -AllowStartTraining -ReviewsPassed 2>&1
  $ExitCode = $LASTEXITCODE
} finally {
  $ErrorActionPreference = $PreviousErrorActionPreference
}
$OutputText = ($Output | Out-String)

if ($ExitCode -eq 0) {
  throw "Direct DML launcher unexpectedly succeeded without autonomous launch context."
}
if ($OutputText -notlike "*Refusing DirectML launch without autonomous launch context.*") {
  throw "Direct DML launcher failed for the wrong autonomous-context reason. Output: $OutputText"
}

$ActiveLockAfterExists = Test-Path $ActiveLock
$ActiveLockAfter = if ($ActiveLockAfterExists) {
  Get-Content -Raw -Encoding UTF8 $ActiveLock
} else {
  ""
}
if ($ActiveLockBeforeExists -ne $ActiveLockAfterExists -or $ActiveLockBefore -ne $ActiveLockAfter) {
  throw "Missing autonomous-context launch guard changed the active lock."
}

$Live = @(Get-CimInstance Win32_Process | Where-Object {
  [string]$_.CommandLine -like "*$RunId*"
})
if ($Live.Count -gt 0) {
  $Preview = ($Live | ForEach-Object { "pid=$($_.ProcessId) command=$($_.CommandLine)" }) -join [Environment]::NewLine
  throw "Missing autonomous-context launch guard left live processes for ${RunId}:`n$Preview"
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
  throw "Missing autonomous-context launch guard created artifacts: $($Artifacts -join ', ')"
}

Remove-TestReviewArtifacts

Write-Output "dml_autonomous_context_launch_guard_ok=true"
exit 0
