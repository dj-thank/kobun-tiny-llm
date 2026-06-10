param(
  [string]$Checkpoint = "",
  [string]$RunId = "",
  [switch]$AllowIncompleteNonReleaseFailtest
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv-dml\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  throw "Missing .venv-dml Python at $Python."
}

if (-not $Checkpoint) {
  throw "Checkpoint is required. Refusing to finalize an implicit or fallback checkpoint."
}

if (-not (Test-Path $Checkpoint)) {
  throw "Checkpoint does not exist: $Checkpoint"
}

if (-not $RunId) {
  $RunId = [IO.Path]::GetFileNameWithoutExtension($Checkpoint)
  if ($RunId -match '_best$') {
    $RunId = $RunId -replace '_best$', ''
  }
}
if ($RunId -notmatch '^old_japanese_0_1b_dml_[0-9A-Za-z][0-9A-Za-z_-]{0,63}$') {
  throw "Invalid RunId: $RunId"
}
& $Python -c "from kobun_autonomy.release_policy import require_release_candidate_run; import sys; require_release_candidate_run(sys.argv[1], context='DML finalizer') " $RunId
if ($LASTEXITCODE -ne 0) {
  throw "DML finalizer refuses known non-release RunId: $RunId"
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
    throw "${Label} mismatch: expected=$Expected actual=$Actual"
  }
}
$ExpectedBestCheckpoint = "checkpoints\${RunId}_best.pt"
Assert-SameRepoPath -Label "finalize checkpoint" -Expected $ExpectedBestCheckpoint -Actual $Checkpoint

& $Python -c "from pathlib import Path; from kobun_llm.checkpoint_io import load_trusted_checkpoint; import sys; payload=load_trusted_checkpoint(Path(sys.argv[1]), map_location='cpu'); metadata=payload.get('metadata') or {}; run_id=metadata.get('run_id') or ''; backend=metadata.get('backend') or ''; sys.exit(0 if run_id == sys.argv[2] and backend == 'dml' else 7)" $Checkpoint $RunId
if ($LASTEXITCODE -ne 0) {
  throw "Checkpoint metadata run_id/backend does not match DML finalizer: checkpoint=$Checkpoint run_id=$RunId"
}

if (-not $AllowIncompleteNonReleaseFailtest.IsPresent) {
  & $Python scripts\check_run_completion.py `
    --run-id $RunId `
    --checkpoint $Checkpoint `
    --backend dml `
    --require-no-active-process
  if ($LASTEXITCODE -ne 0) {
    throw "Run completion sentinel gate failed for finalizer RunId=$RunId."
  }
}

New-Item -ItemType Directory -Force logs | Out-Null
New-Item -ItemType Directory -Force release | Out-Null

$QualityLog = "logs\quality_${RunId}.log"
$EvalJson = "logs\eval_results_${RunId}.json"

Write-Output "finalize_checkpoint=$Checkpoint"
Write-Output "quality_log=$QualityLog"
Write-Output "eval_json=$EvalJson"
Write-Output "hf_export_requested=False"

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_quality_checks_dml.ps1 -Checkpoint $Checkpoint *> $QualityLog
  $QualityExit = $LASTEXITCODE
} finally {
  $ErrorActionPreference = $PreviousErrorActionPreference
}

$ParseArgs = @(
  "scripts\parse_quality_log.py",
  "--log",
  $QualityLog,
  "--out",
  $EvalJson,
  "--checkpoint",
  $Checkpoint,
  "--device",
  "dml",
  "--runner",
  "scripts\run_quality_checks_dml.ps1",
  "--status",
  $(if ($QualityExit -eq 0) { "passed" } else { "failed" })
)

& $Python @ParseArgs
if ($LASTEXITCODE -ne 0) {
  throw "Failed to parse quality log into $EvalJson."
}

if ($QualityExit -ne 0) {
  Write-Output "quality_status=failed"
  throw "Quality checks failed with exit code $QualityExit. See $QualityLog and $EvalJson."
}

Write-Output "hf_export_status=skipped_by_finalizer_policy"
Write-Output "finalize_status=passed"
exit 0
