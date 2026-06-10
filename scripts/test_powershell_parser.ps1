$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Scripts = @(
  "scripts\autonomous_old_japanese_0_1b_loop.ps1",
  "scripts\finalize_old_japanese_0_1b_dml.ps1",
  "scripts\run_quality_checks.ps1",
  "scripts\run_quality_checks_dml.ps1",
  "scripts\run_static_quality_checks.ps1",
  "scripts\start_old_japanese_0_1b_dml_and_watch.ps1",
  "scripts\test_dml_direct_launch_gate.ps1",
  "scripts\test_dml_review_gate_launch_guard.ps1",
  "scripts\test_dml_autonomous_context_launch_guard.ps1",
  "scripts\test_release_training_entrypoint_gates.ps1",
  "scripts\stop_dml_run_as_non_release.ps1",
  "scripts\train_old_japanese_0_1b_dml.ps1",
  "scripts\train_old_japanese_0_1b_gpu.ps1",
  "scripts\watch_and_finalize_old_japanese_0_1b_dml.ps1"
)

$Failures = @()
foreach ($Script in $Scripts) {
  $FullPath = [IO.Path]::GetFullPath((Join-Path $Root $Script))
  if (-not (Test-Path $FullPath)) {
    $Failures += "${Script}: missing"
    continue
  }
  $Tokens = $null
  $ParseErrors = $null
  [System.Management.Automation.Language.Parser]::ParseFile($FullPath, [ref]$Tokens, [ref]$ParseErrors) | Out-Null
  if ($ParseErrors.Count -gt 0) {
    $Messages = ($ParseErrors | ForEach-Object { "line=$($_.Extent.StartLineNumber) message=$($_.Message)" }) -join "; "
    $Failures += "${Script}: $Messages"
  }
}

if ($Failures.Count -gt 0) {
  throw "PowerShell parser check failed: $($Failures -join ' | ')"
}

"powershell_parser_ok=true scripts=$($Scripts.Count)"
