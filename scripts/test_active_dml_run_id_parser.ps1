$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Script = Join-Path $Root "scripts\start_old_japanese_0_1b_dml_and_watch.ps1"
$Text = Get-Content -Raw -Encoding UTF8 $Script

$Start = $Text.IndexOf("function Get-DmlRunIdFromCommandLine")
if ($Start -lt 0) {
  throw "missing Get-DmlRunIdFromCommandLine"
}
$End = $Text.IndexOf("function Get-ActiveDmlRunProcesses", $Start)
if ($End -lt 0) {
  throw "missing Get-ActiveDmlRunProcesses"
}
$FunctionText = $Text.Substring($Start, $End - $Start)
Invoke-Expression $FunctionText

function Assert-RunId {
  param(
    [string]$CommandLine,
    [string]$Expected
  )
  $Actual = Get-DmlRunIdFromCommandLine -CommandLine $CommandLine
  if ($Actual -ne $Expected) {
    throw "run id parser mismatch expected='$Expected' actual='$Actual' command='$CommandLine'"
  }
}

Assert-RunId 'powershell -File scripts\start_old_japanese_0_1b_dml_and_watch.ps1' ''
Assert-RunId 'powershell -File scripts\start_old_japanese_0_1b_dml_and_watch.ps1 -RunId old_japanese_0_1b_dml_20260510_235959' 'old_japanese_0_1b_dml_20260510_235959'
Assert-RunId 'python -m kobun_llm.train --run-id old_japanese_0_1b_dml_20260510_235959 --device dml' 'old_japanese_0_1b_dml_20260510_235959'
Assert-RunId 'powershell -File scripts\train_old_japanese_0_1b_dml.ps1 -RunId=old_japanese_0_1b_dml_20260510_235959' 'old_japanese_0_1b_dml_20260510_235959'
Assert-RunId 'powershell -Command "powershell -File scripts\start_old_japanese_0_1b_dml_and_watch.ps1"' ''

"active_dml_run_id_parser_ok=true"
