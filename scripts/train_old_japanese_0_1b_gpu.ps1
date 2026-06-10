param(
  [string]$RunId = "",
  [switch]$LaunchedBySupervisor
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

throw "CUDA/HIP release-candidate training is disabled until it has a supervisor launch token, active lock, watcher, finalizer, preflight gate, independent review gate, and quality gate equivalent to the DirectML path. Use scripts\autonomous_old_japanese_0_1b_loop.ps1 -Mode TrainWhenReady -AllowStartTraining -ReviewsPassed in a configured local training environment."
