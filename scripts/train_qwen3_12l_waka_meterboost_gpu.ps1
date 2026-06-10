param([switch]$AllowNonReleaseMutableInputs)

$ErrorActionPreference = "Stop"
throw "Deprecated non-release script is disabled because it mutates data/ inputs before reaching the current run-id gate. Use the supervised old-japanese-0.1B DirectML path, or create a new isolated non-release experiment launcher."
Set-Location (Split-Path -Parent $PSScriptRoot)
$RunId = Get-Date -Format "yyyyMMdd_HHmmss"

.\.venv\Scripts\python.exe scripts\build_waka_training_corpus.py
.\.venv\Scripts\python.exe scripts\build_manifest.py
.\.venv\Scripts\python.exe scripts\build_waka_meter_corpus.py
.\.venv\Scripts\python.exe scripts\build_training_corpus.py
.\.venv\Scripts\python.exe scripts\build_preference_boost_corpus.py
.\.venv\Scripts\python.exe scripts\build_worldclass_corpus.py --waka-meter-repeat 12 --rule-repeat 6

$CheckpointOutput = .\.venv\Scripts\python.exe scripts\latest_valid_checkpoint.py `
  --pattern "checkpoints/kobun_qwen3_12l_worldclass_*_best.pt"
$InitCheckpoint = ($CheckpointOutput | Select-Object -Last 1).Trim()
if (-not $InitCheckpoint) {
  throw "No valid worldclass checkpoint found."
}
Write-Host "initializing_from=$InitCheckpoint"

.\.venv\Scripts\python.exe -u -m kobun_llm.train `
  --data data/kobun_worldclass_corpus.txt `
  --val-data data/kobun_labeled_grammar_val.txt `
  --init-from $InitCheckpoint `
  --out "checkpoints/kobun_qwen3_12l_waka_meterboost_$RunId.pt" `
  --best-out "checkpoints/kobun_qwen3_12l_waka_meterboost_${RunId}_best.pt" `
  --run-id $RunId `
  --steps 1000 `
  --batch-size 6 `
  --eval-every 100 `
  --save-every 500 `
  --early-stop-patience 6 `
  --optimizer simple-adamw `
  --lr 8e-5 `
  --min-lr 1e-5 `
  --warmup-steps 50 `
  --cosine-lr `
  --grad-clip 1.0 `
  --amp `
  --device cuda
