param([switch]$AllowNonReleaseMutableInputs)

$ErrorActionPreference = "Stop"
if (-not $AllowNonReleaseMutableInputs) {
  throw "Deprecated non-release script uses mutable data/ inputs. Use train_old_japanese_0_1b_gpu.ps1 for release-candidate runs, or pass -AllowNonReleaseMutableInputs for a local experiment only."
}
Set-Location (Split-Path -Parent $PSScriptRoot)
$RunId = Get-Date -Format "yyyyMMdd_HHmmss"

.\.venv\Scripts\python.exe scripts\build_waka_training_corpus.py
.\.venv\Scripts\python.exe scripts\build_manifest.py
.\.venv\Scripts\python.exe scripts\build_training_corpus.py

.\.venv\Scripts\python.exe -m kobun_llm.train `
  --data data/kobun_labeled_grammar_train.txt `
  --val-data data/kobun_labeled_grammar_val.txt `
  --out "checkpoints/kobun_genji_large_gpu_$RunId.pt" `
  --best-out "checkpoints/kobun_genji_large_gpu_${RunId}_best.pt" `
  --run-id $RunId `
  --steps 6000 `
  --batch-size 16 `
  --block-size 256 `
  --n-layer 6 `
  --n-head 8 `
  --n-embd 256 `
  --eval-every 500 `
  --save-every 1000 `
  --early-stop-patience 6 `
  --optimizer simple-adamw `
  --amp `
  --device cuda
