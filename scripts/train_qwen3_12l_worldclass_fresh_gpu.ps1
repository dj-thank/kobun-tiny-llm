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
.\.venv\Scripts\python.exe scripts\build_worldclass_corpus.py

.\.venv\Scripts\python.exe -u -m kobun_llm.train `
  --data data/kobun_worldclass_corpus.txt `
  --val-data data/kobun_labeled_grammar_val.txt `
  --out "checkpoints/kobun_qwen3_12l_worldclass_$RunId.pt" `
  --best-out "checkpoints/kobun_qwen3_12l_worldclass_${RunId}_best.pt" `
  --run-id $RunId `
  --steps 5000 `
  --batch-size 6 `
  --block-size 512 `
  --n-layer 12 `
  --n-head 8 `
  --num-key-value-heads 4 `
  --n-embd 512 `
  --intermediate-size 1536 `
  --dropout 0.05 `
  --eval-every 250 `
  --save-every 1000 `
  --early-stop-patience 8 `
  --optimizer simple-adamw `
  --lr 3e-4 `
  --min-lr 3e-5 `
  --warmup-steps 200 `
  --cosine-lr `
  --grad-clip 1.0 `
  --amp `
  --qwen3-style `
  --qk-norm `
  --device cuda
