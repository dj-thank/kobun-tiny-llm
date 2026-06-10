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
.\.venv\Scripts\python.exe scripts\build_preference_boost_corpus.py `
  --base data/kobun_labeled_grammar_train.txt `
  --pairs data/grammar/train_preference_pairs.jsonl `
  --out data/kobun_labeled_grammar_boost_train.txt `
  --repeat 80

.\.venv\Scripts\python.exe -u -m kobun_llm.train `
  --data data/kobun_labeled_grammar_boost_train.txt `
  --val-data data/kobun_labeled_grammar_val.txt `
  --out "checkpoints/kobun_qwen3_12l_gpu_grammarboost_$RunId.pt" `
  --best-out "checkpoints/kobun_qwen3_12l_gpu_grammarboost_${RunId}_best.pt" `
  --run-id $RunId `
  --steps 1200 `
  --batch-size 6 `
  --block-size 512 `
  --n-layer 12 `
  --n-head 8 `
  --num-key-value-heads 4 `
  --n-embd 512 `
  --intermediate-size 1536 `
  --dropout 0.05 `
  --lr 1.5e-4 `
  --min-lr 1.0e-5 `
  --warmup-steps 60 `
  --cosine-lr `
  --grad-clip 1.0 `
  --eval-every 100 `
  --save-every 400 `
  --early-stop-patience 8 `
  --optimizer simple-adamw `
  --amp `
  --qwen3-style `
  --qk-norm `
  --device cuda
