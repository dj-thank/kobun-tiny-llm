$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$checkpoint = .\.venv\Scripts\python.exe scripts\latest_valid_checkpoint.py `
  --pattern "checkpoints/kobun_qwen3_12l_worldclass_*_best.pt" `
  --fallback "checkpoints/kobun_qwen3_12l_gpu_grammarboost_best.pt" |
  Select-Object -Last 1

.\.venv\Scripts\python.exe -m kobun_llm.repl `
  --checkpoint $checkpoint `
  --max-new-tokens 260 `
  --temperature 0.7 `
  --top-k 20 `
  --top-p 0.8 `
  --presence-penalty 0.2 `
  --soft-grammar-constraints `
  --candidates 3 `
  --grammar-rerank `
  --style genji `
  --device cuda
