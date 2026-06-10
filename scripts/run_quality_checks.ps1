param(
  [string]$Checkpoint = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  throw "Missing .venv Python at $Python. Recreate with the pinned Python 3.11 setup in README.md."
}
function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
  )
  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
  }
}

Invoke-Checked $Python -c "import sys, torch; print('python=' + sys.executable); print('torch=' + torch.__version__)"

if (-not $Checkpoint) {
  throw "Checkpoint is required for release-quality checks."
}

$checkpoint = $Checkpoint
$CheckpointBase = [IO.Path]::GetFileNameWithoutExtension($checkpoint)

Invoke-Checked $Python scripts\check_checkpoint_model_size.py `
  --checkpoint $checkpoint `
  --strict-config `
  --require-release-prefix old-japanese-0.1B `
  --fail-on-val-oov `
  --require-from-scratch `
  --require-seed `
  --require-optimizer simple-adamw `
  --require-backend cuda
$InputCheck = & $Python scripts\check_checkpoint_training_inputs.py `
  --checkpoint $checkpoint `
  --require-val-data `
  --require-test-data `
  --require-from-scratch `
  --require-run-snapshot `
  --allow-same-run-resume
if ($LASTEXITCODE -ne 0) {
  throw "Checkpoint training input validation failed with exit code ${LASTEXITCODE}."
}
$InputCheck | Write-Output
$TrainingData = (($InputCheck | Select-String -Pattern '^train_data_path=' | Select-Object -First 1).Line -replace '^train_data_path=', '')
if (-not $TrainingData) {
  throw "Could not determine checkpoint train_data_path."
}
$ValData = (($InputCheck | Select-String -Pattern '^val_data_path=' | Select-Object -First 1).Line -replace '^val_data_path=', '')
if (-not $ValData) {
  throw "Could not determine checkpoint val_data_path."
}
$TestData = (($InputCheck | Select-String -Pattern '^test_data_path=' | Select-Object -First 1).Line -replace '^test_data_path=', '')
if (-not $TestData) {
  throw "Could not determine checkpoint test_data_path."
}
$SnapshotManifest = (($InputCheck | Select-String -Pattern '^provenance_file_path=.*corpus_manifest\.jsonl$' | Select-Object -First 1).Line -replace '^provenance_file_path=', '')
if (-not $SnapshotManifest) {
  throw "Could not determine checkpoint snapshot corpus_manifest.jsonl."
}
$AozoraSources = (($InputCheck | Select-String -Pattern '^provenance_file_path=.*aozora_sources\.json$' | Select-Object -First 1).Line -replace '^provenance_file_path=', '')
if (-not $AozoraSources) {
  throw "Could not determine checkpoint snapshot aozora_sources.json."
}
$WakaSources = (($InputCheck | Select-String -Pattern '^provenance_file_path=.*waka_sources\.json$' | Select-Object -First 1).Line -replace '^provenance_file_path=', '')
if (-not $WakaSources) {
  throw "Could not determine checkpoint snapshot waka_sources.json."
}
$TokenizerExtra = (($InputCheck | Select-String -Pattern '^tokenizer_extra_data_path=' | Select-Object -First 1).Line -replace '^tokenizer_extra_data_path=', '')
if (-not $TokenizerExtra) {
  throw "Could not determine checkpoint tokenizer_extra_data_path."
}
$TokenizerMeta = (($InputCheck | Select-String -Pattern '^provenance_file_path=.*tokenizer_public_char_vocab\.meta\.json$' | Select-Object -First 1).Line -replace '^provenance_file_path=', '')
if (-not $TokenizerMeta) {
  throw "Could not determine checkpoint tokenizer_public_char_vocab.meta.json."
}

Invoke-Checked $Python scripts\audit_rule_tables.py
Invoke-Checked $Python scripts\audit_eval_provenance_manifest.py
Invoke-Checked $Python scripts\audit_source_records.py $AozoraSources $WakaSources
Invoke-Checked $Python scripts\audit_public_manifest.py --manifest $SnapshotManifest --out "logs\public_manifest_summary_${CheckpointBase}.json"
Invoke-Checked $Python scripts\eval_grammar_constraints.py --min-cases 28
Invoke-Checked $Python scripts\eval_waka_rules.py --min-cases 20
Invoke-Checked $Python scripts\eval_waka_meter_constraints.py --cases data\eval\waka_meter_constraint_cases.jsonl --min-cases 19
Invoke-Checked $Python scripts\check_tokenizer_vocab_scope.py --manifest $SnapshotManifest --tokenizer-extra-data $TokenizerExtra --tokenizer-meta $TokenizerMeta
Invoke-Checked $Python scripts\check_checkpoint_tokenizer_scope.py --checkpoint $checkpoint --manifest $SnapshotManifest --tokenizer-extra-data $TokenizerExtra --tokenizer-meta $TokenizerMeta
Invoke-Checked $Python scripts\validate_corpus.py `
  $TrainingData `
  $ValData `
  $TestData
Invoke-Checked $Python scripts\validate_corpus.py data\waka\waka_corpus_all.txt --kind waka-poems
Invoke-Checked $Python scripts\check_split_consistency.py --checkpoint $checkpoint
$CleanEvalDir = "data\eval\clean_current"
Invoke-Checked $Python scripts\check_split_leakage.py --manifest $SnapshotManifest --train $TrainingData
Invoke-Checked $Python scripts\check_eval_source_overlap.py `
  --manifest $SnapshotManifest `
  --eval data\eval\grammar_minimal_pairs.jsonl data\eval\grammar_minimal_pairs_heldout.jsonl data\eval\morphology_adversarial_cases.jsonl data\eval\grammar_constraint_cases.jsonl data\eval\waka_rule_cases.jsonl data\eval\waka_meter_constraint_cases.jsonl data\eval\waka_generation_prompts.jsonl
Invoke-Checked $Python scripts\check_eval_contamination.py `
  --train $TrainingData `
  --template-train data\grammar\train_preference_pairs.jsonl `
  --eval data\eval\grammar_minimal_pairs.jsonl data\eval\grammar_minimal_pairs_heldout.jsonl data\eval\morphology_adversarial_cases.jsonl data\eval\grammar_constraint_cases.jsonl data\eval\waka_rule_cases.jsonl data\eval\waka_meter_constraint_cases.jsonl data\eval\waka_generation_prompts.jsonl `
  --strict-prompts `
  --write-clean-dir $CleanEvalDir
Invoke-Checked $Python scripts\check_eval_contamination.py `
  --train $TrainingData `
  --template-train data\grammar\train_preference_pairs.jsonl `
  --eval "$CleanEvalDir\grammar_minimal_pairs.jsonl" "$CleanEvalDir\grammar_minimal_pairs_heldout.jsonl" "$CleanEvalDir\morphology_adversarial_cases.jsonl" "$CleanEvalDir\grammar_constraint_cases.jsonl" "$CleanEvalDir\waka_rule_cases.jsonl" "$CleanEvalDir\waka_meter_constraint_cases.jsonl" "$CleanEvalDir\waka_generation_prompts.jsonl" `
  --strict-prompts
Invoke-Checked $Python scripts\check_eval_source_overlap.py `
  --manifest $SnapshotManifest `
  --eval "$CleanEvalDir\grammar_minimal_pairs.jsonl" "$CleanEvalDir\grammar_minimal_pairs_heldout.jsonl" "$CleanEvalDir\morphology_adversarial_cases.jsonl" "$CleanEvalDir\grammar_constraint_cases.jsonl" "$CleanEvalDir\waka_rule_cases.jsonl" "$CleanEvalDir\waka_meter_constraint_cases.jsonl" "$CleanEvalDir\waka_generation_prompts.jsonl"
$EvalSnapshotDir = "logs\eval_snapshots\$CheckpointBase"
$EvalSnapshotOutput = & $Python scripts\snapshot_eval_files.py `
  --out-dir $EvalSnapshotDir `
  --named "primary=$CleanEvalDir\grammar_minimal_pairs.jsonl" `
  --named "heldout=$CleanEvalDir\grammar_minimal_pairs_heldout.jsonl" `
  --named "morphology=$CleanEvalDir\morphology_adversarial_cases.jsonl" `
  --named "grammar_constraints=$CleanEvalDir\grammar_constraint_cases.jsonl" `
  --named "waka_rules=$CleanEvalDir\waka_rule_cases.jsonl" `
  --named "waka_meter_constraints=$CleanEvalDir\waka_meter_constraint_cases.jsonl" `
  --named "waka_generation_prompts=$CleanEvalDir\waka_generation_prompts.jsonl"
if ($LASTEXITCODE -ne 0) {
  throw "Eval evidence snapshot failed with exit code ${LASTEXITCODE}."
}
$EvalSnapshotOutput | Write-Output
$PrimaryPairs = "$EvalSnapshotDir\primary.jsonl"
$HeldoutPairs = "$EvalSnapshotDir\heldout.jsonl"
$MorphologyCases = "$EvalSnapshotDir\morphology.jsonl"
Invoke-Checked $Python scripts\eval_grammar_constraints.py --cases "$EvalSnapshotDir\grammar_constraints.jsonl" --min-cases 28
Invoke-Checked $Python scripts\eval_waka_rules.py --cases "$EvalSnapshotDir\waka_rules.jsonl" --min-cases 20
Invoke-Checked $Python scripts\eval_waka_meter_constraints.py --cases "$EvalSnapshotDir\waka_meter_constraints.jsonl" --min-cases 19
Invoke-Checked $Python scripts\eval_heldout_lm.py --checkpoint $checkpoint --device cuda --data $TestData --split-name test --max-loss 8.0
Invoke-Checked $Python scripts\eval_minimal_pairs.py --checkpoint $checkpoint --device cuda --pairs $PrimaryPairs --metric-prefix primary --min-cases 8 --min-accuracy 1.0
Invoke-Checked $Python scripts\eval_minimal_pairs.py --checkpoint $checkpoint --device cuda --pairs $HeldoutPairs --metric-prefix heldout --min-cases 12 --min-accuracy 1.0
Invoke-Checked $Python scripts\eval_morphology_adversarial.py --cases $MorphologyCases --min-cases 4
Invoke-Checked $Python scripts\eval_waka_meter_generation.py `
  --checkpoint $checkpoint `
  --device cuda `
  --decoding greedy `
  --seed 20260509 `
  --prompts-file "$EvalSnapshotDir\waka_generation_prompts.jsonl" `
  --min-cases 4
