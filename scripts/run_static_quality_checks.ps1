param(
  [switch]$RefreshEvidence
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  throw "Missing .venv Python at $Python. Recreate the environment first."
}
$StaticLog = if ($RefreshEvidence.IsPresent) { "logs\static_quality_$(Get-Date -Format yyyyMMdd_HHmmss).log" } else { "" }
if ($RefreshEvidence.IsPresent) {
  New-Item -ItemType Directory -Force logs | Out-Null
}

function Write-FailedStaticQualityManifest {
  param([int]$ExitCode)
  if ($RefreshEvidence.IsPresent) {
    & $Python scripts\write_static_quality_manifest.py --log $StaticLog --status failed --exit-code $ExitCode --command "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_static_quality_checks.ps1 -RefreshEvidence"
  }
}

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
  )
  $CommandText = "$FilePath $($Arguments -join ' ')"
  if ($RefreshEvidence.IsPresent) {
    Add-Content -Encoding UTF8 -Path $StaticLog -Value ("COMMAND " + (Get-Date -Format o) + " " + $CommandText)
  }
  & $FilePath @Arguments
  $ExitCode = $LASTEXITCODE
  if ($RefreshEvidence.IsPresent) {
    Add-Content -Encoding UTF8 -Path $StaticLog -Value ("EXIT " + (Get-Date -Format o) + " " + $ExitCode + " " + $CommandText)
  }
  if ($ExitCode -ne 0) {
    Write-FailedStaticQualityManifest -ExitCode $ExitCode
    throw "Command failed with exit code ${ExitCode}: $CommandText"
  }
}

function Invoke-RefreshChecked {
  param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
  )
  if (-not $RefreshEvidence.IsPresent) {
    return
  }
  Invoke-Checked $FilePath @Arguments
}

function Invoke-PostPreflightChecked {
  param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
  )
  $CommandText = "$FilePath $($Arguments -join ' ')"
  & $FilePath @Arguments
  $ExitCode = $LASTEXITCODE
  if ($RefreshEvidence.IsPresent) {
    Add-Content -Encoding UTF8 -Path $StaticLog -Value ("COMMAND " + (Get-Date -Format o) + " " + $CommandText)
    Add-Content -Encoding UTF8 -Path $StaticLog -Value ("EXIT " + (Get-Date -Format o) + " " + $ExitCode + " " + $CommandText)
  }
  if ($ExitCode -ne 0) {
    Write-FailedStaticQualityManifest -ExitCode $ExitCode
    throw "Command failed with exit code ${ExitCode}: $CommandText"
  }
}

Invoke-Checked $Python -c "import sys, torch; print('python=' + sys.executable); print('torch=' + torch.__version__); print('cuda_available=' + str(torch.cuda.is_available()))"
Invoke-Checked $Python scripts\test_optimizer_state_validation.py
Invoke-Checked $Python scripts\test_release_resume_chain_validation.py
Invoke-Checked $Python scripts\test_run_completion_json_encoding.py
Invoke-Checked $Python scripts\test_run_command_capture_streaming.py
Invoke-Checked $Python scripts\test_dml_supervisor_static_contracts.py
Invoke-Checked $Python scripts\test_dml_invalid_lock_supervision_contract.py
Invoke-Checked $Python scripts\test_static_quality_checkonly_contract.py
Invoke-Checked $Python scripts\test_dml_active_lock_atomic_contract.py
Invoke-Checked $Python scripts\test_split_policy_contract.py
Invoke-Checked $Python scripts\test_split_policy_unknown_work.py
Invoke-Checked $Python scripts\test_active_dml_process_scan.py
Invoke-Checked $Python scripts\test_speed_probe_concurrency_guard.py
Invoke-Checked $Python scripts\test_quality_threshold_contracts.py
Invoke-Checked $Python scripts\test_zero_base_review_no_prescribed_wording.py
Invoke-Checked $Python scripts\test_llm_review_packet_sanitization.py
Invoke-Checked $Python scripts\test_source_release_clean_contract.py
Invoke-Checked $Python scripts\check_source_release_clean.py
Invoke-Checked $Python scripts\test_eval_json_quality_log_trust.py
Invoke-Checked $Python scripts\test_eval_snapshot_provenance.py
Invoke-Checked $Python scripts\test_eval_source_overlap.py
Invoke-Checked $Python scripts\test_waka_variant_dedup.py
Invoke-Checked $Python scripts\test_colab_cuda_contracts.py
Invoke-Checked $Python scripts\test_colab_cuda_failure_contracts.py
Invoke-Checked powershell -NoProfile -ExecutionPolicy Bypass -File scripts\test_powershell_parser.ps1
Invoke-Checked powershell -NoProfile -ExecutionPolicy Bypass -File scripts\test_dml_direct_launch_gate.ps1
Invoke-Checked powershell -NoProfile -ExecutionPolicy Bypass -File scripts\test_release_training_entrypoint_gates.ps1
Invoke-Checked powershell -NoProfile -ExecutionPolicy Bypass -File scripts\test_active_dml_run_id_parser.ps1
Invoke-Checked $Python scripts\test_active_lock_launcher_liveness.py
Invoke-Checked $Python scripts\test_startup_mutex_health.py
Invoke-Checked $Python scripts\test_cuda_active_lock_board_governance.py
Invoke-Checked $Python scripts\test_colab_cuda_lease_health.py
Invoke-Checked $Python scripts\test_run_completion_active_scope.py
Invoke-Checked $Python scripts\test_active_lock_corruption_policy.py
Invoke-Checked $Python scripts\test_active_run_release_policy.py
Invoke-Checked $Python scripts\test_run_id_unused_guard.py
Invoke-Checked $Python scripts\test_non_release_registry.py
Invoke-Checked $Python scripts\check_non_release_registry.py
Invoke-Checked $Python scripts\test_progress_reporter_pid_discovery.py
Invoke-Checked $Python scripts\test_snapshot_manifest_release_boundary.py
Invoke-Checked $Python scripts\test_release_policy_guards.py
Invoke-Checked $Python scripts\audit_rule_tables.py
Invoke-Checked $Python scripts\audit_rule_ssot.py
Invoke-Checked $Python scripts\audit_eval_provenance_manifest.py
Invoke-Checked $Python scripts\check_generation_diagnostic_policy.py
if ($RefreshEvidence.IsPresent) {
  Invoke-Checked $Python scripts\audit_public_manifest.py
} else {
  Invoke-Checked $Python scripts\audit_public_manifest.py --check-only
}
Invoke-Checked $Python scripts\audit_source_records.py data\aozora\sources.json data\waka\sources.json
Invoke-Checked $Python scripts\audit_metadata_encoding.py
if ($RefreshEvidence.IsPresent) {
  Invoke-Checked $Python scripts\score_source_quality.py
  Invoke-Checked $Python scripts\build_waka_meter_corpus.py
  Invoke-Checked $Python scripts\build_training_corpus.py
  Invoke-Checked $Python scripts\build_preference_boost_corpus.py
  Invoke-Checked $Python scripts\build_worldclass_corpus.py
  Invoke-Checked $Python scripts\build_training_augmentation_manifest.py
} else {
  Invoke-Checked $Python scripts\score_source_quality.py --check-only
  Invoke-Checked $Python scripts\build_training_augmentation_manifest.py --audit-only
}
Invoke-Checked $Python scripts\check_generated_training_inputs_lf.py
Invoke-Checked $Python scripts\audit_training_augmentation_manifest.py
Invoke-Checked $Python scripts\test_training_augmentation_manifest_audit.py
Invoke-Checked $Python scripts\test_byte_fallback_tokenizer.py
Invoke-Checked $Python scripts\test_checkpoint_tokenizer_scope.py
Invoke-Checked $Python -m py_compile src\kobun_llm\release_resume.py scripts\run_command_capture.py scripts\test_run_command_capture_streaming.py scripts\test_release_resume_chain_validation.py scripts\test_dml_supervisor_static_contracts.py scripts\test_dml_invalid_lock_supervision_contract.py scripts\test_static_quality_checkonly_contract.py scripts\test_dml_active_lock_atomic_contract.py scripts\test_colab_cuda_contracts.py scripts\test_colab_cuda_failure_contracts.py scripts\test_cuda_active_lock_board_governance.py scripts\test_colab_cuda_lease_health.py scripts\test_run_completion_active_scope.py scripts\check_run_completion.py scripts\check_source_release_clean.py scripts\test_source_release_clean_contract.py scripts\check_generated_training_inputs_lf.py scripts\check_colab_cuda_environment.py scripts\start_old_japanese_0_1b_cuda_colab_and_watch.py scripts\create_colab_sync_bundle.py scripts\run_quality_checks_cuda.py scripts\audit_eval_provenance_manifest.py scripts\check_eval_clean_current.py scripts\check_eval_source_overlap.py scripts\test_eval_source_overlap.py scripts\waka_variant_dedup.py scripts\test_waka_variant_dedup.py scripts\test_split_policy_contract.py scripts\test_parse_quality_log_tokenizer_scope.py scripts\check_checkpoint_tokenizer_scope.py scripts\test_checkpoint_tokenizer_scope.py scripts\probe_dml_training_speed.py scripts\test_speed_probe_concurrency_guard.py scripts\test_quality_threshold_contracts.py scripts\test_zero_base_review_no_prescribed_wording.py scripts\test_llm_review_packet_sanitization.py scripts\test_eval_json_quality_log_trust.py scripts\test_eval_snapshot_provenance.py scripts\write_preflight_gate.py scripts\verify_preflight_gate.py scripts\write_static_quality_manifest.py scripts\verify_static_quality_manifest.py scripts\write_zero_base_review_artifact.py scripts\write_zero_base_review_gate.py scripts\verify_zero_base_review_gate.py scripts\test_zero_base_review_gate.py scripts\assert_run_id_unused.py scripts\test_run_id_unused_guard.py scripts\test_startup_mutex_health.py
Invoke-Checked $Python scripts\test_parse_quality_log_tokenizer_scope.py
Invoke-RefreshChecked $Python scripts\build_tokenizer_public_vocab.py
Invoke-Checked $Python scripts\check_tokenizer_vocab_scope.py `
  --manifest data\corpus_manifest.jsonl `
  --tokenizer-extra-data data\tokenizer_public_char_vocab.txt `
  --tokenizer-meta data\tokenizer_public_char_vocab.meta.json
Invoke-Checked $Python scripts\eval_grammar_constraints.py --min-cases 28
Invoke-Checked $Python scripts\eval_waka_rules.py --min-cases 20
Invoke-Checked $Python scripts\eval_waka_meter_constraints.py --cases data\eval\waka_meter_constraint_cases.jsonl --min-cases 19
Invoke-Checked $Python scripts\eval_morphology_adversarial.py --min-cases 4
Invoke-Checked $Python scripts\validate_corpus.py `
  data\kobun_grammar_corpus.txt `
  data\kobun_labeled_grammar_corpus.txt `
  data\kobun_labeled_grammar_train.txt `
  data\kobun_labeled_grammar_val.txt `
  data\kobun_labeled_grammar_test.txt `
  data\kobun_labeled_grammar_boost_train.txt `
  data\kobun_worldclass_corpus.txt
Invoke-Checked $Python scripts\validate_corpus.py data\waka\waka_corpus_all.txt --kind waka-poems
Invoke-Checked $Python scripts\check_split_consistency.py `
  --manifest data\corpus_manifest.jsonl `
  --train data\kobun_worldclass_corpus.txt `
  --val data\kobun_labeled_grammar_val.txt `
  --test data\kobun_labeled_grammar_test.txt
Invoke-Checked $Python scripts\check_split_leakage.py --train data\kobun_worldclass_corpus.txt
Invoke-Checked $Python scripts\check_eval_source_overlap.py `
  --manifest data\corpus_manifest.jsonl `
  --eval data\eval\grammar_minimal_pairs.jsonl data\eval\grammar_minimal_pairs_heldout.jsonl data\eval\morphology_adversarial_cases.jsonl data\eval\grammar_constraint_cases.jsonl data\eval\waka_rule_cases.jsonl data\eval\waka_meter_constraint_cases.jsonl data\eval\waka_generation_prompts.jsonl
Invoke-Checked $Python scripts\check_eval_contamination.py `
  --train data\kobun_worldclass_corpus.txt data\kobun_labeled_grammar_train.txt data\kobun_labeled_grammar_boost_train.txt `
  --template-train data\grammar\train_preference_pairs.jsonl `
  --eval data\eval\grammar_minimal_pairs.jsonl data\eval\grammar_minimal_pairs_heldout.jsonl data\eval\morphology_adversarial_cases.jsonl data\eval\grammar_constraint_cases.jsonl data\eval\waka_rule_cases.jsonl data\eval\waka_meter_constraint_cases.jsonl data\eval\waka_generation_prompts.jsonl `
  --strict-prompts
Invoke-RefreshChecked $Python scripts\check_eval_contamination.py `
  --train data\kobun_worldclass_corpus.txt data\kobun_labeled_grammar_train.txt data\kobun_labeled_grammar_boost_train.txt `
  --template-train data\grammar\train_preference_pairs.jsonl `
  --eval data\eval\grammar_minimal_pairs.jsonl data\eval\grammar_minimal_pairs_heldout.jsonl data\eval\morphology_adversarial_cases.jsonl data\eval\grammar_constraint_cases.jsonl data\eval\waka_rule_cases.jsonl data\eval\waka_meter_constraint_cases.jsonl data\eval\waka_generation_prompts.jsonl `
  --strict-prompts `
  --write-clean-dir data\eval\clean_current
Invoke-Checked $Python scripts\check_eval_clean_current.py
Invoke-Checked $Python scripts\check_eval_source_overlap.py `
  --manifest data\corpus_manifest.jsonl `
  --eval data\eval\clean_current\grammar_minimal_pairs.jsonl data\eval\clean_current\grammar_minimal_pairs_heldout.jsonl data\eval\clean_current\morphology_adversarial_cases.jsonl data\eval\clean_current\grammar_constraint_cases.jsonl data\eval\clean_current\waka_rule_cases.jsonl data\eval\clean_current\waka_meter_constraint_cases.jsonl data\eval\clean_current\waka_generation_prompts.jsonl
Invoke-Checked $Python scripts\test_release_sanitization.py
Invoke-Checked $Python scripts\test_release_package_path_scan.py
Invoke-Checked $Python scripts\check_release_workspace_clean.py
Invoke-RefreshChecked $Python scripts\update_evaluation_board.py
Invoke-RefreshChecked $Python scripts\build_llm_review_packet.py
if ($RefreshEvidence.IsPresent) {
  Invoke-Checked $Python scripts\check_llm_review_packet_fresh.py
}
Invoke-Checked $Python scripts\check_model_config.py `
  --data data\kobun_worldclass_corpus.txt `
  --tokenizer-extra-data data\tokenizer_public_char_vocab.txt `
  --tokenizer-type byte_fallback_char_v1 `
  --block-size 384 `
  --min-params 100000000 `
  --max-params 180000000

if (-not $RefreshEvidence.IsPresent) {
  Write-Output "static_quality_check_only_ok=true"
  exit 0
}

& $Python scripts\write_static_quality_manifest.py --log $StaticLog --status passed --exit-code 0 --command "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_static_quality_checks.ps1 -RefreshEvidence"
if ($LASTEXITCODE -ne 0) {
  throw "Could not write static quality manifest."
}
& $Python scripts\verify_static_quality_manifest.py --max-age-minutes 120
if ($LASTEXITCODE -ne 0) {
  throw "Static quality manifest verification failed."
}
& $Python scripts\write_preflight_gate.py
if ($LASTEXITCODE -ne 0) {
  throw "Could not write preflight gate."
}
& $Python scripts\verify_preflight_gate.py --max-age-minutes 120
if ($LASTEXITCODE -ne 0) {
  throw "Preflight gate verification failed."
}
Invoke-PostPreflightChecked powershell -NoProfile -ExecutionPolicy Bypass -File scripts\test_dml_review_gate_launch_guard.ps1
Invoke-PostPreflightChecked $Python scripts\test_zero_base_review_gate.py
& $Python scripts\write_static_quality_manifest.py --log $StaticLog --status passed --exit-code 0 --command "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_static_quality_checks.ps1 -RefreshEvidence"
if ($LASTEXITCODE -ne 0) {
  throw "Could not refresh static quality manifest before autonomous context launch guard."
}
& $Python scripts\verify_static_quality_manifest.py --max-age-minutes 120
if ($LASTEXITCODE -ne 0) {
  throw "Refreshed static quality manifest verification failed before autonomous context launch guard."
}
& $Python scripts\write_preflight_gate.py
if ($LASTEXITCODE -ne 0) {
  throw "Could not refresh preflight gate before autonomous context launch guard."
}
& $Python scripts\verify_preflight_gate.py --max-age-minutes 120
if ($LASTEXITCODE -ne 0) {
  throw "Refreshed preflight gate verification failed before autonomous context launch guard."
}
Invoke-PostPreflightChecked powershell -NoProfile -ExecutionPolicy Bypass -File scripts\test_dml_autonomous_context_launch_guard.ps1
& $Python scripts\write_static_quality_manifest.py --log $StaticLog --status passed --exit-code 0 --command "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_static_quality_checks.ps1 -RefreshEvidence"
if ($LASTEXITCODE -ne 0) {
  throw "Could not write final static quality manifest."
}
& $Python scripts\verify_static_quality_manifest.py --max-age-minutes 120
if ($LASTEXITCODE -ne 0) {
  throw "Final static quality manifest verification failed."
}
& $Python scripts\write_preflight_gate.py
if ($LASTEXITCODE -ne 0) {
  throw "Could not write final preflight gate."
}
& $Python scripts\verify_preflight_gate.py --max-age-minutes 120
if ($LASTEXITCODE -ne 0) {
  throw "Final preflight gate verification failed."
}
