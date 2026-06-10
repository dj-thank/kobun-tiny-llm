param(
  [string]$RunId = "",
  [switch]$LaunchedBySupervisor
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv-dml\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  throw "Missing DirectML Python at $Python. Create it with Python 3.12 and torch-directml."
}
$CapturePython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $CapturePython)) {
  $CapturePython = $Python
}
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$PreflightGate = $env:OLD_JAPANESE_PREFLIGHT_GATE
if (-not $PreflightGate) {
  $PreflightGate = "logs\preflight_gate_old_japanese_0_1b.json"
}
if (-not $LaunchedBySupervisor.IsPresent) {
  throw "Release-candidate training must be started through scripts\start_old_japanese_0_1b_dml_and_watch.ps1 so the active-run lock and watcher are attached."
}
$ReviewGate = $env:OLD_JAPANESE_REVIEW_GATE
$AutonomousLaunchContext = $env:OLD_JAPANESE_AUTONOMOUS_LAUNCH_CONTEXT

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

if (-not $RunId) {
  $RunId = "old_japanese_0_1b_dml_$(Get-Date -Format "yyyyMMdd_HHmmss")"
} elseif ($RunId -match '^old_japanese_0_1b_' -and $RunId -notmatch '^old_japanese_0_1b_dml_') {
  throw "Invalid DML RunId: non-DML old_japanese_0_1b_* ids are not accepted by the DML launcher: $RunId"
} elseif ($RunId -notmatch '^old_japanese_0_1b_dml_') {
  if ($RunId -match '^dml_') {
    $RunId = $RunId -replace '^dml_', ''
  }
  $RunId = "old_japanese_0_1b_dml_$RunId"
}
if ($RunId -notmatch '^old_japanese_0_1b_dml_[0-9A-Za-z][0-9A-Za-z_-]{0,63}$') {
  throw "Invalid RunId: $RunId"
}
& (Join-Path $Root ".venv\Scripts\python.exe") -c "from kobun_autonomy.release_policy import require_release_candidate_run; import sys; require_release_candidate_run(sys.argv[1], context='DML training wrapper')" $RunId
if ($LASTEXITCODE -ne 0) {
  throw "DML training wrapper refuses known non-release RunId: $RunId"
}

function Get-Sha256Text {
  param([Parameter(Mandatory = $true)][string]$Text)
  $Bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($Text)
  $Hash = [System.Security.Cryptography.SHA256]::Create().ComputeHash($Bytes)
  return -join ($Hash | ForEach-Object { $_.ToString("x2") })
}

function Get-Sha256File {
  param([Parameter(Mandatory = $true)][string]$PathText)
  return (Get-FileHash -Algorithm SHA256 -LiteralPath $PathText).Hash.ToLowerInvariant()
}

function Normalize-Path {
  param([Parameter(Mandatory = $true)][string]$PathText)
  if ([IO.Path]::IsPathRooted($PathText)) {
    return [IO.Path]::GetFullPath($PathText).TrimEnd('\').ToLowerInvariant()
  }
  return [IO.Path]::GetFullPath((Join-Path $Root $PathText)).TrimEnd('\').ToLowerInvariant()
}

function Assert-BoundEvidenceFile {
  param(
    [Parameter(Mandatory = $true)][object]$Lock,
    [Parameter(Mandatory = $true)][string]$Label,
    [Parameter(Mandatory = $true)][string]$EnvPath,
    [Parameter(Mandatory = $true)][string]$PathField,
    [Parameter(Mandatory = $true)][string]$HashField
  )
  $LockPathValue = [string]$Lock.$PathField
  $LockHashValue = [string]$Lock.$HashField
  if (-not $EnvPath) {
    throw "Missing ${Label} environment path."
  }
  if (-not $LockPathValue) {
    throw "Active-run lock missing ${PathField}."
  }
  if ((Normalize-Path $EnvPath) -ne (Normalize-Path $LockPathValue)) {
    throw "Active-run lock ${Label} path does not match supervisor context."
  }
  $Full = if ([IO.Path]::IsPathRooted($EnvPath)) {
    [IO.Path]::GetFullPath($EnvPath)
  } else {
    [IO.Path]::GetFullPath((Join-Path $Root $EnvPath))
  }
  if (-not (Test-Path $Full)) {
    throw "Active-run lock bound ${Label} file is missing: $EnvPath"
  }
  if (-not $LockHashValue -or $LockHashValue -ne (Get-Sha256File $Full)) {
    throw "Active-run lock ${Label} hash mismatch."
  }
}

function Get-ParentProcessIds {
  param([int]$PidValue = $PID)
  $Ids = @()
  $CurrentPid = $PidValue
  while ($CurrentPid -gt 0) {
    $Proc = Get-CimInstance Win32_Process -Filter "ProcessId=$CurrentPid" -ErrorAction SilentlyContinue
    if (-not $Proc) {
      break
    }
    $ParentPid = [int]$Proc.ParentProcessId
    if ($ParentPid -le 0 -or $Ids -contains $ParentPid) {
      break
    }
    $Ids += $ParentPid
    $CurrentPid = $ParentPid
  }
  return $Ids
}

function Get-ProcessCommandLine {
  param([Parameter(Mandatory = $true)][int]$PidValue)
  $Proc = Get-CimInstance Win32_Process -Filter "ProcessId=$PidValue" -ErrorAction SilentlyContinue
  if (-not $Proc) {
    return ""
  }
  return [string]$Proc.CommandLine
}

function Assert-SupervisorContext {
  if ($env:OLD_JAPANESE_SUPERVISOR_RUN_ID -ne $RunId) {
    throw "Missing or mismatched supervisor RunId context."
  }
  if (-not $env:OLD_JAPANESE_SUPERVISOR_TOKEN) {
    throw "Missing supervisor launch token."
  }
  $LockPath = $env:OLD_JAPANESE_ACTIVE_LOCK
  if (-not $LockPath) {
    $LockPath = Join-Path $Root "logs\active_old_japanese_0_1b_dml.lock"
  }
  if (-not (Test-Path $LockPath)) {
    throw "Missing active-run lock for supervisor launch."
  }
  $Lock = Get-Content -Raw -Encoding UTF8 $LockPath | ConvertFrom-Json
  if ($Lock.run_id -ne $RunId) {
    throw "Active-run lock RunId mismatch: lock=$($Lock.run_id) requested=$RunId"
  }
  $Deadline = (Get-Date).AddSeconds(5)
  while (([int]$Lock.train_pid -eq 0) -and (Get-Date) -lt $Deadline) {
    Start-Sleep -Milliseconds 100
    $Lock = Get-Content -Raw -Encoding UTF8 $LockPath | ConvertFrom-Json
    if ($Lock.run_id -ne $RunId) {
      throw "Active-run lock RunId mismatch while waiting for train_pid: lock=$($Lock.run_id) requested=$RunId"
    }
  }
  if ([int]$Lock.train_pid -ne $PID) {
    throw "Active-run lock train_pid does not match current wrapper PID: lock=$($Lock.train_pid) wrapper=$PID"
  }
  if ([string]$Lock.state -notin @("train_started_watcher_pending", "running")) {
    throw "Active-run lock state is not valid for train wrapper start: state=$($Lock.state)"
  }
  if ($Lock.launcher_pid -le 0) {
    throw "Active-run lock missing launcher_pid."
  }
  $LauncherPid = [int]$Lock.launcher_pid
  $Ancestors = @(Get-ParentProcessIds -PidValue $PID)
  if ($Ancestors -notcontains $LauncherPid) {
    throw "Active-run lock launcher_pid is not an ancestor of the train wrapper."
  }
  $LauncherCommandLine = Get-ProcessCommandLine -PidValue $LauncherPid
  if (
    $LauncherCommandLine -notlike "*start_old_japanese_0_1b_dml_and_watch.ps1*" -or
    $LauncherCommandLine -notlike "*$RunId*" -or
    $LauncherCommandLine -notlike "*-AllowStartTraining*" -or
    $LauncherCommandLine -notlike "*-ReviewsPassed*"
  ) {
    throw "Active-run lock launcher_pid is not the authorized DML supervisor for this RunId."
  }
  if ((Get-Sha256Text $env:OLD_JAPANESE_SUPERVISOR_TOKEN) -ne $Lock.launch_token_sha256) {
    throw "Active-run lock token hash mismatch."
  }
  if (-not $env:OLD_JAPANESE_AUTONOMOUS_LAUNCH_NONCE) {
    throw "Missing autonomous launch nonce."
  }
  if ((Get-Sha256Text $env:OLD_JAPANESE_AUTONOMOUS_LAUNCH_NONCE) -ne $Lock.launch_nonce_sha256) {
    throw "Active-run lock autonomous launch nonce hash mismatch."
  }
  if ($Lock.hf_export -ne $false) {
    throw "Active-run lock must attest hf_export=false."
  }
  if ([string]$Lock.selected_action -ne "prepare_next_fresh_run_after_static_gate_and_zero_base_reviews") {
    throw "Active-run lock selected_action is not authorized for training."
  }
  if ([string]$Lock.autonomous_script -ne "scripts\autonomous_old_japanese_0_1b_loop.ps1") {
    throw "Active-run lock autonomous_script mismatch."
  }
  if ([int]$Lock.autonomous_pid -le 0) {
    throw "Active-run lock missing autonomous_pid."
  }
  Assert-BoundEvidenceFile -Lock $Lock -Label "preflight gate" -EnvPath $PreflightGate -PathField "preflight_gate" -HashField "preflight_gate_sha256"
  Assert-BoundEvidenceFile -Lock $Lock -Label "zero-base review gate" -EnvPath $ReviewGate -PathField "review_gate" -HashField "review_gate_sha256"
  Assert-BoundEvidenceFile -Lock $Lock -Label "autonomous launch context" -EnvPath $AutonomousLaunchContext -PathField "autonomous_launch_context" -HashField "autonomous_launch_context_sha256"
}
Assert-SupervisorContext

Invoke-Checked $CapturePython scripts\verify_preflight_gate.py --gate $PreflightGate --max-age-minutes 120
Invoke-Checked $CapturePython scripts\verify_zero_base_review_gate.py --gate $ReviewGate --preflight-gate $PreflightGate --max-age-minutes 120

$Out = "checkpoints/${RunId}.pt"
$BestOut = "checkpoints/${RunId}_best.pt"
New-Item -ItemType Directory -Force logs | Out-Null
$ExitSentinel = "logs\train_exit_${RunId}.json"
$StdoutLog = "logs\${RunId}.out.log"
$StderrLog = "logs\${RunId}.err.log"
$SnapshotDir = "data\run_snapshots\$RunId"

function Assert-UnderDirectory {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Label,
    [Parameter(Mandatory = $true)]
    [string]$PathText,
    [Parameter(Mandatory = $true)]
    [string]$DirectoryText
  )
  $Base = [IO.Path]::GetFullPath((Join-Path $Root $DirectoryText)).TrimEnd('\') + '\'
  $Full = [IO.Path]::GetFullPath((Join-Path $Root $PathText))
  if (-not $Full.StartsWith($Base, [StringComparison]::OrdinalIgnoreCase)) {
    throw "${Label} path escapes ${DirectoryText}: $PathText"
  }
}

Assert-UnderDirectory -Label "checkpoint" -PathText $Out -DirectoryText "checkpoints"
Assert-UnderDirectory -Label "best checkpoint" -PathText $BestOut -DirectoryText "checkpoints"
Assert-UnderDirectory -Label "exit sentinel" -PathText $ExitSentinel -DirectoryText "logs"
Assert-UnderDirectory -Label "stdout log" -PathText $StdoutLog -DirectoryText "logs"
Assert-UnderDirectory -Label "stderr log" -PathText $StderrLog -DirectoryText "logs"
Assert-UnderDirectory -Label "snapshot" -PathText $SnapshotDir -DirectoryText "data\run_snapshots"

Invoke-Checked $CapturePython scripts\assert_run_id_unused.py --run-id $RunId --allow-supervisor-launch-artifacts
if ($LASTEXITCODE -ne 0) {
  throw "Refusing to reuse RunId ${RunId}; run-scoped artifact already exists."
}

function Write-TrainExitSentinel {
  param(
    [Parameter(Mandatory = $true)]
    [int]$ExitCode,
    [string]$Message = ""
  )
  $Payload = [pscustomobject]@{
    run_id = $RunId
    exit_code = $ExitCode
    message = $Message
    completed_at = Get-Date -Format o
    checkpoint = $Out
    best_checkpoint = $BestOut
    hf_export = $false
  } | ConvertTo-Json -Depth 4
  [System.IO.File]::WriteAllText(
    [IO.Path]::GetFullPath((Join-Path $Root $ExitSentinel)),
    $Payload + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
  )
}

$TrainExit = 1
$TrainMessage = ""
$Phase = "preflight"
try {
Invoke-Checked $Python -c "import torch, torch_directml; name = torch_directml.device_name(0).replace(chr(0), '').strip(); print('torch=' + torch.__version__); print('directml_device=' + name)"

Invoke-Checked $Python scripts\build_waka_training_corpus.py
Invoke-Checked $Python scripts\audit_source_records.py data\aozora\sources.json data\waka\sources.json
Invoke-Checked $Python scripts\build_manifest.py
Invoke-Checked $Python scripts\audit_public_manifest.py
Invoke-Checked $Python scripts\audit_source_records.py data\aozora\sources.json data\waka\sources.json
Invoke-Checked $Python scripts\build_waka_meter_corpus.py
Invoke-Checked $Python scripts\build_tokenizer_public_vocab.py
Invoke-Checked $Python scripts\check_tokenizer_vocab_scope.py `
  --manifest data\corpus_manifest.jsonl `
  --tokenizer-extra-data data\tokenizer_public_char_vocab.txt `
  --tokenizer-meta data\tokenizer_public_char_vocab.meta.json
Invoke-Checked $Python scripts\build_training_corpus.py
Invoke-Checked $Python scripts\build_preference_boost_corpus.py
Invoke-Checked $Python scripts\build_external_knowledge_surface_patterns.py
Invoke-Checked $Python scripts\build_worldclass_corpus.py
Invoke-Checked $Python scripts\build_training_augmentation_manifest.py
Invoke-Checked $CapturePython scripts\verify_preflight_gate.py --gate $PreflightGate --max-age-minutes 120
Invoke-Checked $CapturePython scripts\verify_zero_base_review_gate.py --gate $ReviewGate --preflight-gate $PreflightGate --max-age-minutes 120

$SnapshotOutput = & $Python scripts\snapshot_training_inputs.py `
  --run-id $RunId `
  --data data/kobun_worldclass_corpus.txt `
  --val-data data/kobun_labeled_grammar_val.txt `
  --test-data data/kobun_labeled_grammar_test.txt `
  --tokenizer-extra-data data/tokenizer_public_char_vocab.txt `
  --provenance-file data/corpus_manifest.jsonl `
  --provenance-file logs/public_manifest_summary.json `
  --provenance-file data/aozora/sources.json `
  --provenance-file data/waka/sources.json `
  --provenance-file data/tokenizer_public_char_vocab.meta.json `
  --provenance-file data/training_augmentation_manifest.json
if ($LASTEXITCODE -ne 0) {
  throw "Training input snapshot failed with exit code ${LASTEXITCODE}."
}
$SnapshotOutput | Write-Output
$SnapshotTrain = (($SnapshotOutput | Select-String -Pattern '^snapshot_train_data=' | Select-Object -First 1).Line -replace '^snapshot_train_data=', '')
$SnapshotVal = (($SnapshotOutput | Select-String -Pattern '^snapshot_val_data=' | Select-Object -First 1).Line -replace '^snapshot_val_data=', '')
$SnapshotTest = (($SnapshotOutput | Select-String -Pattern '^snapshot_test_data=' | Select-Object -First 1).Line -replace '^snapshot_test_data=', '')
$SnapshotTokenizerExtras = @($SnapshotOutput | Select-String -Pattern '^snapshot_tokenizer_extra_data=' | ForEach-Object { $_.Line -replace '^snapshot_tokenizer_extra_data=', '' })
$SnapshotProvenance = @($SnapshotOutput | Select-String -Pattern '^snapshot_provenance_file=' | ForEach-Object { $_.Line -replace '^snapshot_provenance_file=', '' })
if (-not $SnapshotTrain -or -not $SnapshotVal -or -not $SnapshotTest -or $SnapshotTokenizerExtras.Count -lt 1) {
  throw "Snapshot output did not include required train/val/test/tokenizer paths."
}
$SnapshotTokenizerExtra = $SnapshotTokenizerExtras[0]

Invoke-Checked $Python scripts\check_model_config.py `
  --data $SnapshotTrain `
  --tokenizer-extra-data $SnapshotTokenizerExtra `
  --tokenizer-type byte_fallback_char_v1 `
  --block-size 384 `
  --n-layer 16 `
  --n-head 12 `
  --num-key-value-heads 6 `
  --n-embd 768 `
  --intermediate-size 2304 `
  --min-params 100000000 `
  --max-params 180000000

$TrainArgs = @(
  "-u",
  "-m",
  "kobun_llm.train",
  "--data",
  $SnapshotTrain,
  "--val-data",
  $SnapshotVal,
  "--test-data",
  $SnapshotTest
)
foreach ($Path in $SnapshotTokenizerExtras) {
  $TrainArgs += @("--tokenizer-extra-data", $Path)
}
foreach ($Path in $SnapshotProvenance) {
  $TrainArgs += @("--provenance-file", $Path)
}
$TrainArgs += @(
  "--tokenizer-source-label", "train_split_plus_core_japanese_inventory_plus_utf8_byte_fallback_v1",
  "--tokenizer-type", "byte_fallback_char_v1",
  "--fail-on-val-oov",
  "--out", $Out,
  "--best-out", $BestOut,
  "--run-id", $RunId,
  "--require-supervisor",
  "--seed", "20260509",
  "--steps", "8000",
  "--batch-size", "2",
  "--grad-accum-steps", "8",
  "--block-size", "384",
  "--n-layer", "16",
  "--n-head", "12",
  "--num-key-value-heads", "6",
  "--n-embd", "768",
  "--intermediate-size", "2304",
  "--dropout", "0.05",
  "--eval-every", "250",
  "--log-every", "25",
  "--save-every", "1000",
  "--early-stop-patience", "10",
  "--overfit-stop-gap", "3.0",
  "--overfit-stop-after-evals", "5",
  "--overfit-stop-min-step", "1000",
  "--optimizer", "simple-adamw",
  "--lr", "2e-4",
  "--min-lr", "2e-5",
  "--warmup-steps", "400",
  "--cosine-lr",
  "--grad-clip", "1.0",
  "--qwen3-style",
  "--qk-norm",
  "--device", "dml",
  "--release-name", "old-japanese-0.1B-preview"
)

$Phase = "training"
  [System.IO.File]::WriteAllText(
    [IO.Path]::GetFullPath((Join-Path $Root $StdoutLog)),
    "run_id=$RunId launcher=train_old_japanese_0_1b_dml.ps1 started=$(Get-Date -Format o)" + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
  )
  [System.IO.File]::WriteAllText(
    [IO.Path]::GetFullPath((Join-Path $Root $StderrLog)),
    "",
    [System.Text.UTF8Encoding]::new($false)
  )
  & $CapturePython scripts\run_command_capture.py --stdout $StdoutLog --stderr $StderrLog -- $Python @TrainArgs
  $TrainExit = $LASTEXITCODE
  if ($TrainExit -ne 0) {
    $TrainMessage = "training command failed with exit code $TrainExit"
    throw $TrainMessage
  }
  $TrainMessage = "completed"
  "launcher_completed=$(Get-Date -Format o) exit_code=$TrainExit" | Add-Content -Encoding UTF8 $StdoutLog
} catch {
  if (-not $TrainMessage) {
    $TrainMessage = "phase=$Phase message=$($_.Exception.Message)"
  }
  if (Test-Path $StderrLog) {
    "launcher_failed=$(Get-Date -Format o) message=$TrainMessage" | Add-Content -Encoding UTF8 $StderrLog
  }
  throw
} finally {
  Write-TrainExitSentinel -ExitCode $TrainExit -Message $TrainMessage
}
