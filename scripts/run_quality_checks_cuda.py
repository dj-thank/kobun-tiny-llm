from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CUDA post-run quality checks for an exact best checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--eval-json", type=Path, default=None)
    return parser.parse_args()


def append_log(log: Path, text: str) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        if text and not text.endswith("\n"):
            handle.write("\n")


def run_command(command: list[str], log: Path) -> str:
    append_log(log, "COMMAND " + " ".join(command))
    completed = subprocess.run(command, text=True, capture_output=True, check=False, env={**os.environ, "PYTHONUTF8": "1"})
    output = completed.stdout + completed.stderr
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
        append_log(log, output)
    append_log(log, f"EXIT {completed.returncode} {' '.join(command)}")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed exit={completed.returncode}: {' '.join(command)}")
    return output


def parse_path_from_output(output: str, prefix: str) -> str:
    for line in output.splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"missing {prefix} in command output")


def write_eval_json(log: Path, eval_json: Path, checkpoint: Path, status: str) -> None:
    command = [
        sys.executable,
        "scripts/parse_quality_log.py",
        "--log",
        str(log),
        "--out",
        str(eval_json),
        "--checkpoint",
        str(checkpoint),
        "--device",
        "cuda",
        "--status",
        status,
        "--runner",
        "scripts/run_quality_checks_cuda.py",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    output = completed.stdout + completed.stderr
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
        append_log(log, output)
    if completed.returncode != 0:
        raise RuntimeError(f"could not write eval JSON exit={completed.returncode}: {' '.join(command)}")
    if not eval_json.exists():
        raise RuntimeError(f"parse_quality_log reported success but did not create eval JSON: {eval_json}")
    payload = json.loads(eval_json.read_text(encoding="utf-8-sig"))
    if str(payload.get("checkpoint") or "") not in {str(checkpoint), checkpoint.as_posix()}:
        raise RuntimeError(f"eval JSON is not bound to checkpoint: {eval_json}")


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    checkpoint = args.checkpoint
    checkpoint_base = checkpoint.stem
    run_id = checkpoint_base.removesuffix("_best")
    log = args.log or Path("logs") / f"quality_{run_id}.log"
    eval_json = args.eval_json or Path("logs") / f"eval_results_{run_id}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("", encoding="utf-8")

    py = sys.executable
    try:
        run_command(
            [
                py,
                "-c",
                (
                    "import sys, torch; "
                    "from kobun_llm.device import cuda_runtime_kind, require_real_cuda_runtime; "
                    "require_real_cuda_runtime('CUDA quality checks'); "
                    "print('python=' + sys.executable); "
                    "print('torch=' + torch.__version__); "
                    "print('torch_cuda_version=' + str(torch.version.cuda or '')); "
                    "print('torch_hip_version=' + str(getattr(torch.version, 'hip', '') or '')); "
                    "print('cuda_runtime_kind=' + cuda_runtime_kind()); "
                    "print('real_cuda_runtime=true'); "
                    "print('cuda_available=' + str(torch.cuda.is_available())); "
                    "print('cuda_device=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''))"
                ),
            ],
            log,
        )
        run_command(
            [
                py,
                "scripts/check_run_completion.py",
                "--run-id",
                run_id,
                "--checkpoint",
                str(checkpoint),
                "--backend",
                "cuda",
                "--require-no-active-process",
                "--ignore-pid",
                str(os.getppid()),
            ],
            log,
        )
        run_command(
            [
                py,
                "scripts/check_checkpoint_model_size.py",
                "--checkpoint",
                str(checkpoint),
                "--strict-config",
                "--require-release-prefix",
                "old-japanese-0.1B",
                "--fail-on-val-oov",
                "--require-from-scratch",
                "--require-seed",
                "--require-optimizer",
                "simple-adamw",
                "--require-backend",
                "cuda",
            ],
            log,
        )
        input_output = run_command(
            [
                py,
                "scripts/check_checkpoint_training_inputs.py",
                "--checkpoint",
                str(checkpoint),
                "--require-val-data",
                "--require-test-data",
                "--require-from-scratch",
                "--require-run-snapshot",
                "--allow-same-run-resume",
            ],
            log,
        )
        train_data = parse_path_from_output(input_output, "train_data_path=")
        val_data = parse_path_from_output(input_output, "val_data_path=")
        test_data = parse_path_from_output(input_output, "test_data_path=")
        snapshot_manifest = parse_path_from_output(input_output, "provenance_file_path=")
        provenance_paths = [line.split("=", 1)[1].strip() for line in input_output.splitlines() if line.startswith("provenance_file_path=")]
        corpus_manifest = next(path for path in provenance_paths if path.endswith("corpus_manifest.jsonl"))
        aozora_sources = next(path for path in provenance_paths if path.endswith("aozora_sources.json"))
        waka_sources = next(path for path in provenance_paths if path.endswith("waka_sources.json"))
        tokenizer_meta = next(path for path in provenance_paths if path.endswith("tokenizer_public_char_vocab.meta.json"))
        tokenizer_extra = parse_path_from_output(input_output, "tokenizer_extra_data_path=")

        for command in (
            [py, "scripts/audit_rule_tables.py"],
            [py, "scripts/audit_eval_provenance_manifest.py"],
            [py, "scripts/audit_source_records.py", aozora_sources, waka_sources],
            [py, "scripts/audit_public_manifest.py", "--manifest", corpus_manifest, "--out", f"logs/public_manifest_summary_{checkpoint_base}.json"],
            [py, "scripts/eval_waka_meter_constraints.py", "--cases", "data/eval/waka_meter_constraint_cases.jsonl", "--min-cases", "19"],
            [py, "scripts/check_tokenizer_vocab_scope.py", "--manifest", corpus_manifest, "--tokenizer-extra-data", tokenizer_extra, "--tokenizer-meta", tokenizer_meta],
            [py, "scripts/check_checkpoint_tokenizer_scope.py", "--checkpoint", str(checkpoint), "--manifest", corpus_manifest, "--tokenizer-extra-data", tokenizer_extra, "--tokenizer-meta", tokenizer_meta],
            [py, "scripts/validate_corpus.py", train_data],
            [py, "scripts/validate_corpus.py", val_data],
            [py, "scripts/validate_corpus.py", test_data],
            [py, "scripts/validate_corpus.py", "data/waka/waka_corpus_all.txt", "--kind", "waka-poems"],
            [py, "scripts/check_split_consistency.py", "--checkpoint", str(checkpoint)],
            [py, "scripts/check_split_leakage.py", "--manifest", corpus_manifest, "--train", train_data],
            [
                py,
                "scripts/check_eval_source_overlap.py",
                "--manifest",
                corpus_manifest,
                "--eval",
                "data/eval/grammar_minimal_pairs.jsonl",
                "data/eval/grammar_minimal_pairs_heldout.jsonl",
                "data/eval/morphology_adversarial_cases.jsonl",
                "data/eval/grammar_constraint_cases.jsonl",
                "data/eval/waka_rule_cases.jsonl",
                "data/eval/waka_meter_constraint_cases.jsonl",
                "data/eval/waka_generation_prompts.jsonl",
            ],
            [
                py,
                "scripts/check_eval_contamination.py",
                "--train",
                train_data,
                "--template-train",
                "data/grammar/train_preference_pairs.jsonl",
                "--eval",
                "data/eval/grammar_minimal_pairs.jsonl",
                "data/eval/grammar_minimal_pairs_heldout.jsonl",
                "data/eval/morphology_adversarial_cases.jsonl",
                "data/eval/grammar_constraint_cases.jsonl",
                "data/eval/waka_rule_cases.jsonl",
                "data/eval/waka_meter_constraint_cases.jsonl",
                "data/eval/waka_generation_prompts.jsonl",
                "--strict-prompts",
                "--write-clean-dir",
                "data/eval/clean_current",
            ],
            [
                py,
                "scripts/check_eval_contamination.py",
                "--train",
                train_data,
                "--template-train",
                "data/grammar/train_preference_pairs.jsonl",
                "--eval",
                "data/eval/clean_current/grammar_minimal_pairs.jsonl",
                "data/eval/clean_current/grammar_minimal_pairs_heldout.jsonl",
                "data/eval/clean_current/morphology_adversarial_cases.jsonl",
                "data/eval/clean_current/grammar_constraint_cases.jsonl",
                "data/eval/clean_current/waka_rule_cases.jsonl",
                "data/eval/clean_current/waka_meter_constraint_cases.jsonl",
                "data/eval/clean_current/waka_generation_prompts.jsonl",
                "--strict-prompts",
            ],
            [
                py,
                "scripts/check_eval_source_overlap.py",
                "--manifest",
                corpus_manifest,
                "--eval",
                "data/eval/clean_current/grammar_minimal_pairs.jsonl",
                "data/eval/clean_current/grammar_minimal_pairs_heldout.jsonl",
                "data/eval/clean_current/morphology_adversarial_cases.jsonl",
                "data/eval/clean_current/grammar_constraint_cases.jsonl",
                "data/eval/clean_current/waka_rule_cases.jsonl",
                "data/eval/clean_current/waka_meter_constraint_cases.jsonl",
                "data/eval/clean_current/waka_generation_prompts.jsonl",
            ],
        ):
            run_command(command, log)

        eval_snapshot_dir = f"logs/eval_snapshots/{checkpoint_base}"
        run_command(
            [
                py,
                "scripts/snapshot_eval_files.py",
                "--out-dir",
                eval_snapshot_dir,
                "--named",
                "primary=data/eval/clean_current/grammar_minimal_pairs.jsonl",
                "--named",
                "heldout=data/eval/clean_current/grammar_minimal_pairs_heldout.jsonl",
                "--named",
                "morphology=data/eval/clean_current/morphology_adversarial_cases.jsonl",
                "--named",
                "grammar_constraints=data/eval/clean_current/grammar_constraint_cases.jsonl",
                "--named",
                "waka_rules=data/eval/clean_current/waka_rule_cases.jsonl",
                "--named",
                "waka_meter_constraints=data/eval/clean_current/waka_meter_constraint_cases.jsonl",
                "--named",
                "waka_generation_prompts=data/eval/clean_current/waka_generation_prompts.jsonl",
            ],
            log,
        )
        run_command([py, "scripts/eval_grammar_constraints.py", "--cases", f"{eval_snapshot_dir}/grammar_constraints.jsonl", "--min-cases", "28"], log)
        run_command([py, "scripts/eval_waka_rules.py", "--cases", f"{eval_snapshot_dir}/waka_rules.jsonl", "--min-cases", "20"], log)
        run_command([py, "scripts/eval_waka_meter_constraints.py", "--cases", f"{eval_snapshot_dir}/waka_meter_constraints.jsonl", "--min-cases", "19"], log)
        run_command([py, "scripts/eval_heldout_lm.py", "--checkpoint", str(checkpoint), "--device", "cuda", "--data", test_data, "--split-name", "test", "--max-loss", "8.0"], log)
        run_command([py, "scripts/eval_minimal_pairs.py", "--checkpoint", str(checkpoint), "--device", "cuda", "--pairs", f"{eval_snapshot_dir}/primary.jsonl", "--metric-prefix", "primary", "--min-cases", "8", "--min-accuracy", "1.0"], log)
        run_command([py, "scripts/eval_minimal_pairs.py", "--checkpoint", str(checkpoint), "--device", "cuda", "--pairs", f"{eval_snapshot_dir}/heldout.jsonl", "--metric-prefix", "heldout", "--min-cases", "12", "--min-accuracy", "1.0"], log)
        run_command([py, "scripts/eval_morphology_adversarial.py", "--cases", f"{eval_snapshot_dir}/morphology.jsonl", "--min-cases", "4"], log)
        run_command([py, "scripts/eval_waka_meter_generation.py", "--checkpoint", str(checkpoint), "--device", "cuda", "--decoding", "greedy", "--seed", "20260509", "--prompts-file", f"{eval_snapshot_dir}/waka_generation_prompts.jsonl", "--min-cases", "4"], log)
    except Exception as exc:
        append_log(log, f"QUALITY_FAILED {exc}")
        write_eval_json(log, eval_json, checkpoint, "failed")
        raise SystemExit(str(exc)) from exc

    write_eval_json(log, eval_json, checkpoint, "passed")
    print(f"quality_log={log}")
    print(f"eval_json={eval_json}")


if __name__ == "__main__":
    main()
