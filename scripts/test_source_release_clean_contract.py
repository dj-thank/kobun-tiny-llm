from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_source_release_clean.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_source_release_clean", CHECKER)
    if spec is None or spec.loader is None:
        raise SystemExit("failed to load check_source_release_clean.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tracked_data_paths() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "data"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def main() -> None:
    checker = load_checker()
    scanned = {checker.rel(path) for path in checker.public_source_files()}
    tracked_data = tracked_data_paths()

    required_paths = {
        ".gitattributes",
        "data/eval/grammar_minimal_pairs_heldout.jsonl",
        "data/grammar/rules.jsonl",
        "data/rules/waka_meter_rules.json",
        "data/tokenizer_public_char_vocab.meta.json",
        "data/training_augmentation_manifest.json",
    }
    missing_required = sorted(required_paths - scanned)
    if missing_required:
        raise SystemExit(f"source_release_required_paths_not_scanned={missing_required}")

    outside_allowlist = sorted(path for path in tracked_data if not checker.is_public_data_path(path))
    if outside_allowlist:
        raise SystemExit(f"tracked_data_not_declared_public={outside_allowlist}")

    missing_scans = sorted(path for path in tracked_data if path not in scanned)
    if missing_scans:
        raise SystemExit(f"tracked_public_data_not_scanned={missing_scans}")

    public_data_scanned = [path for path in scanned if path.startswith("data/")]
    if len(public_data_scanned) < 48:
        raise SystemExit(f"too_few_public_data_files_scanned={len(public_data_scanned)}")

    print(f"source_release_clean_contract_ok=true public_data_files={len(public_data_scanned)}")


if __name__ == "__main__":
    main()
