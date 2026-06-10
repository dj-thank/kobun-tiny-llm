from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_file(source: Path, destination: Path) -> dict[str, Any]:
    if not source.exists():
        raise SystemExit(f"missing input file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "source_path": str(source),
        "snapshot_path": str(destination),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def safe_name(path: Path, prefix: str) -> str:
    name = path.as_posix().replace("/", "__").replace("\\", "__").replace(":", "")
    return f"{prefix}_{name}"


def provenance_name(path: Path) -> str:
    normalized = path.as_posix().lower()
    if normalized.endswith("data/corpus_manifest.jsonl"):
        return "corpus_manifest.jsonl"
    if normalized.endswith("logs/public_manifest_summary.json"):
        return "public_manifest_summary.json"
    if normalized.endswith("data/aozora/sources.json"):
        return "aozora_sources.json"
    if normalized.endswith("data/waka/sources.json"):
        return "waka_sources.json"
    if normalized.endswith("data/tokenizer_public_char_vocab.meta.json"):
        return "tokenizer_public_char_vocab.meta.json"
    if normalized.endswith("data/training_augmentation_manifest.json"):
        return "training_augmentation_manifest.json"
    return safe_name(path, "provenance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy mutable training inputs into a run-specific snapshot.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-root", type=Path, default=Path("data/run_snapshots"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--tokenizer-extra-data", action="append", type=Path, default=[])
    parser.add_argument("--provenance-file", action="append", type=Path, default=[])
    return parser.parse_args()


def validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"old_japanese_0_1b(?:_dml|_cuda|_hip)?_[0-9A-Za-z][0-9A-Za-z_-]{0,63}", run_id):
        raise SystemExit(f"invalid run id for snapshot directory: {run_id!r}")


def ensure_under(base: Path, path: Path, label: str) -> None:
    base_resolved = base.resolve(strict=False)
    path_resolved = path.resolve(strict=False)
    if base_resolved != path_resolved and base_resolved not in path_resolved.parents:
        raise SystemExit(f"{label} escapes expected directory: {path}")


def main() -> None:
    args = parse_args()
    validate_run_id(args.run_id)
    snapshot_dir = args.out_root / args.run_id
    ensure_under(args.out_root, snapshot_dir, "snapshot_dir")
    if snapshot_dir.exists() and any(snapshot_dir.iterdir()):
        raise SystemExit(f"snapshot directory already exists and is not empty: {snapshot_dir}")

    data = copy_file(args.data, snapshot_dir / "train.txt")
    val_data = copy_file(args.val_data, snapshot_dir / "validation.txt")
    test_data = copy_file(args.test_data, snapshot_dir / "test.txt")
    tokenizer_extra = [
        copy_file(path, snapshot_dir / "tokenizer" / safe_name(path, "tokenizer"))
        for path in args.tokenizer_extra_data
    ]
    provenance = [
        copy_file(path, snapshot_dir / "provenance" / provenance_name(path))
        for path in args.provenance_file
    ]

    manifest = {
        "run_id": args.run_id,
        "snapshot_dir": str(snapshot_dir),
        "data": data,
        "val_data": val_data,
        "test_data": test_data,
        "tokenizer_extra_data": tokenizer_extra,
        "provenance_files": provenance,
        "policy": "Training checkpoints for this run must point at these immutable input snapshots, not mutable data/ build outputs.",
    }
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    manifest["snapshot_manifest"] = {
        "snapshot_path": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"snapshot_dir={snapshot_dir}")
    print(f"snapshot_train_data={data['snapshot_path']}")
    print(f"snapshot_val_data={val_data['snapshot_path']}")
    print(f"snapshot_test_data={test_data['snapshot_path']}")
    for item in tokenizer_extra:
        print(f"snapshot_tokenizer_extra_data={item['snapshot_path']}")
    for item in provenance:
        print(f"snapshot_provenance_file={item['snapshot_path']}")
    print(f"snapshot_provenance_file={manifest['snapshot_manifest']['snapshot_path']}")


if __name__ == "__main__":
    main()
