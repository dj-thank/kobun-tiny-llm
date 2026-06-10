from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from kobun_llm.checkpoint_io import load_trusted_checkpoint
from kobun_llm.release_resume import validate_release_resume_chain_from_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify checkpoint training-input metadata against local files.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--require-val-data", action="store_true")
    parser.add_argument("--require-test-data", action="store_true")
    parser.add_argument("--require-from-scratch", action="store_true")
    parser.add_argument("--require-run-snapshot", action="store_true")
    parser.add_argument(
        "--allow-same-run-resume",
        action="store_true",
        help="Allow resume metadata only when the resume checkpoint is readable and has the same run/data hashes.",
    )
    return parser.parse_args()


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_from_checkpoint(path: Path) -> dict[str, Any]:
    payload = load_trusted_checkpoint(path, map_location="cpu")
    return dict(payload)


def metadata_from_checkpoint(path: Path) -> dict[str, Any]:
    return dict(payload_from_checkpoint(path).get("metadata", {}) or {})


def resolve_project_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def require_file_hash(metadata: dict[str, Any], path_key: str, hash_key: str, label: str) -> Path:
    raw_path = str(metadata.get(path_key) or "")
    expected_hash = str(metadata.get(hash_key) or "")
    if not raw_path:
        raise SystemExit(f"checkpoint metadata missing {path_key}")
    if not expected_hash:
        raise SystemExit(f"checkpoint metadata missing {hash_key}")
    path = resolve_project_path(raw_path)
    if not path.exists():
        raise SystemExit(f"checkpoint metadata {label} path does not exist: {raw_path}")
    actual_hash = sha256_text(path)
    if actual_hash != expected_hash:
        raise SystemExit(
            f"checkpoint metadata {label} hash mismatch: "
            f"path={raw_path} expected={expected_hash} actual={actual_hash}"
        )
    return path


def require_snapshot_path(path: Path, run_id: str, label: str) -> None:
    if not re.fullmatch(r"old_japanese_0_1b(?:_dml|_cuda|_hip)?_[0-9A-Za-z][0-9A-Za-z_-]{0,63}", run_id):
        raise SystemExit(f"checkpoint metadata has invalid run_id: {run_id!r}")
    snapshot_root = (Path.cwd() / "data" / "run_snapshots" / run_id).resolve(strict=False)
    resolved = path.resolve(strict=False)
    if snapshot_root != resolved and snapshot_root not in resolved.parents:
        raise SystemExit(f"{label} is not inside run snapshot {snapshot_root}: {path}")


def require_not_release_source_path(raw_path: str, label: str) -> None:
    if not raw_path:
        raise SystemExit(f"{label} has empty source_path")
    path = resolve_project_path(raw_path)
    release_root = (Path.cwd() / "release").resolve(strict=False)
    resolved = path.resolve(strict=False)
    if release_root == resolved or release_root in resolved.parents:
        raise SystemExit(f"{label} source_path crosses release artifact boundary: {raw_path}")


def verify_snapshot_manifest_boundaries(snapshot_manifest_path: Path) -> None:
    payload = json.loads(snapshot_manifest_path.read_text(encoding="utf-8-sig"))
    entries: list[tuple[str, Any]] = [
        ("data", payload.get("data")),
        ("val_data", payload.get("val_data")),
        ("test_data", payload.get("test_data")),
    ]
    for index, item in enumerate(payload.get("tokenizer_extra_data") or []):
        entries.append((f"tokenizer_extra_data[{index}]", item))
    for index, item in enumerate(payload.get("provenance_files") or []):
        entries.append((f"provenance_files[{index}]", item))
    if not entries:
        raise SystemExit(f"snapshot manifest has no input records: {snapshot_manifest_path}")
    for label, record in entries:
        if not isinstance(record, dict):
            raise SystemExit(f"snapshot manifest {label} must be an object")
        require_not_release_source_path(str(record.get("source_path") or ""), f"snapshot manifest {label}")


def verify_file_records(
    metadata: dict[str, Any],
    key: str,
    label: str,
    require_records: bool,
    require_run_snapshot: bool,
    run_id: str,
) -> list[Path]:
    records = metadata.get(key) or []
    if require_records and not records:
        raise SystemExit(f"checkpoint metadata missing {key}")
    if not isinstance(records, list):
        raise SystemExit(f"checkpoint metadata {key} must be a list")
    paths: list[Path] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SystemExit(f"checkpoint metadata {key}[{index}] must be an object")
        raw_path = str(record.get("path") or "")
        expected_hash = str(record.get("sha256") or "")
        expected_bytes = record.get("bytes")
        if not raw_path or not expected_hash:
            raise SystemExit(f"checkpoint metadata {key}[{index}] missing path or sha256")
        path = resolve_project_path(raw_path)
        if not path.exists():
            raise SystemExit(f"checkpoint metadata {key}[{index}] path does not exist: {raw_path}")
        if sha256_file(path) != expected_hash:
            raise SystemExit(f"checkpoint metadata {key}[{index}] hash mismatch: {raw_path}")
        if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
            raise SystemExit(f"checkpoint metadata {key}[{index}] byte size mismatch: {raw_path}")
        if require_run_snapshot:
            require_snapshot_path(path, run_id, f"{label}[{index}]")
        paths.append(path)
    return paths


def verify_resume_chain(
    checkpoint_path: Path,
    metadata: dict[str, Any],
    allow_same_run_resume: bool,
    seen: set[str] | None = None,
) -> None:
    payload = payload_from_checkpoint(checkpoint_path)
    try:
        validate_release_resume_chain_from_payload(
            payload,
            checkpoint_path,
            allow_same_run_resume=allow_same_run_resume,
            expected_backend=str(metadata.get("backend") or ""),
            expected_seed=metadata.get("seed"),
            expected_optimizer=str(metadata.get("optimizer") or ""),
            expected_config=payload.get("config"),
            expected_tokenizer=payload.get("tokenizer"),
            expected_tokenizer_extra_data=metadata.get("tokenizer_extra_data") or [],
            expected_provenance_files=metadata.get("provenance_files") or [],
            load_checkpoint=payload_from_checkpoint,
            resolve_path=resolve_project_path,
            seen=seen,
        )
    except ValueError as exc:
        raise SystemExit(f"release resume chain is unsafe: {exc}") from exc


def main() -> None:
    args = parse_args()
    metadata = metadata_from_checkpoint(args.checkpoint)
    run_id = str(metadata.get("run_id") or "")
    if args.require_run_snapshot and not run_id:
        raise SystemExit("checkpoint metadata missing run_id for snapshot validation")
    if args.require_from_scratch and str(metadata.get("init_from") or ""):
        raise SystemExit(f"checkpoint was initialized from another checkpoint: init_from={metadata.get('init_from')!r}")
    verify_resume_chain(args.checkpoint, metadata, args.allow_same_run_resume)

    train_path = require_file_hash(metadata, "data_path", "data_sha256", "train")
    if args.require_run_snapshot:
        require_snapshot_path(train_path, run_id, "train_data_path")
    val_path = None
    if args.require_val_data or str(metadata.get("val_data_path") or ""):
        val_path = require_file_hash(metadata, "val_data_path", "val_data_sha256", "validation")
        if args.require_run_snapshot:
            require_snapshot_path(val_path, run_id, "val_data_path")
    test_path = None
    if args.require_test_data or str(metadata.get("test_data_path") or ""):
        test_path = require_file_hash(metadata, "test_data_path", "test_data_sha256", "test")
        if args.require_run_snapshot:
            require_snapshot_path(test_path, run_id, "test_data_path")
    tokenizer_paths = verify_file_records(
        metadata,
        "tokenizer_extra_data",
        "tokenizer_extra_data",
        args.require_run_snapshot,
        args.require_run_snapshot,
        run_id,
    )
    provenance_paths = verify_file_records(
        metadata,
        "provenance_files",
        "provenance_files",
        args.require_run_snapshot,
        args.require_run_snapshot,
        run_id,
    )
    if args.require_run_snapshot:
        provenance_names = {path.name for path in provenance_paths}
        required_provenance = {
            "aozora_sources.json",
            "corpus_manifest.jsonl",
            "public_manifest_summary.json",
            "snapshot_manifest.json",
            "tokenizer_public_char_vocab.meta.json",
            "training_augmentation_manifest.json",
            "waka_sources.json",
        }
        missing = sorted(required_provenance - provenance_names)
        if missing:
            raise SystemExit(f"checkpoint provenance snapshot missing required files: {missing}")
        snapshot_manifest_paths = [path for path in provenance_paths if path.name == "snapshot_manifest.json"]
        if len(snapshot_manifest_paths) != 1:
            raise SystemExit("checkpoint provenance snapshot must contain exactly one snapshot_manifest.json")
        verify_snapshot_manifest_boundaries(snapshot_manifest_paths[0])

    print(f"train_data_path={train_path}")
    if val_path is not None:
        print(f"val_data_path={val_path}")
    if test_path is not None:
        print(f"test_data_path={test_path}")
    for path in tokenizer_paths:
        print(f"tokenizer_extra_data_path={path}")
    for path in provenance_paths:
        print(f"provenance_file_path={path}")
    print(f"run_id={metadata.get('run_id', '')}")
    print("checkpoint_training_inputs_ok=true")


if __name__ == "__main__":
    main()
