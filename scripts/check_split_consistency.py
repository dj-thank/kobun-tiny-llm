from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_training_corpus import manifest_text, read_manifest_rows, split_manifest_rows_three
from kobun_llm.checkpoint_io import load_trusted_checkpoint
from kobun_autonomy.augmentation_audit import require_clean_augmentation_manifest
from split_policy import SPLIT_POLICY, split_group_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify train/validation/test files against the manifest split.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus_manifest.jsonl"))
    parser.add_argument("--train", type=Path, default=Path("data/kobun_worldclass_corpus.txt"))
    parser.add_argument("--val", type=Path, default=Path("data/kobun_labeled_grammar_val.txt"))
    parser.add_argument("--test", type=Path, default=Path("data/kobun_labeled_grammar_test.txt"))
    parser.add_argument("--augmentation-manifest", type=Path, default=Path("data/training_augmentation_manifest.json"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--min-test-sources", type=int, default=3)
    parser.add_argument("--min-test-groups", type=int, default=1)
    parser.add_argument("--min-test-source-chars", type=int, default=30_000)
    parser.add_argument("--min-test-text-chars", type=int, default=20_000)
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_project_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def checkpoint_provenance_path(metadata: dict[str, Any], filename: str) -> Path:
    for record in metadata.get("provenance_files", []) or []:
        if not isinstance(record, dict):
            continue
        raw_path = str(record.get("path") or "")
        expected_hash = str(record.get("sha256") or "")
        if Path(raw_path).name != filename:
            continue
        path = resolve_project_path(raw_path)
        if not path.exists():
            raise SystemExit(f"checkpoint provenance file is missing: {raw_path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise SystemExit(
                f"checkpoint provenance hash mismatch for {filename}: expected={expected_hash} actual={actual_hash}"
            )
        return path
    raise SystemExit(f"checkpoint metadata missing required provenance file: {filename}")


def checkpoint_paths(checkpoint: Path) -> tuple[Path, Path, Path, Path, Path | None]:
    payload = load_trusted_checkpoint(checkpoint, map_location="cpu")
    metadata = dict(payload.get("metadata", {}) or {})
    manifest = checkpoint_provenance_path(metadata, "corpus_manifest.jsonl")
    augmentation_manifest: Path | None = None
    try:
        augmentation_manifest = checkpoint_provenance_path(metadata, "training_augmentation_manifest.json")
    except SystemExit:
        augmentation_manifest = None
    train = resolve_project_path(str(metadata.get("data_path") or ""))
    val = resolve_project_path(str(metadata.get("val_data_path") or ""))
    test = resolve_project_path(str(metadata.get("test_data_path") or ""))
    for label, path in (("train", train), ("validation", val), ("test", test)):
        if not path.exists():
            raise SystemExit(f"checkpoint {label} path does not exist: {path}")
    return manifest, train, val, test, augmentation_manifest


def split_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    chars = sum(int(row.get("characters") or 0) for row in rows)
    genres = Counter(str(row.get("genre", "unknown")) for row in rows)
    styles = Counter(str(row.get("style", "unknown")) for row in rows)
    titles = [str(row.get("title", "")) for row in rows]
    groups = sorted({split_group_key(row) for row in rows})
    return {
        "sources": len(rows),
        "groups": groups,
        "group_count": len(groups),
        "chars": chars,
        "genres": dict(sorted(genres.items())),
        "styles": dict(sorted(styles.items())),
        "titles": titles,
    }


def expected_split_texts(
    manifest: Path,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], str, str, str]:
    rows = read_manifest_rows(manifest)
    train_rows, val_rows, test_rows = split_manifest_rows_three(rows, val_ratio, test_ratio)
    train_text = manifest_text(train_rows).strip() + "\n"
    val_text = manifest_text(val_rows).strip() + "\n"
    test_text = manifest_text(test_rows).strip() + "\n"
    return train_rows, val_rows, test_rows, train_text, val_text, test_text


def manifest_entries_by_role(path: Path) -> dict[str, dict[str, Any]]:
    require_clean_augmentation_manifest(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    entries: dict[str, dict[str, Any]] = {}
    for entry in payload.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "")
        if role:
            entries[role] = entry
    return entries


def read_entry_text(entries: dict[str, dict[str, Any]], role: str) -> str:
    entry = entries.get(role)
    if entry is None:
        raise SystemExit(f"augmentation manifest missing role: {role}")
    path = Path(str(entry.get("path") or ""))
    if not path.exists():
        raise SystemExit(f"augmentation source missing for role {role}: {path}")
    expected_hash = str(entry.get("sha256") or "")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise SystemExit(f"augmentation source hash mismatch for role {role}: {path}")
    return path.read_text(encoding="utf-8").strip()


def repeat_count(entries: dict[str, dict[str, Any]], role: str) -> int:
    entry = entries.get(role)
    if entry is None:
        raise SystemExit(f"augmentation manifest missing role: {role}")
    value = entry.get("repeat_count")
    if not isinstance(value, int) or value <= 0:
        raise SystemExit(f"augmentation manifest role {role} has invalid repeat_count: {value!r}")
    return value


def expected_augmented_train_text(manifest_train_text: str, augmentation_manifest: Path) -> str:
    entries = manifest_entries_by_role(augmentation_manifest)

    grammar_parts = [manifest_train_text.strip()]
    grammar = read_entry_text(entries, "grammar_rule_text")
    for index in range(repeat_count(entries, "grammar_rule_text")):
        grammar_parts.append(f"文法注入 {index + 1}\n{grammar}")
    morph = read_entry_text(entries, "morphology_examples")
    for index in range(repeat_count(entries, "morphology_examples")):
        grammar_parts.append(f"形態素注入 {index + 1}\n{morph}")
    grammar_train = "\n\n".join(grammar_parts) + "\n"

    preference_lines: list[str] = []
    for line in read_entry_text(entries, "train_preference_pairs").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        preference_lines.append(str(row["prompt"]) + str(row["good"]))
    preference_block = "\n".join(preference_lines)
    boost_parts = [grammar_train.strip()]
    for index in range(repeat_count(entries, "train_preference_pairs")):
        boost_parts.append(f"選好注入 {index + 1}\n{preference_block}")
    boost_train = "\n\n".join(boost_parts) + "\n"

    world_parts = [boost_train.strip()]
    waka_meter = read_entry_text(entries, "waka_meter_training_text")
    for index in range(repeat_count(entries, "waka_meter_training_text")):
        world_parts.append(f"和歌音数制御 五七五七七 反復 {index + 1}\n{waka_meter}")
    aux_rules = read_entry_text(entries, "auxiliary_rule_table")
    genre_rules = read_entry_text(entries, "genre_rule_table")
    rule_repeat = repeat_count(entries, "auxiliary_rule_table")
    if repeat_count(entries, "genre_rule_table") != rule_repeat:
        raise SystemExit("auxiliary_rule_table and genre_rule_table repeat_count must match")
    for index in range(rule_repeat):
        world_parts.append(f"助動詞接続活用表 {index + 1}\n{aux_rules}")
        world_parts.append(f"ジャンル別規則表 {index + 1}\n{genre_rules}")
    external_surfaces = read_entry_text(entries, "external_knowledge_surface_patterns")
    for index in range(repeat_count(entries, "external_knowledge_surface_patterns")):
        world_parts.append(f"古語表面形 {index + 1}\n{external_surfaces}")
    return "\n\n".join(world_parts) + "\n"


def main() -> None:
    args = parse_args()
    manifest = args.manifest
    train = args.train
    val = args.val
    test = args.test
    augmentation_manifest: Path | None = args.augmentation_manifest
    if args.checkpoint is not None:
        manifest, train, val, test, augmentation_manifest = checkpoint_paths(args.checkpoint)

    train_rows, val_rows, test_rows, expected_train, expected_val, expected_test = expected_split_texts(
        manifest,
        args.val_ratio,
        args.test_ratio,
    )
    train_text = train.read_text(encoding="utf-8")
    val_text = val.read_text(encoding="utf-8")
    test_text = test.read_text(encoding="utf-8")
    validation_match = sha256_text(val_text) == sha256_text(expected_val)
    test_match = sha256_text(test_text) == sha256_text(expected_test)
    if augmentation_manifest is None or not augmentation_manifest.exists():
        raise SystemExit(f"augmentation manifest is required for full train reconstruction: {augmentation_manifest}")
    expected_train_full = expected_augmented_train_text(expected_train, augmentation_manifest)
    train_match = sha256_text(train_text) == sha256_text(expected_train_full)

    test_summary = split_summary(test_rows)
    validation_summary = split_summary(val_rows)
    train_summary = split_summary(train_rows)
    if not validation_match:
        raise SystemExit(
            f"validation split mismatch: actual_sha256={sha256_text(val_text)} expected_sha256={sha256_text(expected_val)}"
        )
    if not test_match:
        raise SystemExit(
            f"test split mismatch: actual_sha256={sha256_text(test_text)} expected_sha256={sha256_text(expected_test)}"
        )
    if not train_match:
        raise SystemExit(
            f"train split full reconstruction mismatch: train={train} "
            f"actual_sha256={sha256_text(train_text)} expected_sha256={sha256_text(expected_train_full)} "
            f"augmentation_manifest={augmentation_manifest}"
        )
    if int(test_summary["sources"]) < args.min_test_sources:
        raise SystemExit(
            f"test split has too few sources: {test_summary['sources']} < {args.min_test_sources}"
        )
    if int(test_summary["group_count"]) < args.min_test_groups:
        raise SystemExit(
            f"test split has too few groups: {test_summary['group_count']} < {args.min_test_groups}"
        )
    if int(test_summary["chars"]) < args.min_test_source_chars:
        raise SystemExit(
            f"test split has too few source chars: {test_summary['chars']} < {args.min_test_source_chars}"
        )
    if len(test_text) < args.min_test_text_chars:
        raise SystemExit(
            f"test split has too few generated test text chars: {len(test_text)} < {args.min_test_text_chars}"
        )
    train_groups = set(train_summary["groups"])
    validation_groups = set(validation_summary["groups"])
    test_groups = set(test_summary["groups"])
    if train_groups & validation_groups or train_groups & test_groups or validation_groups & test_groups:
        raise SystemExit(
            f"split groups are not disjoint: train={sorted(train_groups)} validation={sorted(validation_groups)} test={sorted(test_groups)}"
        )

    print(
        "split_consistency "
        f"split_policy={SPLIT_POLICY} "
        f"manifest={manifest} "
        f"manifest_sha256={sha256_file(manifest)} "
        f"train={train} "
        f"validation={val} "
        f"test={test} "
        f"train_manifest_sources={train_summary['sources']} "
        f"validation_sources={validation_summary['sources']} "
        f"test_sources={test_summary['sources']} "
        f"train_groups={train_summary['group_count']} "
        f"validation_groups={validation_summary['group_count']} "
        f"test_groups={test_summary['group_count']} "
        f"group_disjoint=true "
        f"test_source_chars={test_summary['chars']} "
        f"test_text_chars={len(test_text)} "
        f"validation_text_chars={len(val_text)} "
        f"train_match={str(train_match).lower()} "
        f"train_reconstruction_sha256={sha256_text(expected_train_full)} "
        f"augmentation_manifest={augmentation_manifest} "
        f"augmentation_manifest_sha256={sha256_file(augmentation_manifest)} "
        f"validation_match={str(validation_match).lower()} "
        f"test_match={str(test_match).lower()} "
        f"validation_sha256={sha256_text(val_text)} "
        f"test_sha256={sha256_text(test_text)}"
    )
    print(f"split_consistency_test_genres={test_summary['genres']}")
    print(f"split_consistency_test_titles={test_summary['titles']}")
    print(f"split_consistency_test_groups={test_summary['groups']}")


if __name__ == "__main__":
    main()
