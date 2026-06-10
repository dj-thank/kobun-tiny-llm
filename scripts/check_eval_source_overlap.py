from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from build_training_corpus import clean_training_text, manifest_path, read_manifest_rows
from split_policy import SPLIT_POLICY, split_name


SOURCE_ROLES = ("train", "validation", "test", "reference", "excluded")


@dataclass(frozen=True)
class SourceItem:
    role: str
    label: str
    kind: str
    value: str


@dataclass(frozen=True)
class EvalCandidate:
    path: Path
    line_no: int
    label: str
    kind: str
    value: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if smoke/regression eval rows copy source text, heldout prose, "
            "or waka poem/reading items."
        )
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus_manifest.jsonl"))
    parser.add_argument("--eval", nargs="+", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--min-prose-chars", type=int, default=18)
    parser.add_argument("--min-waka-chars", type=int, default=18)
    parser.add_argument("--waka-fuzzy-threshold", type=float, default=0.86)
    parser.add_argument("--allow-hits", action="store_true")
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected_hash: str, label: str) -> None:
    if not expected_hash:
        raise SystemExit(f"{label} missing sha256: {path}")
    if not path.exists():
        raise SystemExit(f"{label} missing file: {path}")
    actual = sha256_file(path)
    if actual != expected_hash:
        raise SystemExit(f"{label} hash mismatch: path={path} expected={expected_hash} actual={actual}")


def normalize_prose(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", "", text)


def normalize_waka(text: str) -> str:
    text = text.replace("－", "-").replace("ー", "")
    return re.sub(r"[\s/|,，、。・「」『』（）()［］\[\]{}<>《》\-]+", "", text)


def is_waka_eval_path(path: Path) -> bool:
    name = path.name.lower()
    return "waka" in name


def eval_row_kind(path: Path, row: dict[str, Any]) -> str:
    rule_ids = row.get("rule_ids")
    if is_waka_eval_path(path):
        return "waka"
    if isinstance(rule_ids, list) and any("waka" in str(item) for item in rule_ids):
        return "waka"
    if "expected_meter" in row:
        return "waka"
    return "prose"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line_no"] = line_no
        rows.append(row)
    return rows


def candidate_values(row: dict[str, Any]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    prompt = str(row.get("prompt", ""))
    if prompt:
        values.append(("prompt", prompt))
    if "good" in row:
        good = str(row["good"])
        values.append(("good", good))
        values.append(("prompt_good", prompt + good))
    if "bad" in row:
        bad = str(row["bad"])
        values.append(("bad", bad))
        values.append(("prompt_bad", prompt + bad))
    for key in ("text", "reading", "context", "prefix"):
        if key in row:
            values.append((key, str(row[key])))
    prefixes = row.get("prefixes")
    if isinstance(prefixes, list):
        for index, prefix in enumerate(prefixes):
            values.append((f"prefixes[{index}]", str(prefix)))
    return values


def manifest_role(row: dict[str, Any], val_ratio: float, test_ratio: float) -> str:
    role = split_name(row, val_ratio, test_ratio)
    if role not in {"train", "validation", "test", "reference", "excluded"}:
        raise SystemExit(f"unsupported manifest split role={role!r} source={row.get('source_id')!r}")
    return role


def prose_windows(text: str, min_chars: int, width: int = 180, stride: int = 90) -> list[str]:
    normalized = normalize_prose(text)
    if len(normalized) < min_chars:
        return []
    if len(normalized) <= width:
        return [normalized]
    values: set[str] = set()
    for start in range(0, max(1, len(normalized) - width + 1), stride):
        values.add(normalized[start : start + width])
    values.add(normalized[-width:])
    return sorted(values)


def source_items(manifest: Path, val_ratio: float, test_ratio: float, min_prose_chars: int) -> list[SourceItem]:
    items: list[SourceItem] = []
    for row in read_manifest_rows(manifest):
        role = manifest_role(row, val_ratio, test_ratio)
        source_id = str(row.get("source_id") or row.get("title") or "unknown_source")
        title = str(row.get("title") or source_id)
        clean_file = manifest_path(row.get("clean_file") or "")
        require_hash(clean_file, str(row.get("clean_sha256") or ""), f"{source_id} clean_file")
        clean_text = clean_training_text(clean_file.read_text(encoding="utf-8"))
        for index, window in enumerate(prose_windows(clean_text, min_prose_chars), start=1):
            items.append(SourceItem(role, f"{source_id}:{title}:clean:{index}", "prose", window))
        if row.get("style") != "waka":
            continue
        records_file = manifest_path(row.get("records_file") or "")
        readings_file = manifest_path(row.get("readings_file") or "")
        require_hash(records_file, str(row.get("records_sha256") or ""), f"{source_id} records_file")
        require_hash(readings_file, str(row.get("readings_sha256") or ""), f"{source_id} readings_file")
        for line_no, raw_line in enumerate(records_file.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            for field in ("poem", "reading"):
                value = normalize_waka(str(record.get(field) or ""))
                if value:
                    items.append(
                        SourceItem(role, f"{source_id}:{records_file.name}:{line_no}:{field}", "waka", value)
                    )
        for line_no, raw_line in enumerate(readings_file.read_text(encoding="utf-8").splitlines(), start=1):
            value = normalize_waka(raw_line)
            if value:
                items.append(SourceItem(role, f"{source_id}:{readings_file.name}:{line_no}:reading", "waka", value))
    return items


def eval_candidates(paths: list[Path]) -> list[EvalCandidate]:
    candidates: list[EvalCandidate] = []
    for path in paths:
        for row in read_jsonl(path):
            kind = eval_row_kind(path, row)
            for label, raw_value in candidate_values(row):
                if kind == "waka" or label == "reading":
                    value = normalize_waka(raw_value)
                    item_kind = "waka"
                else:
                    value = normalize_prose(raw_value)
                    item_kind = "prose"
                if value:
                    candidates.append(EvalCandidate(path, int(row["_line_no"]), label, item_kind, value))
    return candidates


def waka_variant_match(left: str, right: str, threshold: float) -> bool:
    if len(left) < 18 or len(right) < 18:
        return False
    if abs(len(left) - len(right)) > max(6, int(max(len(left), len(right)) * 0.25)):
        return False
    ratio = SequenceMatcher(None, left, right, autojunk=False).ratio()
    if ratio >= threshold:
        return True
    left_grams = {left[i : i + 5] for i in range(max(0, len(left) - 4))}
    right_grams = {right[i : i + 5] for i in range(max(0, len(right) - 4))}
    if not left_grams or not right_grams:
        return False
    jaccard = len(left_grams & right_grams) / len(left_grams | right_grams)
    return jaccard >= 0.62


def detect_hits(
    candidates: list[EvalCandidate],
    items: list[SourceItem],
    min_prose_chars: int,
    min_waka_chars: int,
    waka_fuzzy_threshold: float,
) -> tuple[list[str], dict[str, int]]:
    hits: list[str] = []
    counts = {"prose_hits": 0, "waka_exact_hits": 0, "waka_variant_hits": 0}
    prose_items = [item for item in items if item.kind == "prose"]
    waka_items = [item for item in items if item.kind == "waka"]
    for candidate in candidates:
        if candidate.kind == "prose":
            if len(candidate.value) < min_prose_chars:
                continue
            for item in prose_items:
                if candidate.value in item.value or item.value in candidate.value:
                    counts["prose_hits"] += 1
                    hits.append(format_hit(candidate, item, "prose_exact"))
                    break
            continue
        if len(candidate.value) < min_waka_chars:
            continue
        exact_hit = False
        for item in waka_items:
            if candidate.value in item.value or item.value in candidate.value:
                counts["waka_exact_hits"] += 1
                hits.append(format_hit(candidate, item, "waka_exact"))
                exact_hit = True
                break
        if exact_hit:
            continue
        for item in waka_items:
            if waka_variant_match(candidate.value, item.value, waka_fuzzy_threshold):
                counts["waka_variant_hits"] += 1
                hits.append(format_hit(candidate, item, "waka_variant"))
                break
    return hits, counts


def format_hit(candidate: EvalCandidate, item: SourceItem, kind: str) -> str:
    return (
        f"{kind} eval={candidate.path}:{candidate.line_no}:{candidate.label} "
        f"eval_sha256={sha256_text(candidate.value)[:16]} "
        f"source_role={item.role} source={item.label} source_sha256={sha256_text(item.value)[:16]}"
    )


def main() -> None:
    args = parse_args()
    manifest_sha = sha256_file(args.manifest)
    items = source_items(args.manifest, args.val_ratio, args.test_ratio, args.min_prose_chars)
    candidates = eval_candidates(args.eval)
    source_items_by_role = {role: 0 for role in SOURCE_ROLES}
    for item in items:
        source_items_by_role[item.role] = source_items_by_role.get(item.role, 0) + 1
    source_roles_checked = sorted(role for role, count in source_items_by_role.items() if count > 0)
    hits, counts = detect_hits(
        candidates,
        items,
        args.min_prose_chars,
        args.min_waka_chars,
        args.waka_fuzzy_threshold,
    )
    print("eval_source_overlap_eval=" + json.dumps([str(path) for path in args.eval], ensure_ascii=False))
    print(
        "eval_source_overlap_checked="
        f"{len(candidates)} "
        f"source_items={len(items)} "
        "source_roles_checked="
        f"{json.dumps(source_roles_checked, ensure_ascii=False, separators=(',', ':'))} "
        "source_items_by_role="
        f"{json.dumps(source_items_by_role, ensure_ascii=False, separators=(',', ':'))} "
        f"split_policy={SPLIT_POLICY} "
        f"val_ratio={args.val_ratio} test_ratio={args.test_ratio} "
        f"prose_hits={counts['prose_hits']} "
        f"waka_exact_hits={counts['waka_exact_hits']} "
        f"waka_variant_hits={counts['waka_variant_hits']} "
        f"hits={len(hits)} "
        f"manifest={args.manifest} manifest_sha256={manifest_sha}"
    )
    for hit in hits:
        print("EVAL_SOURCE_OVERLAP " + hit)
    if hits and not args.allow_hits:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
