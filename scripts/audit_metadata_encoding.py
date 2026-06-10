from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PATHS = [
    Path("data/aozora/sources.json"),
    Path("data/waka/sources.json"),
    Path("data/corpus_manifest.jsonl"),
    Path("logs/public_manifest_summary.json"),
    Path("logs/source_quality_board.json"),
]

SUSPICIOUS_MARKERS = (
    "\ufffd",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail if source metadata contains replacement-character mojibake or duplicate source URLs."
    )
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_PATHS)
    return parser.parse_args()


def load_records(path: Path) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


def walk_strings(value: Any, prefix: str = "$") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(prefix, value)]
    if isinstance(value, list):
        found: list[tuple[str, str]] = []
        for index, item in enumerate(value):
            found.extend(walk_strings(item, f"{prefix}[{index}]"))
        return found
    if isinstance(value, dict):
        found = []
        for key, item in value.items():
            found.extend(walk_strings(item, f"{prefix}.{key}"))
        return found
    return []


def main() -> None:
    args = parse_args()
    issues: list[str] = []
    checked_paths = 0
    checked_strings = 0
    duplicate_url_count = 0

    for path in args.paths:
        if not path.exists():
            continue
        checked_paths += 1
        records = load_records(path)
        for record_index, record in enumerate(records):
            for field_path, text in walk_strings(record, f"{path}:{record_index}"):
                checked_strings += 1
                if any(marker in text for marker in SUSPICIOUS_MARKERS):
                    issues.append(f"{field_path}: contains Unicode replacement character")

        if path.name == "sources.json":
            by_url: dict[str, list[str]] = defaultdict(list)
            for record in records:
                if not isinstance(record, dict):
                    continue
                url = str(record.get("source_url") or "")
                if not url:
                    continue
                by_url[url].append(str(record.get("title") or ""))
            for url, titles in sorted(by_url.items()):
                if len(titles) > 1:
                    duplicate_url_count += 1
                    issues.append(f"{path}: duplicate source_url={url} titles={titles}")

    print(
        "metadata_encoding_audit "
        f"paths={checked_paths} strings={checked_strings} "
        f"duplicate_source_urls={duplicate_url_count} issues={len(issues)}"
    )
    for issue in issues[:50]:
        print("ISSUE " + issue)
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
