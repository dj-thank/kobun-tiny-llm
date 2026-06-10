from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a training-safe list of classical-language surface patterns.")
    parser.add_argument("--allowlist", type=Path, default=Path("data/external_knowledge/training_allowlist.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/external_knowledge/classical_surface_patterns.txt"))
    return parser.parse_args()


def iter_items(path: Path) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        training_use = str(row.get("training_use") or "")
        if training_use != "allowed_surface_patterns_only":
            continue
        raw_items = row.get("items")
        if not isinstance(raw_items, list):
            raise SystemExit(f"allowlist line {line_number} has no items list")
        for raw_item in raw_items:
            item = str(raw_item).strip()
            if not item or "..." in item:
                continue
            if item not in seen:
                seen.add(item)
                items.append(item)
    return items


def main() -> None:
    args = parse_args()
    items = iter_items(args.allowlist)
    if not items:
        raise SystemExit(f"no allowed surface patterns found in {args.allowlist}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(items) + "\n", encoding="utf-8", newline="\n")
    print(f"external_knowledge_surface_patterns path={args.out} items={len(items)} bytes={args.out.stat().st_size}")


if __name__ == "__main__":
    main()
