from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_training_corpus import clean_training_text
from validate_corpus import validate_text


def include_record(record: dict[str, object]) -> bool:
    title = str(record.get("title", ""))
    source_url = str(record.get("source_url", ""))
    download_url = str(record.get("download_url", ""))
    clean_file = str(record.get("clean_file", ""))
    return not (
        title == "竹取物語"
        and "aozora.gr.jp" in source_url
        and ("48310" in source_url or "48310" in download_url or "taketori_monogatari" in clean_file)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the legacy Aozora corpus_all.txt from sources.json without excluded records.")
    parser.add_argument("--sources", type=Path, default=Path("data/aozora/sources.json"))
    parser.add_argument("--out", type=Path, default=Path("data/aozora/corpus_all.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(args.sources.read_text(encoding="utf-8"))
    parts = []
    skipped = 0
    for record in records:
        if not include_record(record):
            skipped += 1
            continue
        text = Path(str(record["clean_file"])).read_text(encoding="utf-8")
        parts.append(clean_training_text(text).strip())
    args.out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    validate_text(args.out, "training")
    print(f"wrote {args.out} included={len(parts)} skipped={skipped} bytes={args.out.stat().st_size}")


if __name__ == "__main__":
    main()
