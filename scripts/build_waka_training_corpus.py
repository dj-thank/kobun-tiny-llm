from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fetch_waka_corpus import clean_filename, parse_waka_records, write_jsonl
from split_policy import split_role_for_work, canonical_work_id
from validate_corpus import validate_text
from waka_variant_dedup import WakaVariantIndex, normalize_waka


ROLE_PRIORITY = {
    "train": 0,
    "validation": 1,
    "test": 2,
    "reference": 3,
    "excluded": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build structured waka records and poem-only training text from fetched clean files.")
    parser.add_argument("--sources", type=Path, default=Path("data/waka/sources.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/waka"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_fingerprint(row: dict[str, str]) -> tuple[str, str]:
    return normalize_waka(row.get("poem", "")), normalize_waka(row.get("reading", ""))


def split_role_for_title(title: str) -> str:
    return split_role_for_work(canonical_work_id(title), include_in_training=True)


def manifest_path(value: object) -> Path:
    return Path(str(value).replace("\\", "/"))


def item_fingerprints(row: dict[str, str]) -> list[str]:
    return [value for value in record_fingerprint(row) if value]


def main() -> None:
    args = parse_args()
    records_dir = args.out_dir / "records"
    training_dir = args.out_dir / "training"
    records_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    updated_sources: list[dict[str, object]] = []
    poem_parts: list[str] = []
    all_records: list[dict[str, str]] = []
    parsed_by_source: list[tuple[dict[str, object], list[dict[str, str]]]] = []
    for source in sources:
        title = str(source["title"])
        clean_path = manifest_path(source["clean_file"])
        clean_text = clean_path.read_text(encoding="utf-8")
        rows = parse_waka_records(clean_text, title)
        parsed_by_source.append((source, rows))

    variant_index = WakaVariantIndex()
    role_filtered: dict[int, list[dict[str, str]]] = {}
    dropped_role_overlap = 0
    dropped_role_variant_overlap = 0
    ordered_sources = sorted(
        enumerate(parsed_by_source),
        key=lambda item: (
            ROLE_PRIORITY.get(split_role_for_title(str(item[1][0]["title"])), 99),
            item[0],
        ),
    )
    for source_index, (source, rows) in ordered_sources:
        title = str(source["title"])
        role = split_role_for_title(title)
        filtered_rows: list[dict[str, str]] = []
        for row in rows:
            fingerprints = item_fingerprints(row)
            cross_role_matches = [
                variant_index.find_cross_role(role, value)
                for value in fingerprints
            ]
            cross_role_matches = [match for match in cross_role_matches if match is not None]
            if cross_role_matches:
                if any(match.kind == "variant" for match in cross_role_matches):
                    dropped_role_variant_overlap += 1
                else:
                    dropped_role_overlap += 1
                continue
            for value in fingerprints:
                variant_index.add(role, f"{title}:{source_index}", value)
            filtered_rows.append(row)
        role_filtered[source_index] = filtered_rows

    for source_index, (source, rows) in enumerate(parsed_by_source):
        rows = role_filtered[source_index]
        title = str(source["title"])
        stem = clean_filename(title)
        records_path = records_dir / f"{stem}.jsonl"
        training_path = training_dir / f"{stem}.txt"
        readings_path = training_dir / f"{stem}_readings.txt"
        poems = "\n".join(row["poem"] for row in rows)
        readings = "\n".join(row["reading"] for row in rows if row["reading"])
        write_jsonl(records_path, rows)
        training_path.write_text(poems + ("\n" if poems else ""), encoding="utf-8")
        readings_path.write_text(readings + ("\n" if readings else ""), encoding="utf-8")
        source["records_file"] = str(records_path)
        source["training_file"] = str(training_path)
        source["readings_file"] = str(readings_path)
        source["records_sha256"] = sha256_file(records_path)
        source["training_sha256"] = sha256_file(training_path)
        source["readings_sha256"] = sha256_file(readings_path)
        source["license_note"] = (
            "Japanese Wikisource; verify page-specific notices. "
            "User contributions are generally under CC BY-SA 4.0 and GFDL."
        )
        updated_sources.append(source)
        if poems:
            poem_parts.append(poems)
        all_records.extend(rows)

    corpus_path = args.out_dir / "waka_corpus_all.txt"
    records_all_path = args.out_dir / "waka_records_all.jsonl"
    corpus_path.write_text("\n\n".join(poem_parts) + "\n", encoding="utf-8")
    write_jsonl(records_all_path, all_records)
    args.sources.write_text(json.dumps(updated_sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validate_text(corpus_path, "waka-poems")
    print(f"wrote {corpus_path} poems={len(all_records)} bytes={corpus_path.stat().st_size}")
    print(f"wrote {records_all_path}")
    print(f"updated {args.sources}")
    print(f"waka_role_overlap_filtered={dropped_role_overlap}")
    print(f"waka_role_variant_overlap_filtered={dropped_role_variant_overlap}")


if __name__ == "__main__":
    main()
