from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from split_policy import (
    SPLIT_POLICY,
    canonical_work_id,
    grammar_scope,
    split_role_for_work,
)


def infer_meta(title: str) -> dict[str, str]:
    if (
        title.startswith("古今和歌集")
        or title.startswith("後撰和歌集")
        or title.startswith("拾遺和歌集")
        or title.startswith("小倉百人一首")
    ):
        return {"period": "中古", "genre": "和歌", "style": "waka"}
    if title.startswith("源氏物語"):
        return {"period": "中古", "genre": "作り物語", "style": "genji"}
    if title.startswith("枕草子"):
        return {"period": "中古", "genre": "随筆", "style": "makura"}
    if title.startswith("更級日記") or title.startswith("さらしな日記"):
        return {"period": "中古", "genre": "日記", "style": "nikki"}
    if title.startswith("紫式部日記"):
        return {"period": "中古", "genre": "日記", "style": "nikki"}
    if title.startswith("和泉式部日記"):
        return {"period": "中古", "genre": "日記", "style": "nikki"}
    if title.startswith("蜻蛉日記"):
        return {"period": "中古", "genre": "日記", "style": "nikki"}
    if title.startswith("伊勢物語"):
        return {"period": "中古", "genre": "歌物語", "style": "uta_monogatari"}
    if title.startswith("宇治拾遺物語"):
        return {"period": "中世", "genre": "説話", "style": "setsuwa"}
    if title == "竹取物語":
        return {"period": "中古", "genre": "作り物語", "style": "setsuwa"}
    if title == "土佐日記":
        return {"period": "中古", "genre": "日記", "style": "nikki"}
    if title == "方丈記":
        return {"period": "中世", "genre": "随筆", "style": "zuihitsu"}
    return {"period": "unknown", "genre": "unknown", "style": "unknown"}


def training_filter(record: dict[str, object], title: str, work_id: str) -> dict[str, object]:
    source_url = str(record.get("source_url", ""))
    download_url = str(record.get("download_url", ""))
    clean_file = str(record.get("clean_file", ""))
    if (
        title == "竹取物語"
        and "aozora.gr.jp" in source_url
        and ("48310" in source_url or "48310" in download_url or "taketori_monogatari" in clean_file)
    ):
        return {
            "include_in_training": False,
            "exclude_reason": "Aozora 48310 is a modern retelling for children, not the classical original text required for this corpus.",
        }
    if split_role_for_work(work_id, include_in_training=True) == "reference":
        return {
            "include_in_training": False,
            "exclude_reason": "Reference-only source outside the Genji-era grammar scope for this release-candidate corpus.",
        }
    if split_role_for_work(work_id, include_in_training=True) == "excluded":
        return {
            "include_in_training": False,
            "exclude_reason": "Unregistered source is excluded until split role and Genji-era scope are explicitly reviewed.",
        }
    return {"include_in_training": True, "exclude_reason": ""}


def canonical_source_key(record: dict[str, object], clean_file: Path) -> str:
    parts = [
        str(record.get("source_url", "")),
        str(record.get("download_url", "")),
        str(record.get("source_revision", "")),
        str(record.get("source_revision_timestamp", "")),
        str(record.get("source_payload_sha256", "")),
        str(record.get("download_payload_sha256", "")),
        str(record.get("api_payload_sha256", "")),
        str(record.get("records_sha256", "")),
        str(record.get("readings_sha256", "")),
        str(record.get("training_sha256", "")),
        file_sha256(clean_file),
        str(record.get("title", "")),
    ]
    return "|".join(parts)


def source_id(title: str, key: str) -> str:
    safe = title.replace(" ", "_").replace("/", "_").replace("\\", "_")
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=6).hexdigest()
    return f"{digest}_{safe}"


def file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def optional_string(record: dict[str, object], field: str) -> str:
    return str(record.get(field, "") or "")


def manifest_path(value: object) -> Path:
    return Path(str(value).replace("\\", "/"))


def manifest_path_text(value: object) -> str:
    if not str(value or ""):
        return ""
    return manifest_path(value).as_posix()


def source_kind(record: dict[str, object]) -> str:
    source_url = str(record.get("source_url", ""))
    if "wikisource.org" in source_url:
        return "wikisource"
    if "aozora.gr.jp" in source_url:
        return "aozora"
    return "unknown"


def license_name(record: dict[str, object]) -> str:
    kind = source_kind(record)
    if kind == "wikisource":
        return "CC BY-SA/GFDL page license notices"
    if kind == "aozora":
        return "Aozora Bunko work card terms"
    return "unknown"


def license_note(record: dict[str, object]) -> str:
    kind = source_kind(record)
    if kind == "wikisource":
        return "Japanese Wikisource page history and license notice; attribution and share-alike obligations apply."
    if kind == "aozora":
        return "Aozora Bunko source card handling rules apply; release artifacts do not include source text."
    return str(record.get("license_note", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a period/style corpus manifest from source records.")
    parser.add_argument("--sources", nargs="+", type=Path, default=[Path("data/aozora/sources.json"), Path("data/waka/sources.json")])
    parser.add_argument("--out", type=Path, default=Path("data/corpus_manifest.jsonl"))
    parser.add_argument("--excluded-out", type=Path, default=Path("data/excluded_sources.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for source_path in args.sources:
        if source_path.exists():
            records.extend(json.loads(source_path.read_text(encoding="utf-8")))
    rows = []
    excluded = []
    seen_files = set()
    for record in records:
        clean_file = manifest_path(record.get("training_file") or record["clean_file"])
        if str(clean_file) in seen_files:
            continue
        seen_files.add(str(clean_file))
        title = str(record["title"])
        key = canonical_source_key(record, clean_file)
        work_id = canonical_work_id(title)
        meta = infer_meta(title)
        filter_meta = training_filter(record, title, work_id)
        split_role = split_role_for_work(work_id, bool(filter_meta["include_in_training"]))
        row = {
            "source_id": source_id(title, key),
            "split_key": key,
            "split_policy": SPLIT_POLICY,
            "work_id": work_id,
            "split_group_key": work_id,
            "split_role": split_role,
            "grammar_scope": grammar_scope(work_id),
            "title": title,
            **meta,
            **filter_meta,
            "source_kind": source_kind(record),
            "license_name": license_name(record),
            "license_note": license_note(record),
            "redistribution_policy": "corpus_text_not_distributed",
            "source_url": record.get("source_url", ""),
            "download_url": record.get("download_url", ""),
            "raw_file": manifest_path_text(record.get("raw_file", "")),
            "clean_file": clean_file.as_posix(),
            "records_file": manifest_path_text(record.get("records_file", "")),
            "readings_file": manifest_path_text(record.get("readings_file", "")),
            "records_sha256": optional_string(record, "records_sha256"),
            "readings_sha256": optional_string(record, "readings_sha256"),
            "training_sha256": optional_string(record, "training_sha256"),
            "characters": record.get("characters", 0),
            "source_revision": optional_string(record, "source_revision"),
            "source_revision_timestamp": optional_string(record, "source_revision_timestamp"),
            "retrieved_at_utc": optional_string(record, "retrieved_at_utc"),
            "source_payload_sha256": optional_string(record, "source_payload_sha256"),
            "download_payload_sha256": optional_string(record, "download_payload_sha256"),
            "api_payload_sha256": optional_string(record, "api_payload_sha256"),
            "clean_sha256": file_sha256(clean_file),
        }
        rows.append(row)
        if not row["include_in_training"]:
            excluded.append(row)
    args.out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    args.excluded_out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in excluded) + ("\n" if excluded else ""),
        encoding="utf-8",
        newline="\n",
    )
    included_total = sum(int(row["characters"]) for row in rows if row["include_in_training"])
    print(f"wrote {args.out} rows={len(rows)} included_chars={included_total} excluded={len(excluded)}")


if __name__ == "__main__":
    main()
