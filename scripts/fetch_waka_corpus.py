from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


KOKIN_TITLES = [
    "古今和歌集/巻一",
    "古今和歌集/巻二",
    "古今和歌集/巻三",
    "古今和歌集/巻四",
    "古今和歌集/巻五",
    "古今和歌集/巻六",
    "古今和歌集/巻七",
    "古今和歌集/巻八",
    "古今和歌集/巻九",
    "古今和歌集/巻十",
    "古今和歌集/巻十一",
    "古今和歌集/巻十二",
    "古今和歌集/巻十三",
    "古今和歌集/巻十四",
    "古今和歌集/巻十五",
    "古今和歌集/巻十六",
    "古今和歌集/巻十七",
    "古今和歌集/巻十八",
    "古今和歌集/巻十九",
    "古今和歌集/巻二十",
]
IMPERIAL_WAKA_VOLUME_ORDINALS = [
    "一",
    "二",
    "三",
    "四",
    "五",
    "六",
    "七",
    "八",
    "九",
    "十",
    "十一",
    "十二",
    "十三",
    "十四",
    "十五",
    "十六",
    "十七",
    "十八",
    "十九",
    "二十",
]
GOSEN_TITLES = [f"後撰和歌集/巻第{volume}" for volume in IMPERIAL_WAKA_VOLUME_ORDINALS]
SHUI_TITLES = [f"拾遺和歌集/巻第{volume}" for volume in IMPERIAL_WAKA_VOLUME_ORDINALS]
DEFAULT_WIKISOURCE_TITLES = KOKIN_TITLES + GOSEN_TITLES + SHUI_TITLES


@dataclass
class WakaSourceRecord:
    title: str
    source_url: str
    download_url: str
    source_revision: str
    source_revision_timestamp: str
    retrieved_at_utc: str
    api_payload_sha256: str
    raw_file: str
    clean_file: str
    clean_sha256: str
    records_file: str
    training_file: str
    readings_file: str
    characters: int
    license_note: str


def fetch_bytes(url: str, retries: int = 4, backoff: float = 5.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "kobun-tiny-llm-waka-corpus/0.1"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt >= retries:
                raise
            wait = backoff * (attempt + 1)
            print(f"  rate limited; waiting {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def mediawiki_extract(title: str) -> tuple[str, dict[str, str]]:
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts|revisions",
            "explaintext": "1",
            "rvprop": "ids|timestamp",
            "format": "json",
            "titles": title,
        }
    )
    payload = fetch_bytes(f"https://ja.wikisource.org/w/api.php?{query}")
    data = json.loads(payload.decode("utf-8"))
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page:
        raise ValueError(f"Missing Wikisource page: {title}")
    revision = next(iter(page.get("revisions", []) or []), {})
    metadata = {
        "source_revision": str(revision.get("revid", "")),
        "source_revision_timestamp": str(revision.get("timestamp", "")),
        "retrieved_at_utc": utc_now_iso(),
        "api_payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return str(page.get("extract", "")).strip() + "\n", metadata


def clean_filename(value: str) -> str:
    value = re.sub(r'[\\/:*?"<>|]', "_", value)
    value = re.sub(r"\s+", "_", value)
    return value.strip("_") or "waka_work"


def clean_waka_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n?カテゴリ:.*", "", text)
    text = re.sub(r"\n?底本:.*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def parse_waka_records(text: str, source_title: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    section = ""
    current_number = ""
    block: list[str] = []

    def flush() -> None:
        nonlocal block, current_number
        if not current_number:
            return
        lines = [line.strip() for line in block if line.strip()]
        headnote = ""
        if lines and lines[0].startswith("[詞書]"):
            headnote = lines.pop(0).removeprefix("[詞書]").strip()
        reading = ""
        if lines and "－" in lines[-1]:
            reading = lines.pop()
        author = ""
        poem = ""
        if len(lines) >= 2:
            author = lines[0]
            poem = lines[1]
        elif lines:
            poem = lines[0]
        if poem:
            records.append(
                {
                    "source_title": source_title,
                    "section": section,
                    "number": current_number,
                    "headnote": headnote,
                    "author": author,
                    "poem": poem,
                    "reading": reading,
                }
            )
        block = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"\d{5}", line):
            flush()
            current_number = line
            block = []
            continue
        if not current_number and not section:
            section = line
            continue
        block.append(line)
    flush()
    return records


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def fetch_wikisource_title(title: str, out_dir: Path) -> WakaSourceRecord:
    raw_dir = out_dir / "raw"
    clean_dir = out_dir / "clean"
    records_dir = out_dir / "records"
    training_dir = out_dir / "training"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    text, source_metadata = mediawiki_extract(title)
    clean_text = clean_waka_text(text)
    records = parse_waka_records(clean_text, title)
    filename = f"{clean_filename(title)}.txt"
    stem = clean_filename(title)
    raw_path = raw_dir / filename
    clean_path = clean_dir / filename
    records_path = records_dir / f"{stem}.jsonl"
    training_path = training_dir / filename
    readings_path = training_dir / f"{stem}_readings.txt"
    raw_path.write_text(text, encoding="utf-8", newline="\n")
    clean_path.write_text(clean_text, encoding="utf-8", newline="\n")
    write_jsonl(records_path, records)
    poems = "\n".join(record["poem"] for record in records)
    readings = "\n".join(record["reading"] for record in records if record["reading"])
    training_path.write_text(poems + ("\n" if poems else ""), encoding="utf-8", newline="\n")
    readings_path.write_text(readings + ("\n" if readings else ""), encoding="utf-8", newline="\n")
    page_url = "https://ja.wikisource.org/wiki/" + urllib.parse.quote(title)
    return WakaSourceRecord(
        title=title,
        source_url=page_url,
        download_url="MediaWiki API extracts",
        source_revision=source_metadata["source_revision"],
        source_revision_timestamp=source_metadata["source_revision_timestamp"],
        retrieved_at_utc=source_metadata["retrieved_at_utc"],
        api_payload_sha256=source_metadata["api_payload_sha256"],
        raw_file=str(raw_path),
        clean_file=str(clean_path),
        clean_sha256=hashlib.sha256(clean_text.encode("utf-8")).hexdigest(),
        records_file=str(records_path),
        training_file=str(training_path),
        readings_file=str(readings_path),
        characters=len(clean_text),
        license_note="Japanese Wikisource; verify page-specific notices. User contributions are generally under CC BY-SA 4.0 and GFDL.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch public waka texts from Japanese Wikisource.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/waka"))
    parser.add_argument("--titles", nargs="*", default=DEFAULT_WIKISOURCE_TITLES)
    parser.add_argument("--sleep", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[WakaSourceRecord] = []
    for index, title in enumerate(args.titles, start=1):
        print(f"[{index}/{len(args.titles)}] {title}")
        record = fetch_wikisource_title(title, args.out_dir)
        records.append(record)
        print(f"  {record.title}: {record.characters} chars")
        time.sleep(args.sleep)
    sources_path = args.out_dir / "sources.json"
    sources_path.write_text(
        json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    corpus = "\n\n".join(Path(record.training_file).read_text(encoding="utf-8").strip() for record in records)
    corpus_path = args.out_dir / "waka_corpus_all.txt"
    corpus_path.write_text(corpus + "\n", encoding="utf-8")
    records_corpus = args.out_dir / "waka_records_all.jsonl"
    with records_corpus.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(Path(record.records_file).read_text(encoding="utf-8"))
    print(f"wrote {sources_path}")
    print(f"wrote {corpus_path} ({len(corpus)} chars)")
    print(f"wrote {records_corpus}")


if __name__ == "__main__":
    main()
