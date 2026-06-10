from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
from html.parser import HTMLParser
from dataclasses import asdict
from pathlib import Path

from fetch_aozora_corpus import SourceRecord, clean_filename, fetch_bytes, include_in_training, mediawiki_extract
from validate_corpus import validate_text


DEFAULT_EXACT_TITLES = [
    "宇治拾遺物語",
    "更級日記 (有朋堂文庫)",
    "紫式部日記 (渋谷栄一校訂)",
    "伊勢物語",
    "和泉式部日記",
    "蜻蛉日記 (國文大觀)",
]
DEFAULT_PREFIXES: list[str] = []
DEFAULT_INDEX_PAGES = ["枕草子_(Wikisource)"]

PARSE_FALLBACK_START_MARKERS = {
    "伊勢物語": ["伊勢物語　朱雀院塗籠御本", "むかしおとこありけり"],
    "和泉式部日記": ["夢よりもはかなき世中", "夢よりもはかなき世中を"],
    "蜻蛉日記 (國文大觀)": ["蜻蛉日記卷上", "蜻蛉日記巻上", "かくありし時過ぎて"],
}


class MediaWikiTextStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "table", "sup"}:
            self.skip_depth += 1
        if self.skip_depth == 0 and tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "table", "sup"} and self.skip_depth:
            self.skip_depth -= 1
        if self.skip_depth == 0 and tag in {"p", "div", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0:
            self.parts.append(data)


def clean_wikisource_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def api_payload(query: dict[str, str]) -> dict[str, object]:
    url = "https://ja.wikisource.org/w/api.php?" + urllib.parse.urlencode(query)
    payload = fetch_bytes(url)
    return json.loads(payload.decode("utf-8"))


def mediawiki_parse_plain_text(title: str) -> tuple[str, str]:
    query = urllib.parse.urlencode(
        {
            "action": "parse",
            "page": title,
            "prop": "text|revid",
            "format": "json",
            "formatversion": "2",
        }
    )
    payload = fetch_bytes("https://ja.wikisource.org/w/api.php?" + query)
    data = json.loads(payload.decode("utf-8"))
    if "error" in data:
        raise ValueError(f"Missing Wikisource parse page: {title}")
    html = str(data["parse"]["text"])
    stripper = MediaWikiTextStripper()
    stripper.feed(html)
    text = "".join(stripper.parts).replace("[編集]", "")
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text + "\n", hashlib.sha256(payload).hexdigest()


def trim_parse_fallback(title: str, text: str) -> str:
    for marker in PARSE_FALLBACK_START_MARKERS.get(title, []):
        index = text.find(marker)
        if index >= 0:
            return text[index:].strip() + "\n"
    if title in PARSE_FALLBACK_START_MARKERS:
        raise ValueError(f"Could not find classical text start marker for Wikisource page: {title}")
    return text


def list_pages_by_prefix(prefix: str, sleep: float) -> list[str]:
    titles: list[str] = []
    params = {
        "action": "query",
        "list": "allpages",
        "apnamespace": "0",
        "aplimit": "max",
        "apprefix": prefix,
        "format": "json",
    }
    while True:
        data = api_payload(params)
        pages = data.get("query", {}).get("allpages", [])
        for page in pages:
            title = str(page.get("title", ""))
            if title:
                titles.append(title)
        continuation = data.get("continue")
        if not isinstance(continuation, dict):
            break
        for key, value in continuation.items():
            params[key] = str(value)
        time.sleep(sleep)
    return sorted(set(titles))


def list_pages_from_index(page: str) -> list[str]:
    data = api_payload({"action": "parse", "page": page, "prop": "links", "format": "json"})
    if "parse" not in data:
        raise SystemExit(f"Could not parse index page {page}: {data.get('error')}")
    links = data["parse"].get("links", [])
    titles = [
        str(link.get("*", ""))
        for link in links
        if int(link.get("ns", -1)) == 0 and str(link.get("*", "")).startswith("枕草子 (Wikisource)/")
    ]
    return sorted(set(titles))


def fetch_title(title: str, out_dir: Path) -> SourceRecord:
    text, source_metadata = mediawiki_extract(title)
    if len(text.strip()) < 500:
        parsed_text, parsed_payload_sha256 = mediawiki_parse_plain_text(title)
        parsed_text = trim_parse_fallback(title, parsed_text)
        if len(parsed_text.strip()) > len(text.strip()):
            text = parsed_text
            source_metadata["source_payload_sha256"] = parsed_payload_sha256
    text = clean_wikisource_text(text)
    raw_dir = out_dir / "raw"
    clean_dir = out_dir / "clean"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    filename = clean_filename(title) + ".txt"
    raw_path = raw_dir / filename
    clean_path = clean_dir / filename
    raw_path.write_text(text, encoding="utf-8")
    clean_path.write_text(text, encoding="utf-8")
    page_url = "https://ja.wikisource.org/wiki/" + urllib.parse.quote(title)
    return SourceRecord(
        title=title.replace("/", " "),
        source_url=page_url,
        download_url="MediaWiki API extracts",
        source_revision=source_metadata["source_revision"],
        source_revision_timestamp=source_metadata["source_revision_timestamp"],
        retrieved_at_utc=source_metadata["retrieved_at_utc"],
        source_payload_sha256=source_metadata["source_payload_sha256"],
        download_payload_sha256=source_metadata["download_payload_sha256"],
        raw_file=str(raw_path),
        clean_file=str(clean_path),
        clean_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        characters=len(text),
        license_note="Japanese Wikisource page history and license notice; attribution/share-alike obligations apply.",
    )


def merge_records(existing: list[dict[str, object]], fetched: list[SourceRecord]) -> list[dict[str, object]]:
    merged = {
        str(record.get("source_url") or record.get("title") or index): dict(record)
        for index, record in enumerate(existing)
    }
    for record in fetched:
        merged[record.source_url] = asdict(record)
    return list(merged.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch extra same-era/reference Wikisource prose with revision metadata.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/aozora"))
    parser.add_argument("--sources", type=Path, default=Path("data/aozora/sources.json"))
    parser.add_argument("--exact-title", action="append", default=DEFAULT_EXACT_TITLES)
    parser.add_argument("--prefix", action="append", default=DEFAULT_PREFIXES)
    parser.add_argument("--index-page", action="append", default=DEFAULT_INDEX_PAGES)
    parser.add_argument("--no-default-index-pages", action="store_true")
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--target-chars", type=int, default=60_000)
    parser.add_argument("--sleep", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    titles: list[str] = []
    for title in args.exact_title:
        titles.append(title)
    for prefix in args.prefix:
        prefix_titles = list_pages_by_prefix(prefix, args.sleep)
        print(f"prefix={prefix} pages={len(prefix_titles)}")
        titles.extend(prefix_titles)
    if args.no_default_index_pages:
        args.index_page = []
    for page in args.index_page:
        index_titles = list_pages_from_index(page)
        print(f"index_page={page} pages={len(index_titles)}")
        titles.extend(index_titles)
    titles = sorted(set(titles))

    fetched: list[SourceRecord] = []
    fetched_chars = 0
    target_work_chars = 0
    for index, title in enumerate(titles, start=1):
        print(f"[extra {index}/{len(titles)}] {title}")
        try:
            record = fetch_title(title, args.out_dir)
        except ValueError as exc:
            print(f"  skipped_missing title={title} reason={exc}")
            time.sleep(args.sleep)
            continue
        if record.characters < args.min_chars:
            print(f"  skipped_short title={title} chars={record.characters}")
        else:
            fetched.append(record)
            fetched_chars += record.characters
            if title.startswith("枕草子"):
                target_work_chars += record.characters
            print(f"  kept title={title} chars={record.characters}")
        if args.target_chars > 0 and target_work_chars >= args.target_chars and title.startswith("枕草子"):
            print(f"target_chars_reached={target_work_chars}")
            break
        time.sleep(args.sleep)

    existing = json.loads(args.sources.read_text(encoding="utf-8")) if args.sources.exists() else []
    merged = merge_records(existing, fetched)
    args.sources.parent.mkdir(parents=True, exist_ok=True)
    args.sources.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    corpus = "\n\n".join(
        Path(str(record["clean_file"])).read_text(encoding="utf-8")
        for record in merged
        if Path(str(record.get("clean_file", ""))).exists()
        and include_in_training(SourceRecord(**record))
    )
    corpus_path = args.out_dir / "corpus_all.txt"
    corpus_path.write_text(corpus, encoding="utf-8")
    validate_text(corpus_path, "training")
    print(f"fetched_extra={len(fetched)} merged_sources={len(merged)} wrote={args.sources}")


if __name__ == "__main__":
    main()
