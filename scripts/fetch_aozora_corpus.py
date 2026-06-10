from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from validate_corpus import validate_text


DEFAULT_CARDS = [
    "https://www.aozora.gr.jp/cards/001072/card48310.html",  # 竹取物語
    "https://www.aozora.gr.jp/cards/000196/card975.html",  # 方丈記
    "https://www.aozora.gr.jp/cards/000155/card832.html",  # 土佐日記
]

GENJI_CHAPTERS = [
    "桐壺",
    "帚木",
    "空蝉",
    "夕顔",
    "若紫",
    "末摘花",
    "紅葉賀",
    "花宴",
    "葵",
    "賢木",
    "花散里",
    "須磨",
    "明石",
    "澪標",
    "蓬生",
    "関屋",
    "絵合",
    "松風",
    "薄雲",
    "朝顔",
    "乙女",
    "玉鬘",
    "初音",
    "胡蝶",
    "蛍",
    "常夏",
    "篝火",
    "野分",
    "行幸",
    "藤袴",
    "真木柱",
    "梅枝",
    "藤裏葉",
    "若菜上",
    "若菜下",
    "柏木",
    "横笛",
    "鈴虫",
    "夕霧",
    "御法",
    "幻",
    "匂兵部卿",
    "紅梅",
    "竹河",
    "橋姫",
    "椎本",
    "総角",
    "早蕨",
    "宿木",
    "東屋",
    "浮舟",
    "蜻蛉",
    "手習",
    "夢浮橋",
]


@dataclass
class SourceRecord:
    title: str
    source_url: str
    download_url: str
    source_revision: str
    source_revision_timestamp: str
    retrieved_at_utc: str
    source_payload_sha256: str
    download_payload_sha256: str
    raw_file: str
    clean_file: str
    clean_sha256: str
    characters: int
    license_note: str


def fetch_bytes(url: str, retries: int = 4, backoff: float = 8.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "kobun-tiny-llm-corpus/0.1"})
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


def decode_aozora(payload: bytes) -> str:
    for encoding in ("cp932", "shift_jis", "utf-8"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("cp932", errors="replace")


def find_title(card_html: str) -> str:
    title_tag = re.search(r"<title>(.*?)</title>", card_html, re.S | re.I)
    if title_tag:
        text = re.sub(r"<[^>]+>", "", title_tag.group(1)).strip()
        return clean_filename(text.replace("図書カード：", ""))
    match = re.search(r"作品名：\s*</?[^>]*>*\s*([^<\n]+)", card_html)
    if match:
        return clean_filename(match.group(1).strip())
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", card_html, re.S)
    if h1:
        text = re.sub(r"<[^>]+>", "", h1.group(1)).strip()
        return clean_filename(text.replace("図書カード：", ""))
    return "aozora_work"


def find_zip_url(card_url: str, card_html: str) -> str:
    candidates = re.findall(r'href="([^"]+\.zip)"', card_html, re.I)
    if not candidates:
        raise ValueError(f"No zip text link found in card: {card_url}")
    ruby_or_text = [c for c in candidates if "_ruby_" in c or "_txt_" in c]
    href = ruby_or_text[0] if ruby_or_text else candidates[0]
    return urllib.parse.urljoin(card_url, href)


def clean_filename(value: str) -> str:
    value = re.sub(r'[\\/:*?"<>|]', "_", value)
    value = re.sub(r"\s+", "_", value)
    return value.strip("_") or "aozora_work"


def extract_first_text(zip_payload: bytes) -> tuple[str, bytes]:
    tmp = Path("_aozora_tmp.zip")
    tmp.write_bytes(zip_payload)
    try:
        with zipfile.ZipFile(tmp) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".txt")]
            if not names:
                raise ValueError("No .txt file found in zip")
            name = names[0]
            return name, archive.read(name)
    finally:
        tmp.unlink(missing_ok=True)


def clean_aozora_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-{20,}\n.*?(-{20,}\n)", "", text, flags=re.S)
    text = re.sub(r"【テキスト中に現れる記号について】.*?(?:-------------------------------------------------------|\Z)", "", text, flags=re.S)
    text = re.sub(r"《[^》]*》", "", text)
    text = text.replace("｜", "")
    text = re.sub(r"［＃[^］]*］", "", text)
    text = re.sub(r"※［＃[^］]*］", "", text)
    text = re.sub(r"底本：.*\Z", "", text, flags=re.S)
    text = re.sub(r"入力：.*\Z", "", text, flags=re.S)
    text = re.sub(r"校正：.*\Z", "", text, flags=re.S)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def fetch_card(card_url: str, out_dir: Path) -> SourceRecord:
    retrieved_at_utc = utc_now_iso()
    card_payload = fetch_bytes(card_url)
    card_html = decode_aozora(card_payload)
    title = find_title(card_html)
    zip_url = find_zip_url(card_url, card_html)
    zip_payload = fetch_bytes(zip_url)
    member_name, raw_payload = extract_first_text(zip_payload)
    raw_text = decode_aozora(raw_payload)
    clean_text = clean_aozora_text(raw_text)

    raw_dir = out_dir / "raw"
    clean_dir = out_dir / "clean"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    suffix = clean_filename(Path(member_name).stem)
    raw_path = raw_dir / f"{title}_{suffix}.txt"
    clean_path = clean_dir / f"{title}_{suffix}.txt"
    raw_path.write_text(raw_text, encoding="utf-8")
    clean_path.write_text(clean_text, encoding="utf-8")
    return SourceRecord(
        title=title,
        source_url=card_url,
        download_url=zip_url,
        source_revision="",
        source_revision_timestamp="",
        retrieved_at_utc=retrieved_at_utc,
        source_payload_sha256=hashlib.sha256(card_payload).hexdigest(),
        download_payload_sha256=hashlib.sha256(zip_payload).hexdigest(),
        raw_file=str(raw_path),
        clean_file=str(clean_path),
        clean_sha256=hashlib.sha256(clean_text.encode("utf-8")).hexdigest(),
        characters=len(clean_text),
        license_note="Aozora Bunko work card; check card and Aozora handling rules for reuse.",
    )


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
        "source_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "download_payload_sha256": "",
    }
    return str(page.get("extract", "")).strip() + "\n", metadata


def fetch_genji_wikisource(out_dir: Path, sleep: float) -> list[SourceRecord]:
    raw_dir = out_dir / "raw"
    clean_dir = out_dir / "clean"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    records: list[SourceRecord] = []
    for index, chapter in enumerate(GENJI_CHAPTERS, start=1):
        title = f"源氏物語/{chapter}"
        print(f"[genji {index}/{len(GENJI_CHAPTERS)}] {title}")
        text, source_metadata = mediawiki_extract(title)
        text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
        filename = f"源氏物語_{index:02d}_{clean_filename(chapter)}.txt"
        raw_path = raw_dir / filename
        clean_path = clean_dir / filename
        raw_path.write_text(text, encoding="utf-8")
        clean_path.write_text(text, encoding="utf-8")
        page_url = "https://ja.wikisource.org/wiki/" + urllib.parse.quote(title)
        records.append(
            SourceRecord(
                title=f"源氏物語 {chapter}",
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
                license_note="Japanese Wikisource; CC BY-SA text, based on 源氏物語 (渋谷栄一校訂). Attribution/share-alike required for redistribution.",
            )
        )
        time.sleep(sleep)
    return records


def include_in_training(record: SourceRecord) -> bool:
    return not (
        record.title == "竹取物語"
        and "aozora.gr.jp" in record.source_url
        and ("48310" in record.source_url or "48310" in record.download_url or "taketori_monogatari" in record.clean_file)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch copyable classical Japanese texts from Aozora Bunko.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/aozora"))
    parser.add_argument("--cards", nargs="*", default=DEFAULT_CARDS)
    parser.add_argument("--no-genji", action="store_true", help="Do not fetch Genji from Japanese Wikisource.")
    parser.add_argument("--sleep", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for index, card_url in enumerate(args.cards, start=1):
        print(f"[{index}/{len(args.cards)}] {card_url}")
        record = fetch_card(card_url, args.out_dir)
        records.append(record)
        print(f"  {record.title}: {record.characters} chars")
        time.sleep(args.sleep)
    if not args.no_genji:
        records.extend(fetch_genji_wikisource(args.out_dir, args.sleep))

    corpus = "\n\n".join(Path(record.clean_file).read_text(encoding="utf-8") for record in records if include_in_training(record))
    corpus_path = args.out_dir / "corpus_all.txt"
    corpus_path.write_text(corpus, encoding="utf-8")
    validate_text(corpus_path, "training")
    sources_path = args.out_dir / "sources.json"
    sources_path.write_text(
        json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {corpus_path} ({len(corpus)} chars)")
    print(f"wrote {sources_path}")


if __name__ == "__main__":
    main()
