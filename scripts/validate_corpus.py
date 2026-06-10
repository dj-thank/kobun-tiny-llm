from __future__ import annotations

import argparse
import re
from pathlib import Path


COMMON_FORBIDDEN_MARKERS = (
    "【テキスト中に現れる記号について】",
    "青空文庫作成ファイル",
    "-------------------------------------------------------",
    "入力：",
    "校正：",
    "底本：",
    "むかし、いつの頃でありましたか、竹取りの翁",
    "ほんとうの名は讃岐の造麻呂",
)


def find_common_issues(text: str) -> list[str]:
    issues = [f"forbidden marker: {marker}" for marker in COMMON_FORBIDDEN_MARKERS if marker in text]
    if re.search(r"^［＃.*］$", text, flags=re.M):
        issues.append("Aozora inline note line remains")
    return issues


def find_waka_poem_issues(text: str) -> list[str]:
    issues = find_common_issues(text)
    checks = (
        (r"^\d{5}$", "waka record id remains"),
        (r"^\[詞書\]", "waka headnote label remains"),
        (r"－", "waka reading separator remains"),
    )
    for pattern, message in checks:
        if re.search(pattern, text, flags=re.M):
            issues.append(message)
    return issues


def validate_text(path: Path, kind: str) -> None:
    text = path.read_text(encoding="utf-8")
    issues = find_waka_poem_issues(text) if kind == "waka-poems" else find_common_issues(text)
    if issues:
        issue_text = "\n".join(f"- {issue}" for issue in issues)
        raise SystemExit(f"{path} failed corpus validation:\n{issue_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail if a generated corpus contains known contamination markers.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--kind", choices=["training", "waka-poems"], default="training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in args.paths:
        validate_text(path, args.kind)
        print(f"validated {path}")


if __name__ == "__main__":
    main()
