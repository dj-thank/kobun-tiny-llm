from __future__ import annotations

import argparse
import json

from old_japanese_run_intel import classify_run, public_board_row, repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a single old-japanese-0.1B run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--full", action="store_true", help="Include internal parsed log summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = classify_run(repo_root(), args.run_id)
    if not args.full:
        payload = public_board_row(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
