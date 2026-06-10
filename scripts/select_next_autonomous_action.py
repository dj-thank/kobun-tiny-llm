from __future__ import annotations

import argparse
import json

from old_japanese_run_intel import load_board, repo_root, select_next_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the next safe autonomous action for old-japanese-0.1B.")
    parser.add_argument("--board", default="logs/evaluation_board.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    board = load_board(root) if args.board == "logs/evaluation_board.json" else None
    if board is None:
        path = root / args.board
        if not path.exists():
            raise SystemExit(f"missing evaluation board: {path}; run scripts/update_evaluation_board.py first")
        board = json.loads(path.read_text(encoding="utf-8-sig"))
    print(json.dumps(select_next_action(board), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
