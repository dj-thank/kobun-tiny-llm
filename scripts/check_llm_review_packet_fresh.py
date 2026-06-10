from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from old_japanese_run_intel import NON_RELEASE_RUNS, select_next_action
from kobun_autonomy.non_release_registry import list_non_release_run_ids


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify sanitized LLM review packets match current run intelligence.")
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_packet_for_run(board: dict[str, Any], run_id: str) -> None:
    packet_path = ROOT / "logs" / "llm_review_packets" / f"{run_id}.json"
    if not packet_path.exists():
        raise SystemExit(f"missing LLM review packet: {packet_path}")
    packet = read_json(packet_path)
    board_rows = {row.get("run_id"): row for row in board.get("runs", [])}
    row = board_rows.get(run_id)
    if not row:
        raise SystemExit(f"run id missing from evaluation board: {run_id}")
    packet_run = packet.get("run") or {}
    for key in ("release_status", "upload_ready", "next_action"):
        if packet_run.get(key) != row.get(key):
            raise SystemExit(
                f"LLM review packet stale for {run_id}: key={key} "
                f"packet={packet_run.get(key)!r} board={row.get(key)!r}"
            )
    if list(packet_run.get("hard_blockers") or []) != list(row.get("hard_blockers") or []):
        raise SystemExit(f"LLM review packet blockers do not match board for {run_id}")
    prompts = packet.get("reviewer_prompts") or {}
    prompt_text = "\n".join(str(value) for value in prompts.values())
    expected_non_release = set(NON_RELEASE_RUNS) | set(list_non_release_run_ids(ROOT))
    missing = sorted(run_id for run_id in expected_non_release if run_id not in prompt_text)
    if missing:
        raise SystemExit(f"LLM review packet prompt missing non-release runs: {missing}")


def check_all_existing_run_packets(board: dict[str, Any]) -> int:
    packet_dir = ROOT / "logs" / "llm_review_packets"
    checked = 0
    for packet_path in sorted(packet_dir.glob("old_japanese_0_1b_*.json")):
        run_id = packet_path.stem
        check_packet_for_run(board, run_id)
        checked += 1
    return checked


def main() -> None:
    args = parse_args()
    board = read_json(ROOT / "logs" / "evaluation_board.json")
    run_id = args.run_id
    expected_next_action = select_next_action(board)
    if not run_id:
        run_id = str(expected_next_action.get("run_id") or "")
    if not run_id:
        packet_path = ROOT / "logs" / "llm_review_packets" / "project.json"
        if not packet_path.exists():
            raise SystemExit(f"missing project LLM review packet: {packet_path}")
        packet = read_json(packet_path)
        if (packet.get("next_action") or {}).get("action") != expected_next_action.get("action"):
            raise SystemExit(
                "project LLM review packet stale: "
                f"packet_action={(packet.get('next_action') or {}).get('action')!r} "
                f"board_action={expected_next_action.get('action')!r}"
            )
        prompts = packet.get("reviewer_prompts") or {}
        prompt_text = "\n".join(str(value) for value in prompts.values())
        expected_non_release = set(NON_RELEASE_RUNS) | set(list_non_release_run_ids(ROOT))
        missing = sorted(run_id for run_id in expected_non_release if run_id not in prompt_text)
        if missing:
            raise SystemExit(f"project LLM review packet prompt missing non-release runs: {missing}")
        checked = check_all_existing_run_packets(board)
        print(f"llm_review_packet_fresh_ok=true run_id=project run_packets_checked={checked}")
        return
    check_packet_for_run(board, run_id)
    print(f"llm_review_packet_fresh_ok=true run_id={run_id}")


if __name__ == "__main__":
    main()
