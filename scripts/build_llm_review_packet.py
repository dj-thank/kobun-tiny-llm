from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from export_hf_release import sanitize_eval_payload
from old_japanese_run_intel import NON_RELEASE_RUNS, classify_run, repo_root, select_next_action, utc_now
from kobun_autonomy.non_release_registry import list_non_release_run_ids


SCOPES = {
    "safety_release": (
        "checkpoint selection, export prohibition, release evidence, secrets, "
        "artifact contents, and manual-only HF policy"
    ),
    "data_eval": (
        "manifest split, Genji-era grammar scope, tokenizer leakage, heldout validity, "
        "eval contamination, split leakage, waka leakage, and metric interpretation"
    ),
    "backend_runtime": (
        "DirectML/CUDA/HIP device handling, active-run monitoring, checkpoint loading, "
        "watcher/finalizer failure detection, and process/sentinel consistency"
    ),
    "grammar_poetics": (
        "Genji-era chuko grammar, auxiliary verbs, kakari-musubi, honorifics, waka meter, "
        "and diagnostic generation-review design"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sanitized independent-review packet for governance.")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--out-dir", type=Path, default=Path("logs/llm_review_packets"))
    parser.add_argument("--docs-dir", type=Path, default=Path("logs/independent_review_packets_md"))
    return parser.parse_args()


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def governance_eval_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "status",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_step",
        "checkpoint_best_val",
        "metadata_run_id",
        "metadata_param_count",
        "model_metrics",
        "smoke_metrics",
        "leakage",
        "eval_contamination",
        "tokenizer_vocab_scope",
        "split_consistency",
        "test_lm",
        "source_record_audits",
        "public_manifest_audit",
        "eval_files",
        "duplicate_metrics",
        "failure_reasons",
        "hf_export",
    }
    return sanitize_eval_payload({key: payload.get(key) for key in sorted(allowed) if key in payload})


def reviewer_prompts(non_release_run_ids: list[str]) -> dict[str, str]:
    prompts: dict[str, str] = {}
    non_release_text = ", ".join(sorted(set(NON_RELEASE_RUNS) | set(non_release_run_ids)))
    for scope_id, scope in SCOPES.items():
        prompts[scope_id] = (
            "Review <repo_root> from scratch.\n"
            "Do not edit files. Do not assume prior conversation context. Do not assume any other reviewer result.\n\n"
            "Project purpose: build old-japanese-0.1B, a from-scratch local Genji-era / Heian-middle "
            "classical Japanese LLM, with public/copyable corpus provenance, clean evaluation, "
            "reproducibility, and safe manual release preparation.\n\n"
            "HF export, package creation, and upload are forbidden unless the user explicitly asks later.\n"
            f"Do not treat these runs as release candidates: {non_release_text}.\n\n"
            "If the evaluation board is in pre-run readiness, absence of checkpoint-bound "
            "model metrics for old/non-release runs is expected; review whether the current "
            "gates, data, and release controls are safe to start a fresh supervised DirectML "
            "or Colab CUDA run through the repository supervisor. After a fresh run completes, "
            "checkpoint-bound metrics become mandatory.\n\n"
            f"Assigned scope: {scope}.\n\n"
            "Check for blockers in your assigned scope. List concrete file paths, reasons, "
            "and recommended fixes for every blocker you find. If you do not find blockers, "
            "write your assessment in your own words and mention any residual risks. "
            "Do not use prescribed wording."
        )
    return prompts


def build_packet(root: Path, board: dict[str, Any], run_id: str, next_action: dict[str, Any]) -> dict[str, Any]:
    run = classify_run(root, run_id) if run_id else {"run_id": "", "release_status": "unknown", "upload_ready": False}
    eval_payload = read_json_if_exists(root / "logs" / f"eval_results_{run_id}.json") if run_id else {}
    return {
        "generated_at_utc": utc_now(),
        "hf_export_performed": False,
        "governance_phase": board.get("governance_phase", "unknown") if board else "unknown",
        "release_evidence_state": board.get("release_evidence_state", {}) if board else {},
        "llm_usage_policy": {
            "allowed": [
                "independent review gate",
                "failure classification",
                "generation diagnostic critique",
                "next-action recommendation",
                "documentation quality review",
            ],
            "forbidden": [
                "creating training corpus text",
                "substituting for checkpoint-bound metrics",
                "using hidden eval answers as training signal",
                "HF export/package/upload without explicit user request",
            ],
        },
        "run": {
            key: run.get(key)
            for key in (
                "run_id",
                "status",
                "backend",
                "params",
                "vocab_size",
                "best_step",
                "best_val_loss",
                "latest_step",
                "latest_val_loss",
                "test_lm_token_nll",
                "grammar_score",
                "waka_score",
                "morphology_score",
                "eval_contamination_hits",
                "split_leaks",
                "waka_leaks",
                "tokenizer_leakage",
                "checkpoint",
                "checkpoint_sha256",
                "release_status",
                "upload_ready",
                "overall_score",
                "hard_blockers",
                "soft_warnings",
                "next_action",
            )
        },
        "eval_evidence": governance_eval_payload(eval_payload) if eval_payload else {},
        "next_action": next_action,
        "reviewer_prompts": reviewer_prompts(list_non_release_run_ids(root)),
        "review_packet_path_policy": {
            "repo_root_label": "<repo_root>",
            "actual_repo_path_shared_out_of_band": True,
            "local_absolute_paths_in_packet": False,
        },
    }


def write_packet(packet: dict[str, Any], out_dir: Path, docs_dir: Path, stem: str) -> tuple[Path, Path]:
    out_json = out_dir / f"{stem}.json"
    out_md = docs_dir / f"INDEPENDENT_REVIEW_PACKET_{stem}.md"
    out_json.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    out_md.write_text(render_md(packet), encoding="utf-8", newline="\n")
    return out_json, out_md


def render_md(packet: dict[str, Any]) -> str:
    run = packet["run"]
    next_action = packet["next_action"]
    lines = [
        f"# Independent Review Packet - {run.get('run_id') or 'project'}",
        "",
        "This is a sanitized packet for independent governance review. It intentionally excludes",
        "raw, clean, train, validation, test text, full logs, optimizer state, caches, run",
        "snapshots, assistant state, and secrets.",
        "",
        f"Generated UTC: `{packet['generated_at_utc']}`",
        f"HF export performed: `{packet['hf_export_performed']}`",
        f"Governance phase: `{packet.get('governance_phase', '')}`",
        f"Next action: `{next_action.get('action')}`",
        f"Release evidence state: `{packet.get('release_evidence_state', {}).get('state', '')}`",
        "",
        "## Run Summary",
        "",
        f"- run_id: `{run.get('run_id', '')}`",
        f"- status: `{run.get('status', '')}`",
        f"- release_status: `{run.get('release_status', '')}`",
        f"- upload_ready: `{run.get('upload_ready', '')}`",
        f"- overall_score: `{run.get('overall_score', '')}`",
        f"- best_step: `{run.get('best_step', '')}`",
        f"- best_val_loss: `{run.get('best_val_loss', '')}`",
        f"- latest_step: `{run.get('latest_step', '')}`",
        f"- latest_val_loss: `{run.get('latest_val_loss', '')}`",
        f"- test_lm_token_nll: `{run.get('test_lm_token_nll', '')}`",
        f"- blockers: `{', '.join(run.get('hard_blockers') or [])}`",
        f"- warnings: `{', '.join(run.get('soft_warnings') or [])}`",
        "",
        "## Reviewer Prompts",
        "",
    ]
    for scope_id, prompt in packet["reviewer_prompts"].items():
        lines.extend([f"### {scope_id}", "", "```text", prompt, "```", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = repo_root()
    board = read_json_if_exists(root / "logs" / "evaluation_board.json")
    run_id = args.run_id
    if not run_id and board:
        next_action = select_next_action(board)
        run_id = str(next_action.get("run_id") or "")
    else:
        next_action = select_next_action(board) if board else {"action": "update_evaluation_board_first"}
    packet = build_packet(root, board, run_id, next_action)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    stem = run_id or "project"
    out_json, out_md = write_packet(packet, args.out_dir, args.docs_dir, stem)
    refreshed = 0
    if not args.run_id and board:
        board_run_ids = {str(row.get("run_id") or "") for row in board.get("runs", [])}
        for existing in sorted(args.out_dir.glob("old_japanese_0_1b_*.json")):
            existing_run_id = existing.stem
            if existing_run_id not in board_run_ids:
                continue
            run_packet = build_packet(root, board, existing_run_id, next_action)
            write_packet(run_packet, args.out_dir, args.docs_dir, existing_run_id)
            refreshed += 1
    print(f"llm_review_packet_json={out_json}")
    print(f"llm_review_packet_md={out_md}")
    if refreshed:
        print(f"llm_review_run_packets_refreshed={refreshed}")


if __name__ == "__main__":
    main()
