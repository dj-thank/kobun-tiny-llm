from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kobun_autonomy.types import EvaluationBoard
from old_japanese_run_intel import (
    active_lock_health,
    classify_run,
    colab_cuda_lease_health,
    discover_run_ids,
    public_board_row,
    repo_root,
    select_next_action,
    startup_mutex_health,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the old-japanese-0.1B evaluation board.")
    parser.add_argument("--out-json", type=Path, default=Path("logs/evaluation_board.json"))
    parser.add_argument("--out-md", type=Path, default=Path("logs/evaluation_boards_md/EVALUATION_BOARD.md"))
    return parser.parse_args()


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else ""
    return str(value)


def render_md(payload: EvaluationBoard) -> str:
    rows = payload["runs"]
    evidence_state = payload.get("release_evidence_state", {})
    lines = [
        "# Evaluation Board",
        "",
        "This board is generated from local logs, checkpoints, and eval JSON. It is internal",
        "project evidence only; it is not a Hugging Face package and contains no raw corpus text.",
        "",
        f"Generated UTC: `{payload['generated_at_utc']}`",
        f"Governance phase: `{payload.get('governance_phase', '')}`",
        f"Next action: `{payload['next_action']['action']}`",
        f"Current release-evidence state: `{evidence_state.get('state', '')}`",
        "",
        "| run_id | status | best | latest | test_nll | release_status | score | governance | next_action | blockers | warnings |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        best = f"{fmt(row.get('best_step'))} / {fmt(row.get('best_val_loss'))}"
        latest = f"{fmt(row.get('latest_step'))} / {fmt(row.get('latest_val_loss'))}"
        governance = (
            f"source={fmt(row.get('source_quality_average'))}; "
            f"gen_diag={fmt(row.get('generation_diagnostic_plan_present'))}; "
            f"diag_metric={fmt(row.get('generation_diagnostic_release_metric'))}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row.get("run_id")),
                    fmt(row.get("status")),
                    best,
                    latest,
                    fmt(row.get("test_lm_token_nll")),
                    fmt(row.get("release_status")),
                    fmt(row.get("overall_score")),
                    governance,
                    fmt(row.get("next_action")),
                    fmt(row.get("hard_blockers")),
                    fmt(row.get("soft_warnings")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- `upload_ready_not_exported` means the evidence is ready for a later manual export decision.",
            "- HF export, package creation, and upload are never performed by this board.",
            "- `non_release_artifact` runs are retained for runtime evidence only.",
            "- `internal_failtest_evidence` rows are negative release-gate tests, not trainable or publishable runs.",
            "- Validation loss is used for early stopping; independent test LM loss is the release metric.",
            "- Smoke/regression gates include grammar, morphology, waka, leakage, and contamination checks.",
            "- When the governance phase is `pre_run_readiness`, missing checkpoint-bound",
            "  `test_lm_token_nll` is expected because no current run is a release candidate.",
            "  The correct next step is a fresh supervised DirectML or Colab CUDA run after",
            "  local gates and zero-base reviews pass; post-run evidence is required only for that exact new",
            "  best checkpoint.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = repo_root()
    run_ids = discover_run_ids(root)
    scored = [public_board_row(classify_run(root, run_id)) for run_id in run_ids]
    scored.sort(
        key=lambda row: (
            row.get("upload_ready") is True,
            row.get("release_status") == "training_active_or_unverified",
            row.get("overall_score") or 0,
            row.get("latest_step") or 0,
        ),
        reverse=True,
    )
    payload = {
        "generated_at_utc": utc_now(),
        "project": "old-japanese-0.1B",
        "hf_export_policy": "manual_only_forbidden_until_explicit_user_request",
        "runs": scored,
        "active_lock_health": active_lock_health(root),
        "colab_cuda_lease_health": colab_cuda_lease_health(root),
        "startup_mutex_health": startup_mutex_health(root),
    }
    payload["global_blockers"] = list(payload["active_lock_health"].get("hard_blockers") or []) + list(
        payload["startup_mutex_health"].get("hard_blockers") or []
    ) + list(
        payload["colab_cuda_lease_health"].get("hard_blockers") or []
    )
    payload["next_action"] = select_next_action(payload)
    if payload["next_action"].get("action") == "prepare_next_fresh_run_after_static_gate_and_zero_base_reviews":
        payload["governance_phase"] = "pre_run_readiness"
        payload["release_evidence_state"] = {
            "state": "no_current_release_candidate_expected",
            "meaning": (
                "Existing runs are non-release/internal evidence. Checkpoint-bound test metrics "
                "must be produced only after a fresh supervised DirectML or Colab CUDA run completes."
            ),
            "post_run_required": [
                "exact best checkpoint",
                "checkpoint-bound test_lm_token_nll",
                "source/provenance/tokenizer/snapshot hashes",
                "smoke/regression metrics",
                "eval contamination hits=0",
                "split leakage leaks=0",
                "waka leaks=0",
            ],
        }
    else:
        payload["governance_phase"] = "run_monitoring_or_post_run_evidence"
        payload["release_evidence_state"] = {
            "state": "see_run_rows",
            "meaning": "Use each run row and exact eval JSON to determine post-run evidence status.",
        }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(f"evaluation_board_json={args.out_json}")
    print(f"evaluation_board_md={args.out_md}")
    print("next_action=" + json.dumps(payload["next_action"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
