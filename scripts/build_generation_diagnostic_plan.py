from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "data" / "rules" / "generation_diagnostic_policy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sanitized generation-diagnostic plan. This does not run inference and is not release evidence."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    return parser.parse_args()


def validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"old_japanese_0_1b_dml_[0-9A-Za-z][0-9A-Za-z_-]{0,63}", run_id):
        raise SystemExit(f"invalid_run_id={run_id}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        f"# Generation Diagnostic Plan: {payload['run_id']}",
        "",
        "This is an internal diagnostic plan only. It is not a release metric, not a",
        "Hugging Face package, and not training data. LLM review may be used only for",
        "diagnostic critique and next-action selection after deterministic hard gates.",
        "",
        f"Generated UTC: `{payload['generated_at_utc']}`",
        f"Scope: `{payload['scope']}`",
        f"Release metric: `{payload['release_metric']}`",
        "",
        "## Policy",
        "",
        "- Do not add generated text or LLM critique to any training corpus.",
        "- Do not replace checkpoint-bound `test_lm_token_nll` or leakage/provenance gates.",
        "- Keep raw, clean, train, validation, test text, logs, snapshots, optimizer state, secrets, and assistant state out of public artifacts.",
        "- Treat results as diagnostic soft evidence unless converted into deterministic rule checks later.",
        "",
        "## Probes",
        "",
        "| probe_id | dimensions | prompt |",
        "| --- | --- | --- |",
    ]
    for probe in payload["probes"]:
        lines.append(
            f"| {probe['id']} | {', '.join(probe['dimension_ids'])} | `{probe['prompt']}` |"
        )
    lines.extend(
        [
            "",
            "## Expected Output Schema",
            "",
            "Each later diagnostic result should contain only sanitized summaries:",
            "",
            "```json",
            json.dumps(payload["expected_result_schema"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    validate_run_id(args.run_id)
    policy_path = args.policy if args.policy.is_absolute() else ROOT / args.policy
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("release_metric") is not False:
        raise SystemExit("policy_release_metric_must_be_false")

    probes = [
        {
            "id": probe["id"],
            "dimension_ids": probe["dimension_ids"],
            "prompt": probe["prompt"],
            "provenance": probe["provenance"],
            "trainable": False,
        }
        for probe in policy["probes"]
    ]
    payload = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "scope": policy["scope"],
        "release_metric": False,
        "metric_role": policy["metric_role"],
        "llm_use_policy": policy["llm_use_policy"],
        "sanitization": policy["sanitization"],
        "probes": probes,
        "expected_result_schema": {
            "run_id": args.run_id,
            "checkpoint_basename": "",
            "checkpoint_sha256": "",
            "deterministic_gates_passed": False,
            "diagnostic_only": True,
            "dimension_summaries": [
                {
                    "dimension_id": "genji_era_scope",
                    "status": "pass|warn|fail|not_run",
                    "short_reason": "sanitized summary only",
                }
            ],
            "soft_warnings": [],
            "recommended_next_action": "",
        },
    }
    out_dir = ROOT / "logs" / "generation_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.run_id}_generation_diagnostic_plan.json"
    md_path = ROOT / "docs" / f"GENERATION_DIAGNOSTIC_PLAN_{args.run_id}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_md(payload), encoding="utf-8")
    print(f"generation_diagnostic_plan_json={json_path.relative_to(ROOT)}")
    print(f"generation_diagnostic_plan_md={md_path.relative_to(ROOT)}")
    print("generation_diagnostic_plan_release_metric=false")


if __name__ == "__main__":
    main()
