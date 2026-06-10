from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from export_hf_release import sanitize_eval_payload
from old_japanese_run_intel import classify_run, repo_root, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a sanitized upload-ready evidence pack for a passed run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--allow-not-ready", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rule_ssot_summary(root: Path) -> dict[str, Any]:
    rule_dir = root / "data" / "rules"
    if not rule_dir.exists():
        return {"present": False, "files": []}
    return {
        "present": True,
        "files": [
            {"basename": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(rule_dir.glob("*.json"))
        ],
    }


def source_quality_summary(root: Path) -> dict[str, Any]:
    payload = load_json_if_exists(root / "logs" / "source_quality_board.json")
    if not payload:
        return {"present": False}
    return {
        "present": True,
        "included_rows": payload.get("included_rows"),
        "average_included_score": payload.get("average_included_score"),
        "hard_blocker_rows": payload.get("hard_blocker_rows"),
        "included_by_source_kind": payload.get("included_by_source_kind", {}),
        "included_by_split_role": payload.get("included_by_split_role", {}),
    }


def generation_diagnostic_summary(root: Path, run_id: str) -> dict[str, Any]:
    policy_path = root / "data" / "rules" / "generation_diagnostic_policy.json"
    plan_path = root / "logs" / "generation_diagnostics" / f"{run_id}_generation_diagnostic_plan.json"
    if not policy_path.exists():
        return {"policy_present": False, "plan_present": plan_path.exists()}
    policy = load_json_if_exists(policy_path)
    return {
        "policy_present": True,
        "policy_basename": policy_path.name,
        "policy_sha256": sha256_file(policy_path),
        "release_metric": bool(policy.get("release_metric")),
        "metric_role": policy.get("metric_role"),
        "dimensions": len(policy.get("dimensions", [])),
        "probes": len(policy.get("probes", [])),
        "llm_training_text_allowed": False,
        "plan_present": plan_path.exists(),
        "plan_basename": plan_path.name if plan_path.exists() else "",
        "plan_sha256": sha256_file(plan_path) if plan_path.exists() else "",
    }


def scrub_eval(eval_payload: dict[str, Any]) -> dict[str, Any]:
    keep = {
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
        "eval_source_overlap",
        "tokenizer_vocab_scope",
        "split_consistency",
        "test_lm",
        "source_record_audits",
        "public_manifest_audit",
        "eval_files",
        "duplicate_metrics",
        "hf_export",
    }
    return sanitize_eval_payload({key: eval_payload.get(key) for key in sorted(keep) if key in eval_payload})


def render_report(evidence: dict[str, Any]) -> str:
    run = evidence["run"]
    title = (
        f"# Upload-Ready Report - {run['run_id']}"
        if run.get("upload_ready")
        else f"# Not Upload-Ready Diagnostic - {run['run_id']}"
    )
    lines = [
        title,
        "",
        "HF export, package creation, and upload were not performed.",
        "",
        f"Generated UTC: `{evidence['generated_at_utc']}`",
        f"Release status: `{run['release_status']}`",
        f"Upload ready: `{run['upload_ready']}`",
        f"Not release evidence: `{evidence.get('not_release_evidence', False)}`",
        f"Overall score: `{run['overall_score']}`",
        "",
        "## Checkpoint",
        "",
        f"- checkpoint: `{run.get('checkpoint', '')}`",
        f"- checkpoint_sha256: `{run.get('checkpoint_sha256', '')}`",
        f"- best_step: `{run.get('best_step', '')}`",
        f"- best_val_loss: `{run.get('best_val_loss', '')}`",
        f"- params: `{run.get('params', '')}`",
        f"- vocab_size: `{run.get('vocab_size', '')}`",
        "",
        "## Model Metric And Smoke Gates",
        "",
        "Model-facing release metric:",
        "",
        f"- test_lm_token_nll: `{run.get('test_lm_token_nll', '')}`",
        "",
        "Smoke/static gates, not standalone competence evidence:",
        "",
        f"- grammar_score: `{run.get('grammar_score', '')}`",
        f"- waka_score: `{run.get('waka_score', '')}`",
        f"- morphology_score: `{run.get('morphology_score', '')}`",
        f"- eval_contamination_hits: `{run.get('eval_contamination_hits', '')}`",
        f"- eval_source_overlap_hits: `{run.get('eval_source_overlap_hits', '')}`",
        f"- split_leaks: `{run.get('split_leaks', '')}`",
        f"- waka_leaks: `{run.get('waka_leaks', '')}`",
        f"- tokenizer_leakage: `{run.get('tokenizer_leakage', '')}`",
        "",
        "## Source And Rule Governance",
        "",
        f"- source_quality_present: `{evidence.get('source_quality', {}).get('present', False)}`",
        f"- source_quality_average: `{evidence.get('source_quality', {}).get('average_included_score', '')}`",
        f"- source_quality_hard_blocker_rows: `{evidence.get('source_quality', {}).get('hard_blocker_rows', '')}`",
        f"- rule_ssot_present: `{evidence.get('rule_ssot', {}).get('present', False)}`",
        f"- rule_ssot_files: `{len(evidence.get('rule_ssot', {}).get('files', []))}`",
        f"- generation_diagnostic_policy_present: `{evidence.get('generation_diagnostics', {}).get('policy_present', False)}`",
        f"- generation_diagnostic_policy_release_metric: `{evidence.get('generation_diagnostics', {}).get('release_metric', '')}`",
        f"- generation_diagnostic_plan_present: `{evidence.get('generation_diagnostics', {}).get('plan_present', False)}`",
        "",
        "## Blockers And Warnings",
        "",
        f"- hard_blockers: `{', '.join(run.get('hard_blockers') or [])}`",
        f"- soft_warnings: `{', '.join(run.get('soft_warnings') or [])}`",
        "",
        "## Public Artifact Policy",
        "",
        "This report intentionally excludes raw, clean, train, validation, test text, logs,",
        "optimizer state, caches, run snapshots, assistant state, and secrets.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = repo_root()
    run = classify_run(root, args.run_id)
    if not run.get("upload_ready") and not args.allow_not_ready:
        raise SystemExit(
            f"run is not upload-ready: run_id={args.run_id} status={run.get('release_status')} "
            f"blockers={run.get('hard_blockers')}"
        )
    eval_path = root / "logs" / f"eval_results_{args.run_id}.json"
    eval_payload = json.loads(eval_path.read_text(encoding="utf-8-sig")) if eval_path.exists() else {}
    evidence = {
        "generated_at_utc": utc_now(),
        "hf_export_performed": False,
        "not_release_evidence": bool(not run.get("upload_ready")),
        "public_artifact_policy": "no raw/clean/train/validation/test/logs/optimizer/caches/snapshots/codex/secrets",
        "source_quality": source_quality_summary(root),
        "rule_ssot": rule_ssot_summary(root),
        "generation_diagnostics": generation_diagnostic_summary(root, args.run_id),
        "llm_usage_policy": {
            "allowed": [
                "zero-base review",
                "failure classification",
                "generation diagnostic critique",
                "next-action recommendation",
                "evidence clarity review",
            ],
            "forbidden": [
                "creating training corpus text",
                "substituting for checkpoint-bound metrics",
                "using hidden eval answers as training signal",
                "HF export/package/upload without explicit user request",
            ],
        },
        "run": {
            key: value
            for key, value in run.items()
            if key
            in {
                "run_id",
                "status",
                "backend",
                "params",
                "vocab_size",
                "best_step",
                "best_val_loss",
                "test_lm_token_nll",
                "grammar_score",
                "waka_score",
                "morphology_score",
                "eval_contamination_hits",
                "eval_source_overlap_hits",
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
            }
        },
        "eval_evidence": scrub_eval(eval_payload) if eval_payload else {},
    }
    if run.get("upload_ready"):
        out_json = root / "logs" / "upload_ready_evidence" / f"{args.run_id}.json"
        out_md = root / "docs" / f"UPLOAD_READY_REPORT_{args.run_id}.md"
    else:
        out_json = root / "logs" / "upload_ready_evidence" / f"not_upload_ready_diagnostic_{args.run_id}.json"
        out_md = root / "docs" / f"NOT_UPLOAD_READY_DIAGNOSTIC_{args.run_id}.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_report(evidence), encoding="utf-8")
    print(f"upload_ready_evidence_json={out_json.relative_to(root)}")
    print(f"upload_ready_report_md={out_md.relative_to(root)}")


if __name__ == "__main__":
    main()
