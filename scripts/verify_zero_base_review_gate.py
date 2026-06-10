from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "old_japanese_0_1b_zero_base_review_gate_v1"
ARTIFACT_SCHEMA = "old_japanese_0_1b_zero_base_review_artifact_v1"
PASS_DECISION = "no_blockers"
REQUIRED_SCOPES = {"safety/release", "data/eval", "backend/runtime"}
ALLOWED_REVIEW_WITNESSES = {
    ("gpt-5.5", "xhigh"),
    ("external-reviewer", "llm_witness"),
}
AGENT_ID_PATTERN = re.compile(r"^019[0-9a-f]{1}[0-9a-f-]{31,40}$")
REVIEW_PACKET_FILES = (
    "logs/llm_review_packets/project.json",
    "logs/independent_review_packets_md/INDEPENDENT_REVIEW_PACKET_project.md",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the sanitized zero-base review pass gate.")
    parser.add_argument("--gate", default="logs/zero_base_review_gate_old_japanese_0_1b.json")
    parser.add_argument("--preflight-gate", default="logs/preflight_gate_old_japanese_0_1b.json")
    parser.add_argument("--max-age-minutes", type=float, default=120.0)
    return parser.parse_args()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("naive datetime")
    return parsed.astimezone(timezone.utc)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing_zero_base_review_gate={path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_artifact(
    path_text: str,
    expected_sha: str,
    expected_scope: str,
    preflight_path: Path,
    packet_hashes: dict[str, str],
    issues: list[str],
) -> dict[str, Any] | None:
    if not path_text:
        issues.append(f"review_artifact_missing scope={expected_scope}")
        return None
    path = (ROOT / path_text).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        issues.append(f"review_artifact_escapes_root scope={expected_scope}")
        return None
    if not path.exists():
        issues.append(f"review_artifact_path_missing scope={expected_scope} path={path_text}")
        return None
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        issues.append(f"review_artifact_sha256_mismatch scope={expected_scope}")
    try:
        artifact = load_json(path)
    except Exception as exc:
        issues.append(f"review_artifact_unreadable scope={expected_scope} error={exc}")
        return None
    if artifact.get("schema") != ARTIFACT_SCHEMA:
        issues.append(f"review_artifact_schema_mismatch scope={expected_scope}")
    if (
        artifact.get("status") != "passed"
        or artifact.get("decision") != PASS_DECISION
        or int(artifact.get("blocking_findings_count", -1)) != 0
    ):
        issues.append(f"review_artifact_not_passed scope={expected_scope}")
    if artifact.get("scope") != expected_scope:
        issues.append(f"review_artifact_scope_mismatch scope={expected_scope}")
    agent_id = str(artifact.get("agent_id") or "")
    if not AGENT_ID_PATTERN.fullmatch(agent_id):
        issues.append(f"review_artifact_invalid_agent_id scope={expected_scope}")
    artifact_witness = (artifact.get("review_model"), artifact.get("review_reasoning_effort"))
    if artifact_witness not in ALLOWED_REVIEW_WITNESSES:
        issues.append(f"review_artifact_model_mismatch scope={expected_scope}")
    for key in ("fork_context", "patch_summary_provided", "previous_findings_provided", "full_review_text_included"):
        if artifact.get(key) is not False:
            issues.append(f"review_artifact_{key}_not_false scope={expected_scope}")
    if artifact.get("hf_export") is not False:
        issues.append(f"review_artifact_hf_export_not_false scope={expected_scope}")
    try:
        preflight_rel = preflight_path.relative_to(ROOT).as_posix()
    except ValueError:
        preflight_rel = str(preflight_path)
    if artifact.get("preflight_gate") != preflight_rel:
        issues.append(f"review_artifact_preflight_path_mismatch scope={expected_scope}")
    if preflight_path.exists() and artifact.get("preflight_gate_sha256") != sha256_file(preflight_path):
        issues.append(f"review_artifact_preflight_hash_mismatch scope={expected_scope}")
    if artifact.get("review_packet_sha256") != packet_hashes:
        issues.append(f"review_artifact_packet_hash_mismatch scope={expected_scope}")
    if not str(artifact.get("prompt_sha256") or ""):
        issues.append(f"review_artifact_prompt_sha256_missing scope={expected_scope}")
    return artifact


def main() -> None:
    args = parse_args()
    gate_path = (ROOT / args.gate).resolve()
    preflight_path = (ROOT / args.preflight_gate).resolve()
    gate_path.relative_to(ROOT)
    preflight_path.relative_to(ROOT)
    gate = load_json(gate_path)
    issues: list[str] = []

    if gate.get("schema") != SCHEMA:
        issues.append(f"schema_mismatch got={gate.get('schema')}")
    if gate.get("status") != "passed":
        issues.append("status_not_passed")
    if gate.get("hf_export") is not False:
        issues.append("hf_export_not_false")
    if gate.get("zero_base") is not True:
        issues.append("zero_base_not_true")
    if gate.get("full_review_text_included") is not False:
        issues.append("full_review_text_included_not_false")
    gate_witness = (gate.get("review_model"), gate.get("review_reasoning_effort"))
    if gate_witness not in ALLOWED_REVIEW_WITNESSES:
        issues.append(f"review_model_mismatch got={gate.get('review_model')}")

    try:
        generated = parse_utc(str(gate.get("generated_at_utc") or ""))
        age_minutes = (datetime.now(timezone.utc) - generated).total_seconds() / 60.0
        if age_minutes < -1.0:
            issues.append(f"review_gate_from_future age_minutes={age_minutes:.2f}")
        if age_minutes > args.max_age_minutes:
            issues.append(f"review_gate_stale age_minutes={age_minutes:.2f} max={args.max_age_minutes:.2f}")
    except Exception as exc:
        issues.append(f"invalid_generated_at_utc error={exc}")

    if not preflight_path.exists():
        issues.append(f"missing_preflight_gate={preflight_path.relative_to(ROOT)}")
    else:
        expected_preflight = preflight_path.relative_to(ROOT).as_posix()
        if gate.get("preflight_gate") != expected_preflight:
            issues.append(f"preflight_gate_path_mismatch got={gate.get('preflight_gate')}")
        if gate.get("preflight_gate_sha256") != sha256_file(preflight_path):
            issues.append("preflight_gate_sha256_mismatch")
        try:
            preflight = json.loads(preflight_path.read_text(encoding="utf-8-sig"))
            preflight_generated = parse_utc(str(preflight.get("generated_at_utc") or ""))
            review_generated = parse_utc(str(gate.get("generated_at_utc") or ""))
            if review_generated < preflight_generated:
                issues.append("review_gate_older_than_preflight_gate")
        except Exception as exc:
            issues.append(f"preflight_review_time_check_failed error={exc}")

    review_packet_hashes = gate.get("review_packet_sha256")
    if not isinstance(review_packet_hashes, dict):
        issues.append("review_packet_sha256_missing")
        review_packet_hashes = {}
    for rel in REVIEW_PACKET_FILES:
        path = ROOT / rel
        if not path.exists():
            issues.append(f"missing_review_packet={rel}")
            continue
        if review_packet_hashes.get(rel) != sha256_file(path):
            issues.append(f"review_packet_sha256_mismatch path={rel}")

    artifact_refs = gate.get("review_artifacts")
    if not isinstance(artifact_refs, dict):
        issues.append("review_artifacts_missing")
        artifact_refs = {}
    loaded_artifacts: dict[str, dict[str, Any]] = {}
    for scope in REQUIRED_SCOPES:
        ref = artifact_refs.get(scope)
        if not isinstance(ref, dict):
            issues.append(f"review_artifact_ref_missing scope={scope}")
            continue
        artifact = validate_artifact(
            str(ref.get("path") or ""),
            str(ref.get("sha256") or ""),
            scope,
            preflight_path,
            dict(review_packet_hashes),
            issues,
        )
        if artifact is not None:
            loaded_artifacts[scope] = artifact

    scopes = gate.get("scopes")
    if not isinstance(scopes, list):
        issues.append("scopes_missing")
        scopes = []
    seen: set[str] = set()
    for row in scopes:
        if not isinstance(row, dict):
            issues.append("scope_row_not_object")
            continue
        scope = str(row.get("scope") or "")
        seen.add(scope)
        if scope not in REQUIRED_SCOPES:
            issues.append(f"unexpected_scope={scope}")
        if row.get("decision") != PASS_DECISION or int(row.get("blocking_findings_count", -1)) != 0:
            issues.append(f"scope_not_passed scope={scope}")
        agent_id = str(row.get("agent_id") or "")
        if not agent_id:
            issues.append(f"scope_missing_agent_id scope={scope}")
        elif not AGENT_ID_PATTERN.fullmatch(agent_id):
            issues.append(f"scope_invalid_agent_id scope={scope}")
        if row.get("fork_context") is not False:
            issues.append(f"scope_fork_context_not_false scope={scope}")
        if row.get("patch_summary_provided") is not False:
            issues.append(f"scope_patch_summary_not_false scope={scope}")
        if row.get("previous_findings_provided") is not False:
            issues.append(f"scope_previous_findings_not_false scope={scope}")
        if row.get("full_review_text_included") is not False:
            issues.append(f"scope_full_text_not_false scope={scope}")
        artifact = loaded_artifacts.get(scope)
        if artifact is None:
            continue
        if row.get("artifact") != artifact_refs.get(scope, {}).get("path"):
            issues.append(f"scope_artifact_path_mismatch scope={scope}")
        if row.get("artifact_sha256") != artifact_refs.get(scope, {}).get("sha256"):
            issues.append(f"scope_artifact_sha256_mismatch scope={scope}")
        if row.get("prompt_sha256") != artifact.get("prompt_sha256"):
            issues.append(f"scope_prompt_sha256_mismatch scope={scope}")
        if row.get("agent_id") != artifact.get("agent_id"):
            issues.append(f"scope_agent_id_artifact_mismatch scope={scope}")
    if seen != REQUIRED_SCOPES:
        issues.append(f"required_scopes_mismatch got={sorted(seen)}")
    agent_ids = [str(row.get("agent_id") or "") for row in scopes if isinstance(row, dict)]
    if len(set(agent_ids)) != len(agent_ids):
        issues.append("duplicate_review_agent_id")
    artifact_witnesses = {
        (artifact.get("review_model"), artifact.get("review_reasoning_effort"))
        for artifact in loaded_artifacts.values()
    }
    if artifact_witnesses and artifact_witnesses != {gate_witness}:
        issues.append("review_witness_mismatch_between_gate_and_artifacts")

    if issues:
        for issue in issues:
            print(issue)
        raise SystemExit(1)
    print(f"zero_base_review_gate_ok=true gate={gate_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
