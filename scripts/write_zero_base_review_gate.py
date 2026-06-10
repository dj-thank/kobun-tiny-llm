from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "zero_base_review_gate_old_japanese_0_1b.json"
SCHEMA = "old_japanese_0_1b_zero_base_review_gate_v1"
ARTIFACT_SCHEMA = "old_japanese_0_1b_zero_base_review_artifact_v1"
PASS_DECISION = "no_blockers"
REQUIRED_SCOPES = ("safety/release", "data/eval", "backend/runtime")
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a sanitized zero-base review pass gate from review artifacts.")
    parser.add_argument("--out", default=str(OUT.relative_to(ROOT)))
    parser.add_argument("--preflight-gate", default="logs/preflight_gate_old_japanese_0_1b.json")
    parser.add_argument("--safety-artifact", required=True)
    parser.add_argument("--data-artifact", required=True)
    parser.add_argument("--backend-artifact", required=True)
    return parser.parse_args()


def current_review_packet_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in REVIEW_PACKET_FILES:
        path = ROOT / rel
        if not path.exists():
            raise SystemExit(f"missing_review_packet={rel}")
        hashes[rel] = sha256_file(path)
    return hashes


def load_artifact(path_text: str, expected_scope: str, preflight_path: Path, packet_hashes: dict[str, str]) -> dict[str, Any]:
    path = (ROOT / path_text).resolve()
    path.relative_to(ROOT)
    if not path.exists():
        raise SystemExit(f"missing_review_artifact={path.relative_to(ROOT)}")
    artifact = read_json(path)
    if artifact.get("schema") != ARTIFACT_SCHEMA:
        raise SystemExit(f"review_artifact_schema_mismatch scope={expected_scope}")
    if (
        artifact.get("status") != "passed"
        or artifact.get("decision") != PASS_DECISION
        or int(artifact.get("blocking_findings_count", -1)) != 0
    ):
        raise SystemExit(f"review_artifact_not_passed scope={expected_scope}")
    if artifact.get("scope") != expected_scope:
        raise SystemExit(f"review_artifact_scope_mismatch expected={expected_scope} got={artifact.get('scope')}")
    agent_id = str(artifact.get("agent_id") or "")
    if not AGENT_ID_PATTERN.fullmatch(agent_id):
        raise SystemExit(f"review_artifact_invalid_agent_id scope={expected_scope}")
    review_witness = (artifact.get("review_model"), artifact.get("review_reasoning_effort"))
    if review_witness not in ALLOWED_REVIEW_WITNESSES:
        raise SystemExit(f"review_artifact_model_mismatch scope={expected_scope}")
    for key in ("fork_context", "patch_summary_provided", "previous_findings_provided", "full_review_text_included"):
        if artifact.get(key) is not False:
            raise SystemExit(f"review_artifact_{key}_not_false scope={expected_scope}")
    if artifact.get("hf_export") is not False:
        raise SystemExit(f"review_artifact_hf_export_not_false scope={expected_scope}")
    if artifact.get("preflight_gate") != preflight_path.relative_to(ROOT).as_posix():
        raise SystemExit(f"review_artifact_preflight_path_mismatch scope={expected_scope}")
    if artifact.get("preflight_gate_sha256") != sha256_file(preflight_path):
        raise SystemExit(f"review_artifact_preflight_hash_mismatch scope={expected_scope}")
    if artifact.get("review_packet_sha256") != packet_hashes:
        raise SystemExit(f"review_artifact_packet_hash_mismatch scope={expected_scope}")
    if not str(artifact.get("prompt_sha256") or ""):
        raise SystemExit(f"review_artifact_prompt_sha256_missing scope={expected_scope}")
    artifact["_path"] = path.relative_to(ROOT).as_posix()
    artifact["_sha256"] = sha256_file(path)
    return artifact


def main() -> None:
    args = parse_args()
    out_path = (ROOT / args.out).resolve()
    preflight_path = (ROOT / args.preflight_gate).resolve()
    out_path.relative_to(ROOT)
    preflight_path.relative_to(ROOT)
    if not preflight_path.exists():
        raise SystemExit(f"missing_preflight_gate={preflight_path.relative_to(ROOT)}")
    packet_hashes = current_review_packet_hashes()
    artifacts = [
        load_artifact(args.safety_artifact, "safety/release", preflight_path, packet_hashes),
        load_artifact(args.data_artifact, "data/eval", preflight_path, packet_hashes),
        load_artifact(args.backend_artifact, "backend/runtime", preflight_path, packet_hashes),
    ]
    agent_ids = [str(row["agent_id"]) for row in artifacts]
    if len(set(agent_ids)) != len(agent_ids):
        raise SystemExit("duplicate_review_agent_id")
    review_witnesses = {
        (str(row["review_model"]), str(row["review_reasoning_effort"]))
        for row in artifacts
    }
    if len(review_witnesses) != 1:
        raise SystemExit("mixed_review_witnesses")
    review_model, review_reasoning_effort = next(iter(review_witnesses))

    payload = {
        "schema": SCHEMA,
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hf_export": False,
        "preflight_gate": preflight_path.relative_to(ROOT).as_posix(),
        "preflight_gate_sha256": sha256_file(preflight_path),
        "review_packet_sha256": packet_hashes,
        "review_model": review_model,
        "review_reasoning_effort": review_reasoning_effort,
        "zero_base": True,
        "full_review_text_included": False,
        "review_artifacts": {
            str(row["scope"]): {"path": row["_path"], "sha256": row["_sha256"]}
            for row in artifacts
        },
        "scopes": [
            {
                "scope": row["scope"],
                "agent_id": row["agent_id"],
                "decision": PASS_DECISION,
                "blocking_findings_count": 0,
                "fork_context": False,
                "patch_summary_provided": False,
                "previous_findings_provided": False,
                "full_review_text_included": False,
                "prompt_sha256": row["prompt_sha256"],
                "artifact": row["_path"],
                "artifact_sha256": row["_sha256"],
            }
            for row in artifacts
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"zero_base_review_gate_written={out_path.relative_to(ROOT)}")
    print(f"zero_base_review_gate_schema={SCHEMA}")


if __name__ == "__main__":
    main()
