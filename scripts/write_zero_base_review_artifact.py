from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "old_japanese_0_1b_zero_base_review_artifact_v1"
SCOPES = ("safety/release", "data/eval", "backend/runtime")
PASS_DECISION = "no_blockers"
DEFAULT_REVIEW_MODEL = "external-reviewer"
DEFAULT_REVIEW_REASONING_EFFORT = "llm_witness"
AGENT_ID_PATTERN = re.compile(r"^019[0-9a-f]{1}[0-9a-f-]{31,40}$")
REVIEW_PACKET_FILES = (
    "logs/llm_review_packets/project.json",
    "logs/independent_review_packets_md/INDEPENDENT_REVIEW_PACKET_project.md",
)
LOCAL_PROJECT_PARTS = ("ExampleWorkstation", "ExampleProjects", "kobun-tiny-llm")
WINDOWS_LOCAL_PATH_PATTERN = re.compile(
    r"[A-Za-z]:\\Users\\[^\\\r\n`\"']+?\\"
    + r"\\".join(re.escape(part) for part in LOCAL_PROJECT_PARTS),
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sanitize_review_text(value: str) -> str:
    text = str(value)
    root_win = str(ROOT)
    root_posix = ROOT.as_posix()
    text = text.replace(root_win, "<repo_root>")
    text = text.replace(root_posix, "<repo_root>")
    text = WINDOWS_LOCAL_PATH_PATTERN.sub("<repo_root>", text)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write one sanitized zero-base review result artifact.")
    parser.add_argument("--scope", choices=SCOPES, required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--decision", choices=(PASS_DECISION,), required=True)
    parser.add_argument("--blocking-findings-count", type=int, required=True)
    parser.add_argument("--summary", default="No blocking findings recorded; full review text intentionally omitted.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--review-model", default=DEFAULT_REVIEW_MODEL)
    parser.add_argument("--review-reasoning-effort", default=DEFAULT_REVIEW_REASONING_EFFORT)
    parser.add_argument("--preflight-gate", default="logs/preflight_gate_old_japanese_0_1b.json")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def review_packet_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in REVIEW_PACKET_FILES:
        path = ROOT / rel
        if not path.exists():
            raise SystemExit(f"missing_review_packet={rel}")
        hashes[rel] = sha256_file(path)
    return hashes


def clean_scope(scope: str) -> str:
    return scope.replace("/", "_")


def main() -> None:
    args = parse_args()
    if not AGENT_ID_PATTERN.fullmatch(args.agent_id):
        raise SystemExit(f"invalid_agent_id={args.agent_id}")
    if args.decision != PASS_DECISION or args.blocking_findings_count != 0:
        raise SystemExit(f"review_result_not_passed scope={args.scope}")
    preflight = (ROOT / args.preflight_gate).resolve()
    preflight.relative_to(ROOT)
    if not preflight.exists():
        raise SystemExit(f"missing_preflight_gate={preflight.relative_to(ROOT)}")
    out = Path(args.out) if args.out else Path("logs/zero_base_review_artifacts") / f"{clean_scope(args.scope)}_{args.agent_id}.json"
    out_path = (ROOT / out).resolve()
    out_path.relative_to(ROOT)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": args.scope,
        "agent_id": args.agent_id,
        "decision": PASS_DECISION,
        "blocking_findings_count": 0,
        "result_summary": args.summary[:240],
        "review_model": args.review_model,
        "review_reasoning_effort": args.review_reasoning_effort,
        "fork_context": False,
        "patch_summary_provided": False,
        "previous_findings_provided": False,
        "full_review_text_included": False,
        "prompt_sha256": sha256_text(args.prompt),
        "prompt_preview": sanitize_review_text(args.prompt)[:240],
        "preflight_gate": preflight.relative_to(ROOT).as_posix(),
        "preflight_gate_sha256": sha256_file(preflight),
        "review_packet_sha256": review_packet_hashes(),
        "hf_export": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"zero_base_review_artifact_written={out_path.relative_to(ROOT)}")
    print(f"zero_base_review_artifact_scope={args.scope}")


if __name__ == "__main__":
    main()
