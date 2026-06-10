from __future__ import annotations

import json
from pathlib import Path

from build_llm_review_packet import build_packet, reviewer_prompts, write_packet
from old_japanese_run_intel import repo_root
from write_zero_base_review_artifact import sanitize_review_text


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = tuple(
    sorted(
        {
            str(ROOT),
            ROOT.as_posix(),
            str(ROOT).replace("\\", "/"),
        }
    )
)


def assert_sanitized(label: str, text: str) -> None:
    for needle in FORBIDDEN:
        if needle in text:
            raise SystemExit(f"{label} contains local absolute path marker: {needle}")


def main() -> None:
    prompts = reviewer_prompts(["old_japanese_0_1b_dml_synthetic"])
    joined_prompts = "\n".join(prompts.values())
    assert_sanitized("reviewer_prompts", joined_prompts)
    if "<repo_root>" not in joined_prompts:
        raise SystemExit("reviewer_prompts_missing_repo_root_placeholder")

    raw_prompt = f"Review {ROOT} from scratch."
    preview = sanitize_review_text(raw_prompt)
    assert_sanitized("prompt_preview", preview)
    if "<repo_root>" not in preview:
        raise SystemExit("prompt_preview_missing_repo_root_placeholder")

    packet = build_packet(repo_root(), {}, "", {"action": "synthetic_sanitization_test"})
    packet_text = json.dumps(packet, ensure_ascii=False)
    assert_sanitized("packet_json", packet_text)
    if (packet.get("review_packet_path_policy") or {}).get("repo_root_label") != "<repo_root>":
        raise SystemExit("packet_missing_repo_root_path_policy")

    out_dir = ROOT / "logs" / "test_llm_review_packet_sanitization"
    docs_dir = ROOT / "logs" / "test_llm_review_packet_sanitization_docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    out_json, out_md = write_packet(packet, out_dir, docs_dir, "synthetic")
    assert_sanitized("written_packet_json", out_json.read_text(encoding="utf-8-sig"))
    assert_sanitized("written_packet_md", out_md.read_text(encoding="utf-8-sig"))
    out_json.unlink(missing_ok=True)
    out_md.unlink(missing_ok=True)
    print("llm_review_packet_sanitization_ok=true")


if __name__ == "__main__":
    main()
