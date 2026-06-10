from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_PASS_PHRASE = "".join(chr(code) for code in (0x554F, 0x984C, 0x306A, 0x3044, 0x3002))
FORBIDDEN_SNIPPETS = (
    LEGACY_PASS_PHRASE,
    "answer " + "exactly",
    "PASS" + "_TEXT",
    "stop" + "_phrase",
    "answer_no_blockers" + "_exactly",
    "continue_until_stop" + "_phrase",
)
CHECK_PATHS = (
    "codex_review_loop.config.toml",
    "scripts/build_llm_review_packet.py",
    "scripts/write_zero_base_review_artifact.py",
    "scripts/write_zero_base_review_gate.py",
    "scripts/verify_zero_base_review_gate.py",
)
OPTIONAL_CHECK_PATHS = (
    "logs/independent_review_packets_md/INDEPENDENT_REVIEW_PACKET_project.md",
    "logs/llm_review_packets/project.json",
)
FORBIDDEN_PATH_SNIPPETS = tuple(
    sorted(
        {
            str(ROOT),
            ROOT.as_posix(),
            str(ROOT).replace("\\", "/"),
        }
    )
)


def main() -> None:
    issues: list[str] = []
    for rel in CHECK_PATHS + OPTIONAL_CHECK_PATHS:
        path = ROOT / rel
        if not path.exists():
            if rel in CHECK_PATHS:
                issues.append(f"missing={rel}")
            continue
        text = path.read_text(encoding="utf-8-sig")
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet in text:
                issues.append(f"{rel} contains prescribed review wording marker: {snippet!r}")
        if rel.startswith(("logs/independent_review_packets_md/INDEPENDENT_REVIEW_PACKET_", "logs/llm_review_packets/")) or rel == "codex_review_loop.config.toml":
            for snippet in FORBIDDEN_PATH_SNIPPETS:
                if snippet in text:
                    issues.append(f"{rel} contains local absolute path marker: {snippet!r}")
    if issues:
        for issue in issues:
            print(issue)
        raise SystemExit(1)
    print("zero_base_review_no_prescribed_wording_ok=true")


if __name__ == "__main__":
    main()
