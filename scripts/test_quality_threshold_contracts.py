from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(pattern: str, text: str, label: str) -> None:
    if not re.search(pattern, text, flags=re.DOTALL):
        raise SystemExit(f"quality threshold contract missing: {label}")


def forbid(pattern: str, text: str, label: str) -> None:
    if re.search(pattern, text, flags=re.DOTALL):
        raise SystemExit(f"quality threshold contract forbidden: {label}")


def main() -> None:
    scripts = {
        "scripts/run_quality_checks.ps1": read("scripts/run_quality_checks.ps1"),
        "scripts/run_quality_checks_dml.ps1": read("scripts/run_quality_checks_dml.ps1"),
    }
    for path, text in scripts.items():
        require(
            r"eval_minimal_pairs\.py\s+--checkpoint\s+\S+\s+--device\s+\S+\s+--pairs\s+\$PrimaryPairs\s+--metric-prefix\s+primary\s+--min-cases\s+8\s+--min-accuracy\s+1\.0",
            text,
            f"{path} primary minimal-pair threshold",
        )
        require(
            r"eval_minimal_pairs\.py\s+--checkpoint\s+\S+\s+--device\s+\S+\s+--pairs\s+\$HeldoutPairs\s+--metric-prefix\s+heldout\s+--min-cases\s+12\s+--min-accuracy\s+1\.0",
            text,
            f"{path} heldout minimal-pair threshold",
        )
        forbid(
            r"--pairs\s+\$HeldoutPairs\s+--metric-prefix\s+heldout\s+--min-cases\s+8\b",
            text,
            f"{path} stale heldout threshold",
        )

    for path in ("scripts/export_hf_release.py", "scripts/check_release_package.py"):
        text = read(path)
        require(
            r'"heldout_contrastive_preference_accuracy":\s*\{"min_value":\s*1\.0,\s*"min_total":\s*12\}',
            text,
            f"{path} release heldout threshold",
        )

    print("quality_threshold_contracts_ok=true")


if __name__ == "__main__":
    main()
