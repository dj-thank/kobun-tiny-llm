from __future__ import annotations

from pathlib import Path

from .non_release_registry import NonReleaseRecordError, is_non_release_recorded

NON_RELEASE_RUNS = {
    "old_japanese_0_1b_dml_20260509_193645",
    "old_japanese_0_1b_dml_20260510_015630",
    "old_japanese_0_1b_dml_20260510_120643",
    "old_japanese_0_1b_dml_20260510_140342",
    "old_japanese_0_1b_dml_20260510_190639",
    "old_japanese_0_1b_dml_20260511_004057",
    "old_japanese_0_1b_dml_20260511_152714",
    "old_japanese_0_1b_dml_20260512_004913",
    "old_japanese_0_1b_dml_20260512_011227",
}


def is_non_release_run(run_id: str | None) -> bool:
    normalized = str(run_id or "")
    project_root = Path(__file__).resolve().parents[2]
    return normalized in NON_RELEASE_RUNS or is_non_release_recorded(normalized, project_root)


def require_release_candidate_run(run_id: str | None, *, context: str = "release gate") -> None:
    normalized = str(run_id or "")
    try:
        non_release = is_non_release_run(normalized)
    except NonReleaseRecordError as exc:
        raise SystemExit(f"{context} refuses untrusted non-release record for run id {normalized}: {exc}") from exc
    if non_release:
        raise SystemExit(f"{context} refuses known non-release run id: {normalized}")
