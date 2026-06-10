from __future__ import annotations

"""compatibility shim for the autonomy release policy namespace."""

from kobun_autonomy.release_policy import (
    NON_RELEASE_RUNS,
    is_non_release_run,
    require_release_candidate_run,
)

__all__ = ["NON_RELEASE_RUNS", "is_non_release_run", "require_release_candidate_run"]
