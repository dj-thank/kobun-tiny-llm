from __future__ import annotations

"""compatibility shim for the autonomy non-release registry namespace."""

from kobun_autonomy.non_release_registry import (
    NonReleaseRecordError,
    is_non_release_recorded,
    list_non_release_run_ids,
    non_release_dir,
    non_release_record_path,
    read_non_release_record,
)

__all__ = [
    "NonReleaseRecordError",
    "is_non_release_recorded",
    "list_non_release_run_ids",
    "non_release_dir",
    "non_release_record_path",
    "read_non_release_record",
]
