from __future__ import annotations

"""compatibility shim for the autonomy augmentation-audit namespace."""

from kobun_autonomy.augmentation_audit import (
    ALLOWED_AUGMENTATION_SOURCE_TYPES,
    REQUIRED_AUGMENTATION_ROLES,
    audit_augmentation_manifest,
    load_augmentation_manifest,
    require_clean_augmentation_manifest,
    sha256_file,
)

__all__ = [
    "ALLOWED_AUGMENTATION_SOURCE_TYPES",
    "REQUIRED_AUGMENTATION_ROLES",
    "audit_augmentation_manifest",
    "load_augmentation_manifest",
    "require_clean_augmentation_manifest",
    "sha256_file",
]
