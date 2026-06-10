from __future__ import annotations

import argparse
from pathlib import Path

from kobun_autonomy.augmentation_audit import audit_augmentation_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit training augmentation provenance.")
    parser.add_argument("manifest", type=Path, nargs="?", default=Path("data/training_augmentation_manifest.json"))
    parser.add_argument("--no-local-file-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = audit_augmentation_manifest(args.manifest, require_local_files=not args.no_local_file_check)
    print(f"training_augmentation_manifest_audit path={args.manifest} errors={len(errors)}")
    if errors:
        for error in errors[:20]:
            print(error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
