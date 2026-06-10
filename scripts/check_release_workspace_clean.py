from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail if release/ contains mutable evidence or package-like artifacts before explicit export."
    )
    parser.add_argument("--release-dir", type=Path, default=Path("release"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_dir = args.release_dir
    if not release_dir.exists():
        print(f"release_workspace_clean=true release_dir={release_dir} files=0")
        return
    files = [path for path in release_dir.rglob("*") if path.is_file()]
    if files:
        for path in files[:40]:
            print(f"release_workspace_artifact={path}")
        raise SystemExit(f"release workspace is not clean: files={len(files)}")
    print(f"release_workspace_clean=true release_dir={release_dir} files={len(files)}")


if __name__ == "__main__":
    main()
