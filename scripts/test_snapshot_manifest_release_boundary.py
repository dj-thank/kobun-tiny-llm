from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_checkpoint_training_inputs.py"


def require(text: str, needle: str) -> None:
    if needle not in text:
        raise SystemExit(f"snapshot_manifest_release_boundary_static_missing={needle}")


def main() -> None:
    script_text = SCRIPT.read_text(encoding="utf-8")
    require(script_text, "verify_snapshot_manifest_boundaries")
    require(script_text, "crosses release artifact boundary")
    require(script_text, "source_path")

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        manifest = tmp / "snapshot_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "data": {"source_path": "data/train.txt"},
                    "val_data": {"source_path": "data/validation.txt"},
                    "test_data": {"source_path": "data/test.txt"},
                    "tokenizer_extra_data": [],
                    "provenance_files": [
                        {"source_path": "release/public_manifest_summary.json"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        probe = (
            "from scripts.check_checkpoint_training_inputs import verify_snapshot_manifest_boundaries; "
            f"verify_snapshot_manifest_boundaries(__import__('pathlib').Path(r'{manifest}'))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            raise SystemExit("snapshot_manifest_release_boundary_dynamic_failed_to_reject")
        if "crosses release artifact boundary" not in (result.stderr + result.stdout):
            raise SystemExit("snapshot_manifest_release_boundary_dynamic_wrong_error")
    print("snapshot_manifest_release_boundary_ok=true")


if __name__ == "__main__":
    main()
