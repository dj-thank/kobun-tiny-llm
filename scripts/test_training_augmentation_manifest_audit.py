from __future__ import annotations

import json
import tempfile
from pathlib import Path

from kobun_autonomy.augmentation_audit import audit_augmentation_manifest


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = ROOT / "data" / "training_augmentation_manifest.json"
    errors = audit_augmentation_manifest(source)
    if errors:
        raise SystemExit(f"expected clean augmentation manifest, got {errors[:3]}")

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["entries"][0]["llm_generated_corpus_text"] = True
    payload["entries"][1]["role"] = "unexpected_role"
    payload["entries"][2]["sha256"] = ""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "training_augmentation_manifest.json"
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        corrupt_errors = audit_augmentation_manifest(target, require_local_files=False)
    expected_fragments = (
        "llm_generated_corpus_text",
        "unsupported role",
        "missing sha256",
        "missing augmentation roles",
    )
    missing = [fragment for fragment in expected_fragments if not any(fragment in error for error in corrupt_errors)]
    if missing:
        raise SystemExit(f"corrupt augmentation manifest did not trigger expected errors: {missing}; got={corrupt_errors}")
    print("training_augmentation_manifest_negative_audit_ok=true")


if __name__ == "__main__":
    main()
