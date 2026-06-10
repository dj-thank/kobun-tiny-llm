from __future__ import annotations

import json
import tempfile
from pathlib import Path


def main() -> None:
    payload = {"run_id": "old_japanese_0_1b_dml_test", "exit_code": 0}
    with tempfile.TemporaryDirectory() as tmp:
        no_bom = Path(tmp) / "no_bom.json"
        bom = Path(tmp) / "bom.json"
        no_bom.write_text(json.dumps(payload), encoding="utf-8")
        bom.write_text(json.dumps(payload), encoding="utf-8-sig")
        for path in (no_bom, bom):
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
            if loaded != payload:
                raise SystemExit(f"sentinel JSON encoding read mismatch: {path}")
        if no_bom.read_bytes().startswith(b"\xef\xbb\xbf"):
            raise SystemExit("BOM-less fixture unexpectedly has a BOM")
        if not bom.read_bytes().startswith(b"\xef\xbb\xbf"):
            raise SystemExit("BOM fixture unexpectedly lacks a BOM")
    print("run_completion_json_encoding_ok=true")


if __name__ == "__main__":
    main()
