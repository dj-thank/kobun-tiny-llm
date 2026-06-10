from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
OUT_DIR = ROOT / "logs" / "eval_snapshots" / "eval_snapshot_provenance_contract_test"


def main() -> None:
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    try:
        completed = subprocess.run(
            [
                str(PYTHON),
                "scripts/snapshot_eval_files.py",
                "--out-dir",
                str(OUT_DIR.relative_to(ROOT)),
                "--named",
                "primary=data/eval/clean_current/grammar_minimal_pairs.jsonl",
                "--named",
                "waka_generation_prompts=data/eval/clean_current/waka_generation_prompts.jsonl",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(completed.stdout + completed.stderr)
        output = completed.stdout
        for needle in (
            "content_hashes=",
            "source_sha256=",
            "audited_source=",
            "audited_source_sha256=",
            "eval_provenance_manifest_sha256=",
            "removed_from_source=",
        ):
            if needle not in output:
                raise SystemExit(f"eval snapshot provenance output missing {needle}")
        if "audited_source=data\\eval\\grammar_minimal_pairs.jsonl" not in output and "audited_source=data/eval/grammar_minimal_pairs.jsonl" not in output:
            raise SystemExit("clean_current primary snapshot was not bound to audited source eval file")
    finally:
        shutil.rmtree(OUT_DIR, ignore_errors=True)
    print("eval_snapshot_provenance_ok=true")


if __name__ == "__main__":
    main()
