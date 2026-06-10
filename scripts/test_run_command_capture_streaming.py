from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stdout = root / "out.log"
        stderr = root / "err.log"
        command = [
            sys.executable,
            "scripts/run_command_capture.py",
            "--stdout",
            str(stdout),
            "--stderr",
            str(stderr),
            "--",
            sys.executable,
            "-u",
            "-c",
            "import sys,time; print('first', flush=True); time.sleep(3); print('second', flush=True)",
        ]
        proc = subprocess.Popen(command)
        try:
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if stdout.exists() and "first" in stdout.read_text(encoding="utf-8"):
                    break
                time.sleep(0.1)
            else:
                raise SystemExit("run_command_capture did not stream the first small stdout line before process exit")
        finally:
            proc.wait(timeout=10)
        if proc.returncode != 0:
            raise SystemExit(f"run_command_capture child failed: exit={proc.returncode}")
        text = stdout.read_text(encoding="utf-8")
        if "first" not in text or "second" not in text:
            raise SystemExit(f"run_command_capture missing expected output: {text!r}")
    print("run_command_capture_streaming_ok=true")


if __name__ == "__main__":
    main()
