from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command and append raw stdout/stderr bytes to UTF-8-friendly log files."
    )
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def pump(stream, path: Path) -> None:
    fd = stream.fileno()
    with path.open("ab") as out:
        while True:
            chunk = os.read(fd, 64 * 1024)
            if not chunk:
                break
            out.write(chunk)
            out.flush()


def main() -> None:
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing command after --")
    args.stdout.parent.mkdir(parents=True, exist_ok=True)
    args.stderr.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    assert proc.stdout is not None
    assert proc.stderr is not None
    out_thread = threading.Thread(target=pump, args=(proc.stdout, args.stdout), daemon=True)
    err_thread = threading.Thread(target=pump, args=(proc.stderr, args.stderr), daemon=True)
    out_thread.start()
    err_thread.start()
    return_code = proc.wait()
    out_thread.join()
    err_thread.join()
    sys.exit(return_code)


if __name__ == "__main__":
    main()
