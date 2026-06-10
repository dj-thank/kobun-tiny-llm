from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from kobun_llm.checkpoint_io import load_trusted_checkpoint
from kobun_autonomy.release_policy import is_non_release_run


def is_loadable_checkpoint(path: Path) -> bool:
    try:
        payload = load_trusted_checkpoint(path, map_location="cpu")
    except Exception as exc:
        print(f"skip_invalid_checkpoint={path} reason={type(exc).__name__}: {exc}", flush=True)
        return False
    required = {"model", "config", "tokenizer"}
    missing = required - set(payload)
    if missing:
        print(f"skip_invalid_checkpoint={path} reason=missing_keys:{','.join(sorted(missing))}", flush=True)
        return False
    return True


def is_valid_checkpoint(path: Path) -> bool:
    """Backward-compatible loadability predicate for interactive scripts."""

    return is_loadable_checkpoint(path)


def release_gate_ok(path: Path, release_prefix: str) -> bool:
    run_id = path.stem.removesuffix("_best")
    if is_non_release_run(run_id):
        print(f"skip_release_ineligible_checkpoint={path} reason=known_non_release_run", flush=True)
        return False
    scripts_dir = Path(__file__).parent
    expected = Path("checkpoints") / f"{run_id}_best.pt"
    if path.as_posix() != expected.as_posix() and path.resolve(strict=False) != expected.resolve(strict=False):
        print(f"skip_release_ineligible_checkpoint={path} reason=not_exact_best_path expected={expected}", flush=True)
        return False
    command = [
        sys.executable,
        str(scripts_dir / "check_release_gate.py"),
        "--run-id",
        run_id,
        "--checkpoint",
        str(path),
        "--eval-results",
        str(Path("logs") / f"eval_results_{run_id}.json"),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        reason = (result.stderr or result.stdout).strip().splitlines()[:2]
        print(f"skip_release_ineligible_checkpoint={path} reason={' | '.join(reason)}", flush=True)
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print the newest loadable checkpoint matching a glob. By default this is "
            "for interactive smoke use only; pass --release-eligible for release selection."
        )
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Glob pattern to search. May be passed multiple times; earlier patterns have priority.",
    )
    parser.add_argument("--fallback", type=Path, default=None)
    parser.add_argument("--release-eligible", action="store_true")
    parser.add_argument("--require-release-prefix", default="old-japanese-0.1B")
    return parser.parse_args()


def checkpoint_ok(path: Path, release_eligible: bool, release_prefix: str) -> bool:
    if not is_loadable_checkpoint(path):
        return False
    if release_eligible and not release_gate_ok(path, release_prefix):
        return False
    return True


def main() -> None:
    args = parse_args()
    patterns = args.pattern or ["checkpoints/kobun_qwen3_12l_worldclass_*_best.pt"]
    for pattern in patterns:
        candidates = sorted(Path().glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        for candidate in candidates:
            if checkpoint_ok(candidate, args.release_eligible, args.require_release_prefix):
                print(candidate)
                return
    if args.fallback is not None and args.fallback.exists() and checkpoint_ok(
        args.fallback,
        args.release_eligible,
        args.require_release_prefix,
    ):
        print(args.fallback)
        return
    raise SystemExit(f"No valid checkpoint found for patterns: {patterns}")


if __name__ == "__main__":
    main()
