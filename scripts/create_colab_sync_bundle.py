from __future__ import annotations

import argparse
import json
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ROOT_FILES = {
    ".gitignore",
    "LICENSE",
    "NOTICE.md",
    "README.md",
    "SECURITY.md",
    "codex_review_loop.config.toml",
    "pyproject.toml",
    "uv.lock",
}

DOC_ALLOWLIST = {
    "docs/ARCHITECTURE.md",
    "docs/AUTONOMY_ARCHITECTURE.md",
    "docs/DATA_AND_RELEASE_POLICY.md",
    "docs/PUBLICATION_AUDIT.md",
}

ROOT_DIRS = {
    "data",
    "docs",
    "notebooks",
    "scripts",
    "src",
}

EXCLUDED_PARTS = {
    ".git",
    ".tools",
    ".venv",
    ".venv-dml",
    ".pytest_cache",
    "__pycache__",
    "checkpoints",
    "dist",
    "release",
    "run_snapshots",
    "training_snapshots",
}

EXCLUDED_SUFFIXES = {
    ".egg-info",
    ".log",
    ".pyc",
    ".pt",
    ".safetensors",
    ".tmp",
    ".zip",
}

LOG_ALLOWLIST = {
    "logs/evaluation_board.json",
    "logs/llm_review_packets/project.json",
    "logs/preflight_gate_old_japanese_0_1b.json",
    "logs/public_manifest_summary.json",
    "logs/source_quality_board.json",
    "logs/static_quality_manifest_old_japanese_0_1b.json",
    "logs/zero_base_review_gate_old_japanese_0_1b.json",
}

LOG_ALLOW_PREFIXES = (
    "logs/zero_base_review_artifacts/",
)

FORBIDDEN_EXACT_NAMES = {
    ".env",
    ".netrc",
    "auth.json",
    "cookies.sqlite",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "login data",
    "token.json",
}

FORBIDDEN_NAME_PARTS = {
    "handoff",
    "raw_thread",
    "thread_log",
    "codex_context",
}

INTERNAL_STATE_PATTERNS = {
    "codex_thread_uri": re.compile("codex" + r"://", re.IGNORECASE),
    "local_user_path": re.compile(r"(?:[A-Za-z]:[\\/]+Users[\\/]+|[A-Za-z]:\\\\+Users\\\\+)", re.IGNORECASE),
    "local_project_path": re.compile(r"ExampleWorkstation(?:[\\/]+|\\\\+)ExampleProjects", re.IGNORECASE),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
}

TEXT_SUFFIXES = {
    ".cfg",
    ".ipynb",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

CONTENT_SCAN_EXEMPTIONS: set[str] = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a sanitized project bundle for a private Colab/Drive CUDA run.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--project-root-name", default="kobun-tiny-llm")
    parser.add_argument("--allow-missing-review-gate", action="store_true")
    return parser.parse_args()


def rel_posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def is_under_selected_root(path: Path) -> bool:
    if path.parent == ROOT and path.name in ROOT_FILES:
        return True
    try:
        first = path.relative_to(ROOT).parts[0]
    except ValueError:
        return False
    return first in ROOT_DIRS or first == "logs"


def path_allowed(path: Path) -> bool:
    rel = rel_posix(path)
    if not is_under_selected_root(path):
        return False
    parts = set(path.relative_to(ROOT).parts)
    if parts & EXCLUDED_PARTS:
        return False
    if any(part.endswith(".egg-info") for part in parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    lowered = rel.lower()
    if path.name.lower() in FORBIDDEN_EXACT_NAMES:
        return False
    if any(part in lowered for part in FORBIDDEN_NAME_PARTS):
        return False
    if rel.startswith("docs/") and rel not in DOC_ALLOWLIST:
        return False
    if rel.startswith("logs/"):
        return rel in LOG_ALLOWLIST or any(rel.startswith(prefix) for prefix in LOG_ALLOW_PREFIXES)
    return True


def iter_files() -> list[Path]:
    files: list[Path] = []
    for file_name in sorted(ROOT_FILES):
        path = ROOT / file_name
        if path.exists() and path.is_file():
            files.append(path)
    for dir_name in sorted(ROOT_DIRS | {"logs"}):
        root = ROOT / dir_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path_allowed(path):
                files.append(path)
    return files


def validate_required(files: list[Path], allow_missing_review_gate: bool) -> None:
    present = {rel_posix(path) for path in files}
    required = {
        "notebooks/old_japanese_0_1b_colab_cuda.ipynb",
        "scripts/start_old_japanese_0_1b_cuda_colab_and_watch.py",
        "scripts/check_colab_cuda_environment.py",
        "logs/preflight_gate_old_japanese_0_1b.json",
        "logs/static_quality_manifest_old_japanese_0_1b.json",
    }
    if not allow_missing_review_gate:
        required.add("logs/zero_base_review_gate_old_japanese_0_1b.json")
    missing = sorted(required - present)
    if missing:
        raise SystemExit(f"colab_bundle_missing_required_files={missing}")


def validate_no_forbidden(files: list[Path]) -> None:
    forbidden: list[str] = []
    for path in files:
        rel = rel_posix(path)
        lowered = rel.lower()
        if lowered.startswith(("checkpoints/", "release/")):
            forbidden.append(rel)
        if any(part in {".venv", ".venv-dml", ".git", ".tools"} for part in path.relative_to(ROOT).parts):
            forbidden.append(rel)
        if path.suffix.lower() in {".pt", ".safetensors", ".log", ".tmp"}:
            forbidden.append(rel)
    if forbidden:
        raise SystemExit(f"colab_bundle_forbidden_files={forbidden[:20]}")


def should_scan_content(path: Path) -> bool:
    rel = rel_posix(path)
    if rel in CONTENT_SCAN_EXEMPTIONS:
        return False
    return path.suffix.lower() in TEXT_SUFFIXES


def validate_no_internal_state(files: list[Path]) -> None:
    issues: list[str] = []
    for path in files:
        rel = rel_posix(path)
        lowered = rel.lower()
        if path.name.lower() in FORBIDDEN_EXACT_NAMES:
            issues.append(f"{rel}: forbidden credential-like filename")
        if any(part in lowered for part in FORBIDDEN_NAME_PARTS):
            issues.append(f"{rel}: forbidden internal-context filename")
        if rel.startswith("docs/") and rel not in DOC_ALLOWLIST:
            issues.append(f"{rel}: docs file is not allowlisted for Colab transfer")
        if not should_scan_content(path):
            continue
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            issues.append(f"{rel}: unreadable text file: {exc}")
            continue
        for label, pattern in INTERNAL_STATE_PATTERNS.items():
            if pattern.search(text):
                issues.append(f"{rel}: matched internal/secret pattern {label}")
    if issues:
        raise SystemExit(f"colab_bundle_internal_state_pattern={issues[:20]}")


def default_out() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    downloads = Path.home() / "Downloads"
    return downloads / f"kobun-tiny-llm-colab-sync-{stamp}.zip"


def main() -> None:
    args = parse_args()
    out = (args.out or default_out()).resolve()
    if out.exists():
        raise SystemExit(f"refusing_to_overwrite_existing_bundle={out}")
    files = iter_files()
    validate_required(files, args.allow_missing_review_gate)
    validate_no_forbidden(files)
    validate_no_internal_state(files)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "old_japanese_0_1b_colab_sync_bundle_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root_name": args.project_root_name,
        "file_count": len(files),
        "hf_export": False,
        "google_credentials_included": False,
        "contains_checkpoints": False,
        "contains_release_package": False,
        "contains_codex_state": False,
        "contains_training_corpus_text_for_private_colab": True,
        "docs_allowlist": sorted(DOC_ALLOWLIST),
        "notes": [
            "Private Colab/Drive training transfer only.",
            "This private transfer includes local training/evaluation data needed for Colab execution; it is not a public release artifact.",
            "HF export, package creation, and upload are not included.",
            "Google authentication stays in browser drive.mount flow.",
            "Handoff docs, assistant thread URIs, credentials, checkpoints, release packages, raw logs, and local absolute user paths are excluded.",
        ],
    }
    start = time.time()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(f"{args.project_root_name}/logs/colab_sync_bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        for path in files:
            archive.write(path, f"{args.project_root_name}/{rel_posix(path)}")
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"colab_sync_bundle={out}")
    print(f"colab_sync_bundle_size_mb={size_mb:.2f}")
    print(f"colab_sync_bundle_file_count={len(files)}")
    print(f"colab_sync_bundle_seconds={time.time() - start:.2f}")
    print("hf_export=false")


if __name__ == "__main__":
    main()
