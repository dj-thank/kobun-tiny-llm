from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PUBLIC_ROOT_FILES = {
    ".gitignore",
    "LICENSE",
    "NOTICE.md",
    "README.md",
    "SECURITY.md",
    "codex_review_loop.config.toml",
    "pyproject.toml",
}

PUBLIC_DIRS = {
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
    "data",
    "logs",
    "private_notes",
    "release",
}

TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".ipynb",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

FORBIDDEN_NAME_PARTS = {
    "handoff",
    "raw_thread",
    "thread_log",
    "codex_context",
}

FORBIDDEN_PATTERNS = {
    "codex_thread_uri": re.compile("codex" + r"://", re.IGNORECASE),
    "local_user_path": re.compile(r"(?:[A-Za-z]:[\\/]+Users[\\/]+|[A-Za-z]:\\\\+Users\\\\+)", re.IGNORECASE),
    "local_project_path": re.compile(r"ExampleWorkstation(?:[\\/]+|\\\\+)ExampleProjects", re.IGNORECASE),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def public_source_files() -> list[Path]:
    files: list[Path] = []
    for name in sorted(PUBLIC_ROOT_FILES):
        path = ROOT / name
        if path.exists() and path.is_file():
            files.append(path)
    for dirname in sorted(PUBLIC_DIRS):
        root = ROOT / dirname
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            parts = set(path.relative_to(ROOT).parts)
            if parts & EXCLUDED_PARTS:
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            files.append(path)
    return files


def main() -> None:
    issues: list[str] = []
    for path in public_source_files():
        path_rel = rel(path)
        lowered = path_rel.lower()
        if any(part in lowered for part in FORBIDDEN_NAME_PARTS):
            issues.append(f"{path_rel}: forbidden internal-context filename")
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            issues.append(f"{path_rel}: unreadable text file: {exc}")
            continue
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                issues.append(f"{path_rel}: matched {label}")
    if issues:
        preview = "\n".join(issues[:40])
        raise SystemExit("source_release_hygiene_failed:\n" + preview)
    print(f"source_release_hygiene_ok=true files={len(public_source_files())}")


if __name__ == "__main__":
    main()
