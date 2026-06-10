from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the Colab CUDA environment without reading Google credentials.")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--preflight-gate", default="logs/preflight_gate_old_japanese_0_1b.json")
    parser.add_argument("--review-gate", default="logs/zero_base_review_gate_old_japanese_0_1b.json")
    parser.add_argument("--min-vram-gb", type=float, default=8.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--allow-no-cuda", action="store_true", help="Local dry-run mode for tests outside Colab.")
    return parser.parse_args()


def run_text(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        return 127, str(exc)
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def verify_gate(project_root: Path, gate: str, review_gate: str) -> list[str]:
    issues: list[str] = []
    python = sys.executable
    checks = [
        [python, "scripts/verify_preflight_gate.py", "--gate", gate, "--max-age-minutes", "240"],
        [
            python,
            "scripts/verify_zero_base_review_gate.py",
            "--gate",
            review_gate,
            "--preflight-gate",
            gate,
            "--max-age-minutes",
            "240",
        ],
    ]
    for command in checks:
        code, output = run_text(command)
        if code != 0:
            issues.append(f"gate_check_failed command={' '.join(command)} output={output}")
    return issues


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    if not project_root.exists():
        raise SystemExit(f"project_root_missing={project_root}")
    os.chdir(project_root)

    import torch
    from kobun_llm.device import cuda_runtime_kind, real_cuda_runtime_available

    cuda_available = bool(torch.cuda.is_available())
    torch_hip_version = str(getattr(torch.version, "hip", "") or "")
    runtime_kind = cuda_runtime_kind()
    real_cuda_runtime = real_cuda_runtime_available()
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else ""
    vram_bytes = 0
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        vram_bytes = int(props.total_memory)

    issues: list[str] = []
    if not cuda_available and not args.allow_no_cuda:
        issues.append("cuda_not_available")
    if cuda_available and not real_cuda_runtime:
        issues.append(f"hip_runtime_is_not_cuda torch_hip_version={torch_hip_version}")
    if cuda_available and vram_bytes < int(args.min_vram_gb * 1024**3):
        issues.append(f"vram_below_minimum actual_gb={vram_bytes / 1024**3:.2f} min_gb={args.min_vram_gb:.2f}")

    preflight = project_root / args.preflight_gate
    review = project_root / args.review_gate
    for label, path in (("preflight_gate", preflight), ("review_gate", review)):
        if not path.exists():
            issues.append(f"{label}_missing={path}")
    if preflight.exists() and review.exists():
        issues.extend(verify_gate(project_root, args.preflight_gate, args.review_gate))

    nvidia_smi_code, nvidia_smi_output = run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )

    payload: dict[str, Any] = {
        "schema": "old_japanese_0_1b_colab_cuda_environment_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "python": sys.executable,
        "platform": platform.platform(),
        "in_colab": "google.colab" in sys.modules or Path("/content").exists(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "torch_hip_version": torch_hip_version,
        "cuda_runtime_kind": runtime_kind,
        "real_cuda_runtime": real_cuda_runtime,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "gpu_name": gpu_name,
        "vram_bytes": vram_bytes,
        "vram_gb": round(vram_bytes / 1024**3, 3) if vram_bytes else 0.0,
        "nvidia_smi_exit_code": nvidia_smi_code,
        "nvidia_smi": nvidia_smi_output[:2000],
        "preflight_gate": args.preflight_gate,
        "preflight_gate_sha256": sha256_file(preflight) if preflight.exists() else "",
        "review_gate": args.review_gate,
        "review_gate_sha256": sha256_file(review) if review.exists() else "",
        "google_credentials_read": False,
        "hf_export": False,
        "issues": issues,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
