from __future__ import annotations

from typing import Any

import torch


def clean_device_description(text: str) -> str:
    cleaned = "".join(ch for ch in str(text) if ord(ch) >= 32 and ord(ch) != 127).strip()
    return cleaned or "unknown-device"


def dml_available() -> bool:
    try:
        import torch_directml
    except ImportError:
        return False
    return torch_directml.device_count() > 0


def resolve_device(requested: str) -> Any:
    if requested == "auto":
        if real_cuda_runtime_available():
            return "cuda"
        if dml_available():
            import torch_directml

            return torch_directml.device()
        return "cpu"
    if requested == "cuda":
        require_real_cuda_runtime("CUDA")
        return "cuda"
    if requested == "dml":
        try:
            import torch_directml
        except ImportError as exc:
            raise SystemExit("DirectML was requested but torch-directml is not installed.") from exc
        if torch_directml.device_count() < 1:
            raise SystemExit("DirectML was requested but no DirectML device is available.")
        return torch_directml.device()
    return requested


def is_cuda_device(device: Any) -> bool:
    return str(device).startswith("cuda")


def is_hip_runtime() -> bool:
    return bool(getattr(torch.version, "hip", None))


def cuda_runtime_kind() -> str:
    if is_hip_runtime():
        return "hip"
    if torch.cuda.is_available():
        return "cuda"
    return "none"


def real_cuda_runtime_available() -> bool:
    return torch.cuda.is_available() and not is_hip_runtime()


def require_real_cuda_runtime(context: str = "CUDA") -> None:
    if is_hip_runtime():
        raise SystemExit(
            f"{context} requires an NVIDIA CUDA PyTorch runtime, but this process is using "
            f"ROCm/HIP torch_hip={torch.version.hip}. Use a HIP-specific supervised backend "
            "policy instead of labeling this run or evidence as CUDA."
        )
    if not torch.cuda.is_available():
        built = torch.backends.cuda.is_built()
        version = torch.version.cuda
        raise SystemExit(
            f"{context} was requested but PyTorch cannot use CUDA.\n"
            f"torch={torch.__version__} cuda_built={built} torch_cuda={version}\n"
            "Install a CUDA-enabled PyTorch build, then retry with --device cuda."
        )


def is_dml_device(device: Any) -> bool:
    return str(device).startswith("privateuseone")


def device_backend(device: Any) -> str:
    if is_cuda_device(device):
        if is_hip_runtime():
            return "hip"
        return "cuda"
    if is_dml_device(device):
        return "dml"
    if str(device) == "cpu":
        return "cpu"
    return str(device).split(":", 1)[0]


def describe_device(device: Any) -> str:
    if is_cuda_device(device) and torch.cuda.is_available():
        index = torch.cuda.current_device()
        name = clean_device_description(torch.cuda.get_device_name(index))
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / (1024**3)
        if is_hip_runtime():
            return f"hip:{index} {name} ({total_gb:.1f} GiB VRAM, rocm={torch.version.hip})"
        return f"cuda:{index} {name} ({total_gb:.1f} GiB VRAM)"
    if is_dml_device(device):
        try:
            import torch_directml
        except ImportError:
            return "directml"
        return f"directml:{clean_device_description(torch_directml.device_name(0))}"
    return "cpu"
