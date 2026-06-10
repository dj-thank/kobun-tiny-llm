from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def checkpoint_map_location(device: Any = None) -> Any:
    """Keep full project checkpoints on CPU when loading.

    Full training checkpoints include optimizer, scaler, RNG, tokenizer, and
    metadata in addition to model tensors. Loading that whole payload directly
    onto CUDA can waste VRAM or OOM before the caller has a chance to keep only
    the tensors it needs. Callers should instantiate the model/optimizer on the
    target device and let state loading copy tensors across.
    """

    return "cpu"


def load_trusted_checkpoint(path: str | Path, map_location: Any = None) -> Any:
    """Load a local checkpoint produced by this project.

    PyTorch 2.6 changed torch.load's default to weights_only=True, which rejects
    full training checkpoints that contain metadata/RNG/optimizer state. These
    project checkpoints are local artifacts, so callers that need the full payload
    should opt into the legacy behavior explicitly.
    """

    effective_map_location = checkpoint_map_location(map_location)
    try:
        return torch.load(path, map_location=effective_map_location, weights_only=False)
    except TypeError as exc:
        message = str(exc)
        if "weights_only" not in message and "unexpected keyword" not in message:
            raise
        return torch.load(path, map_location=effective_map_location)
