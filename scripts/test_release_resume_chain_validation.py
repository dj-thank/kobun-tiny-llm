from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from kobun_llm.release_resume import validate_release_resume_chain_from_payload


CONFIG = {
    "vocab_size": 16,
    "block_size": 8,
    "n_layer": 1,
    "n_head": 1,
    "n_embd": 8,
    "dropout": 0.0,
}
TOKENIZER = {"tokenizer_type": "byte_fallback_char_v1", "tokens": ["<unk>", "あ"]}
FILE_RECORD = {"path": "data/run_snapshots/run/file.txt", "sha256": "f" * 64, "bytes": 12}


def optimizer_state(step: int) -> dict[str, Any]:
    tensor = torch.zeros(1, dtype=torch.float32)
    return {
        "optimizer_type": "simple-adamw",
        "step_count": step,
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "exp_avg": [tensor.clone()],
        "exp_avg_sq": [tensor.clone()],
    }


def payload(step: int, *, resume: str = "") -> dict[str, Any]:
    return {
        "step": step,
        "best_val": 1.0,
        "config": copy.deepcopy(CONFIG),
        "tokenizer": copy.deepcopy(TOKENIZER),
        "optimizer": optimizer_state(step),
        "metadata": {
            "run_id": "old_japanese_0_1b_dml_test",
            "backend": "dml",
            "seed": 20260509,
            "optimizer": "simple-adamw",
            "data_sha256": "a" * 64,
            "val_data_sha256": "b" * 64,
            "test_data_sha256": "c" * 64,
            "tokenizer_source": "train_split_plus_core_japanese_inventory_plus_utf8_byte_fallback_v1",
            "tokenizer_type": "byte_fallback_char_v1",
            "byte_fallback": True,
            "val_oov_chars": "",
            "test_oov_chars": "",
            "release_name": "old-japanese-0.1B-preview",
            "tokenizer_extra_data": [copy.deepcopy(FILE_RECORD)],
            "provenance_files": [copy.deepcopy(FILE_RECORD)],
            "init_from": "",
            "resume": resume,
        },
    }


def validate(current: dict[str, Any], checkpoints: dict[str, dict[str, Any]], *, allow: bool = True) -> None:
    def load_checkpoint(path: Path) -> dict[str, Any]:
        key = path.name
        if key not in checkpoints:
            raise ValueError(f"unexpected checkpoint path: {path}")
        return checkpoints[key]

    validate_release_resume_chain_from_payload(
        current,
        Path("current.pt"),
        allow_same_run_resume=allow,
        expected_backend="dml",
        expected_seed=20260509,
        expected_optimizer="simple-adamw",
        expected_config=CONFIG,
        expected_tokenizer=TOKENIZER,
        expected_tokenizer_extra_data=[FILE_RECORD],
        expected_provenance_files=[FILE_RECORD],
        load_checkpoint=load_checkpoint,
        resolve_path=lambda raw: Path(raw),
    )


def must_fail(
    current: dict[str, Any],
    checkpoints: dict[str, dict[str, Any]],
    reason: str,
    *,
    allow: bool = True,
) -> None:
    try:
        validate(current, checkpoints, allow=allow)
    except ValueError:
        return
    raise SystemExit(f"release resume validator unexpectedly accepted: {reason}")


def main() -> None:
    previous = payload(3)
    current = payload(5, resume="previous.pt")
    validate(current, {"previous.pt": previous})

    bad_previous = copy.deepcopy(previous)
    bad_previous["metadata"]["backend"] = "cuda"
    must_fail(current, {"previous.pt": bad_previous}, "backend mismatch")

    bad_previous = copy.deepcopy(previous)
    bad_previous["config"]["block_size"] = 16
    must_fail(current, {"previous.pt": bad_previous}, "config mismatch")

    bad_previous = copy.deepcopy(previous)
    bad_previous["metadata"]["seed"] = 7
    must_fail(current, {"previous.pt": bad_previous}, "seed mismatch")

    bad_previous = copy.deepcopy(previous)
    bad_previous["optimizer"]["step_count"] = 2
    must_fail(current, {"previous.pt": bad_previous}, "optimizer step mismatch")

    bad_previous = payload(5)
    must_fail(current, {"previous.pt": bad_previous}, "non-monotonic step")

    bad_previous = copy.deepcopy(previous)
    bad_previous["metadata"]["provenance_files"][0]["sha256"] = "d" * 64
    must_fail(current, {"previous.pt": bad_previous}, "provenance mismatch")

    must_fail(current, {"previous.pt": previous}, "resume disallowed", allow=False)

    print("release_resume_chain_validation_ok=true")


if __name__ == "__main__":
    main()
