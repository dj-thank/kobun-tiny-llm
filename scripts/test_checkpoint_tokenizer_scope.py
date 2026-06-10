from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from build_tokenizer_public_vocab import CORE_JAPANESE_RANGES, TOKENIZER_POLICY, core_japanese_inventory
from kobun_llm.tokenizer import ByteFallbackCharTokenizer
from split_policy import SPLIT_POLICY


ROOT = Path(__file__).resolve().parents[1]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        train = work / "clean" / "train.txt"
        val = work / "clean" / "val.txt"
        test = work / "clean" / "test.txt"
        write_text(train, "あい\n")
        write_text(val, "あ羽\n")
        write_text(test, "あ雲\n")
        manifest = work / "corpus_manifest.jsonl"
        rows = [
            {
                "source_id": "train",
                "include_in_training": True,
                "work_id": "work:源氏物語",
                "split_group_key": "work:源氏物語",
                "split_policy": SPLIT_POLICY,
                "split_role": "train",
                "clean_file": str(train),
            },
            {
                "source_id": "validation",
                "include_in_training": True,
                "work_id": "work:土佐日記",
                "split_group_key": "work:土佐日記",
                "split_policy": SPLIT_POLICY,
                "split_role": "validation",
                "clean_file": str(val),
            },
            {
                "source_id": "test",
                "include_in_training": True,
                "work_id": "work:枕草子",
                "split_group_key": "work:枕草子",
                "split_policy": SPLIT_POLICY,
                "split_role": "test",
                "clean_file": str(test),
            },
        ]
        manifest.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        direct_text = "".join(sorted(set(train.read_text(encoding="utf-8")) | core_japanese_inventory()))
        tokenizer_path = work / "tokenizer_public_char_vocab.txt"
        tokenizer_path.write_text(direct_text, encoding="utf-8", newline="")
        tokenizer_bytes = tokenizer_path.read_bytes()
        meta_path = work / "tokenizer_public_char_vocab.meta.json"
        core_text = "".join(sorted(core_japanese_inventory()))
        meta = {
            "policy": TOKENIZER_POLICY,
            "split_policy": SPLIT_POLICY,
            "byte_fallback": True,
            "byte_fallback_tokens": 256,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "train_source_count": 1,
            "core_inventory_ranges": [
                {"name": name, "start": f"U+{start:04X}", "end": f"U+{end:04X}"}
                for name, start, end in CORE_JAPANESE_RANGES
            ],
            "core_inventory_chars": len(core_japanese_inventory()),
            "direct_vocab_chars": len(set(direct_text)),
            "vocab_sha256": sha256_bytes(tokenizer_bytes),
            "core_inventory_sha256": hashlib.sha256(core_text.encode("utf-8")).hexdigest(),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tokenizer = ByteFallbackCharTokenizer.from_text(direct_text)
        checkpoint = work / "checkpoint.pt"
        torch.save(
            {
                "config": {"vocab_size": tokenizer.vocab_size},
                "tokenizer": tokenizer.to_dict(),
                "metadata": {
                    "tokenizer_type": "byte_fallback_char_v1",
                    "byte_fallback": True,
                    "tokenizer_source": TOKENIZER_POLICY,
                    "tokenizer_extra_data": [
                        {
                            "path": str(tokenizer_path),
                            "sha256": sha256_bytes(tokenizer_bytes),
                            "bytes": tokenizer_path.stat().st_size,
                        }
                    ],
                    "provenance_files": [
                        {
                            "path": str(manifest),
                            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                            "bytes": manifest.stat().st_size,
                        },
                        {
                            "path": str(meta_path),
                            "sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                            "bytes": meta_path.stat().st_size,
                        },
                    ],
                },
            },
            checkpoint,
        )
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_checkpoint_tokenizer_scope.py"),
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(manifest),
                "--tokenizer-extra-data",
                str(tokenizer_path),
                "--tokenizer-meta",
                str(meta_path),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            raise SystemExit(output)
        for fragment in (
            "checkpoint_tokenizer_vocab_scope",
            "forbidden_heldout_tokenizer_leakage=0",
            "heldout_missing_from_tokenizer=0",
            "checkpoint_bound=true",
        ):
            if fragment not in output:
                raise SystemExit(f"checkpoint_tokenizer_scope_output_missing={fragment!r}\n{output}")
    print("checkpoint_tokenizer_scope_ok=true")


if __name__ == "__main__":
    main()
