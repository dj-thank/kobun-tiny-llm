from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from kobun_autonomy.augmentation_audit import audit_augmentation_manifest


DEFAULT_AUGMENTATIONS = [
    {
        "path": "data/grammar/kobun_grammar_rules.txt",
        "role": "grammar_rule_text",
        "source_type": "project_authored_rule_table",
        "repeat_count": 8,
        "used_by": "scripts/build_training_corpus.py --grammar-repeat",
    },
    {
        "path": "data/grammar/morph_examples.txt",
        "role": "morphology_examples",
        "source_type": "project_authored_rule_table",
        "repeat_count": 12,
        "used_by": "scripts/build_training_corpus.py --morph-repeat",
    },
    {
        "path": "data/grammar/train_preference_pairs.jsonl",
        "role": "train_preference_pairs",
        "source_type": "project_authored_preference_pairs",
        "repeat_count": 80,
        "used_by": "scripts/build_preference_boost_corpus.py --repeat",
    },
    {
        "path": "data/waka/waka_meter_corpus.txt",
        "role": "waka_meter_training_text",
        "source_type": "derived_from_train_split_public_waka_records",
        "repeat_count": 8,
        "used_by": "scripts/build_worldclass_corpus.py --waka-meter-repeat",
    },
    {
        "path": "data/grammar/auxiliary_rules.jsonl",
        "role": "auxiliary_rule_table",
        "source_type": "project_authored_rule_table",
        "repeat_count": 6,
        "used_by": "scripts/build_worldclass_corpus.py --rule-repeat",
    },
    {
        "path": "data/grammar/genre_rules.jsonl",
        "role": "genre_rule_table",
        "source_type": "project_authored_rule_table",
        "repeat_count": 6,
        "used_by": "scripts/build_worldclass_corpus.py --rule-repeat",
    },
    {
        "path": "data/external_knowledge/classical_surface_patterns.txt",
        "role": "external_knowledge_surface_patterns",
        "source_type": "project_authored_rule_table",
        "repeat_count": 4,
        "used_by": "scripts/build_worldclass_corpus.py --external-surface-repeat",
    },
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build/audit provenance for project-authored and derived training augmentations."
    )
    parser.add_argument("--out", type=Path, default=Path("data/training_augmentation_manifest.json"))
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def build_payload() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for item in DEFAULT_AUGMENTATIONS:
        path = Path(str(item["path"]))
        entry = {
            **item,
            "path": path.as_posix(),
            "sha256": sha256_file(path) if path.exists() else "",
            "bytes": path.stat().st_size if path.exists() else 0,
            "lines": line_count(path) if path.exists() else 0,
            "copyability_status": (
                "project-authored, repo-local release rules"
                if str(item["source_type"]).startswith("project_authored")
                else "derived only from public train-split records; no validation/test items"
            ),
            "llm_generated_corpus_text": False,
            "public_release_policy": "publish hashes/roles/repeat counts only; do not publish training text",
        }
        entries.append(entry)
    return {
        "schema": "old_japanese_training_augmentation_manifest_v1",
        "attestation": (
            "These augmentation sources are project-authored rule tables or train-split-derived "
            "control text. They are not LLM-generated corpus text."
        ),
        "llm_generated_corpus_text": False,
        "entries": entries,
        "transform_scripts": [
            {
                "path": path,
                "sha256": sha256_file(Path(path)) if Path(path).exists() else "",
            }
            for path in (
                "scripts/build_training_corpus.py",
                "scripts/build_preference_boost_corpus.py",
                "scripts/build_worldclass_corpus.py",
                "scripts/build_waka_meter_corpus.py",
                "scripts/build_external_knowledge_surface_patterns.py",
            )
        ],
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.out.read_text(encoding="utf-8")) if args.audit_only else build_payload()
    if not args.audit_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors = audit_augmentation_manifest(args.out)
    print(
        f"training_augmentation_manifest path={args.out} entries={len(payload.get('entries') or [])} "
        f"errors={len(errors)} no_llm_generated_corpus_text={payload.get('llm_generated_corpus_text') is False}"
    )
    if errors:
        for error in errors[:20]:
            print(error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
