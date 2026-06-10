from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def run_check(manifest: Path, eval_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/check_eval_source_overlap.py",
            "--manifest",
            str(manifest),
            "--eval",
            str(eval_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def grammar_scope_for_split(split_role: str) -> str:
    if split_role == "reference":
        return "reference_only_outside_genji_era_scope"
    if split_role == "excluded":
        return "unregistered_outside_genji_era_scope"
    return "genji-era-reference"


def manifest_row(tmp: Path, title: str, work_id: str, split_role: str, clean: str, poem: str) -> dict[str, object]:
    slug = split_role
    clean_file = tmp / f"{slug}_clean.txt"
    records_file = tmp / f"{slug}_records.jsonl"
    readings_file = tmp / f"{slug}_readings.txt"
    clean_file.write_text(clean, encoding="utf-8")
    write_jsonl(records_file, [{"poem": poem, "reading": poem}])
    readings_file.write_text(poem + "\n", encoding="utf-8")
    return {
        "source_id": f"test:{slug}",
        "title": title,
        "work_id": work_id,
        "split_group_key": work_id,
        "split_role": split_role,
        "split_policy": "work_group_genji_reference_v1",
        "grammar_scope": grammar_scope_for_split(split_role),
        "include_in_training": True,
        "period": "平安",
        "genre": "waka",
        "style": "waka",
        "clean_file": str(clean_file),
        "clean_sha256": sha256_file(clean_file),
        "records_file": str(records_file),
        "records_sha256": sha256_file(records_file),
        "readings_file": str(readings_file),
        "readings_sha256": sha256_file(readings_file),
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="eval_source_overlap_") as raw_tmp:
        tmp = Path(raw_tmp)
        manifest = tmp / "manifest.jsonl"
        source_waka = "あいうえおかきくけこさしすせそたちつてとなにぬねの"
        near_source_waka = "あいうえおかきくけこさしずせそたちつてとなにぬねの"
        rows = [
            manifest_row(tmp, "源氏物語 synthetic", "work:源氏物語", "train", "これは訓練用の合成本文です。", "まみむめもやゆよらりるれろわをん"),
            manifest_row(tmp, "土佐日記 synthetic", "work:土佐日記", "validation", "これは検証用の合成本文です。", source_waka),
            manifest_row(tmp, "枕草子 synthetic", "work:枕草子", "test", "これは試験用の合成本文です。", "かきくけこさしすせそたちつてとなにぬねのはひふへほ"),
            manifest_row(tmp, "方丈記 synthetic", "work:方丈記", "reference", "これは参考用の合成本文です。長い参照本文として検査します。", "ほへふひはのねぬになとてつちたそせすしさ"),
            manifest_row(tmp, "竹取物語 synthetic", "work:竹取物語", "excluded", "これは除外用の合成本文です。長い除外本文として検査します。", "ざじずぜぞばびぶべぼぱぴぷぺぽがぎぐげご"),
        ]
        write_jsonl(manifest, rows)

        leaking_eval = tmp / "leaking_eval.jsonl"
        write_jsonl(
            leaking_eval,
            [
                {
                    "text": near_source_waka,
                    "reading": near_source_waka,
                    "expected_meter": [5, 7, 5, 7, 7],
                    "rule_ids": ["waka_meter_57577"],
                }
            ],
        )
        leaking = run_check(manifest, leaking_eval)
        if leaking.returncode == 0 or "waka_variant_hits=0" in (leaking.stdout + leaking.stderr):
            raise SystemExit(
                "eval source overlap checker did not reject a near-duplicate waka item.\n"
                + leaking.stdout
                + leaking.stderr
            )

        clean_eval = tmp / "clean_eval.jsonl"
        write_jsonl(
            clean_eval,
            [
                {
                    "text": "がぎぐげござじずぜぞだぢづでどばびぶべぼ",
                    "reading": "がぎぐげござじずぜぞだぢづでどばびぶべぼ",
                    "expected_meter": [5, 7, 5, 7, 7],
                    "rule_ids": ["waka_meter_57577"],
                }
            ],
        )
        clean = run_check(manifest, clean_eval)
        if clean.returncode != 0 or "hits=0" not in clean.stdout:
            raise SystemExit("eval source overlap checker rejected a clean synthetic eval.\n" + clean.stdout + clean.stderr)
        for role in ("train", "validation", "test", "reference", "excluded"):
            if f'"{role}"' not in clean.stdout:
                raise SystemExit("eval source overlap checker did not report every source role.\n" + clean.stdout)

        reference_leak = tmp / "reference_leak_eval.jsonl"
        write_jsonl(reference_leak, [{"text": "これは参考用の合成本文です。長い参照本文として検査します。"}])
        reference = run_check(manifest, reference_leak)
        if reference.returncode == 0 or "source_role=reference" not in (reference.stdout + reference.stderr):
            raise SystemExit(
                "eval source overlap checker did not reject copied reference prose.\n"
                + reference.stdout
                + reference.stderr
            )

        excluded_leak = tmp / "excluded_leak_eval.jsonl"
        write_jsonl(excluded_leak, [{"text": "これは除外用の合成本文です。長い除外本文として検査します。"}])
        excluded = run_check(manifest, excluded_leak)
        if excluded.returncode == 0 or "source_role=excluded" not in (excluded.stdout + excluded.stderr):
            raise SystemExit(
                "eval source overlap checker did not reject copied excluded prose.\n"
                + excluded.stdout
                + excluded.stderr
            )

        prefix_leak = tmp / "prefix_leak_eval.jsonl"
        write_jsonl(
            prefix_leak,
            [
                {
                    "prefix": near_source_waka,
                    "expected_meter": [5, 7, 5, 7, 7],
                    "rule_ids": ["waka_meter_57577"],
                }
            ],
        )
        prefix = run_check(manifest, prefix_leak)
        if prefix.returncode == 0 or ":prefix " not in (prefix.stdout + prefix.stderr):
            raise SystemExit(
                "eval source overlap checker did not reject a copied waka prefix.\n"
                + prefix.stdout
                + prefix.stderr
            )

        prefixes_leak = tmp / "prefixes_leak_eval.jsonl"
        write_jsonl(
            prefixes_leak,
            [
                {
                    "prefixes": ["これは検証用の", near_source_waka],
                    "check": "valid_prefixes",
                    "rule_ids": ["waka_meter_57577"],
                }
            ],
        )
        prefixes = run_check(manifest, prefixes_leak)
        if prefixes.returncode == 0 or ":prefixes[1] " not in (prefixes.stdout + prefixes.stderr):
            raise SystemExit(
                "eval source overlap checker did not reject a copied waka prefixes entry.\n"
                + prefixes.stdout
                + prefixes.stderr
            )
    print("eval_source_overlap_test_ok=true")


if __name__ == "__main__":
    main()
