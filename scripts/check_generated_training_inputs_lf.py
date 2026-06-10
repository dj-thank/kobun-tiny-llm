from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GENERATED_TRAINING_INPUTS = [
    Path("data/kobun_labeled_grammar_train.txt"),
    Path("data/kobun_labeled_grammar_val.txt"),
    Path("data/kobun_labeled_grammar_test.txt"),
    Path("data/kobun_labeled_grammar_boost_train.txt"),
    Path("data/kobun_worldclass_corpus.txt"),
    Path("data/waka/waka_meter_corpus.txt"),
]


def main() -> None:
    issues: list[str] = []
    for rel in GENERATED_TRAINING_INPUTS:
        path = ROOT / rel
        if not path.exists():
            issues.append(f"{rel.as_posix()}: missing")
            continue
        data = path.read_bytes()
        crlf = data.count(b"\r\n")
        cr_only = data.count(b"\r") - crlf
        if crlf or cr_only:
            issues.append(f"{rel.as_posix()}: crlf={crlf} cr_only={cr_only}")
    if issues:
        raise SystemExit("generated_training_inputs_not_lf:\n" + "\n".join(issues))
    print(f"generated_training_inputs_lf_ok=true files={len(GENERATED_TRAINING_INPUTS)}")


if __name__ == "__main__":
    main()
