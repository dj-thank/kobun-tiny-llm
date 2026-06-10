from __future__ import annotations

import argparse
import json
from pathlib import Path


CORE_AUXILIARIES = {
    "る",
    "らる",
    "す",
    "さす",
    "しむ",
    "む",
    "むず",
    "まし",
    "ず",
    "じ",
    "まほし",
    "き",
    "けり",
    "つ",
    "ぬ",
    "たり",
    "けむ",
    "たし",
    "らむ",
    "べし",
    "らし",
    "めり",
    "なり",
    "まじ",
    "ごとし",
    "り",
}


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Kobun grammar rule tables for missing coverage.")
    parser.add_argument("--auxiliaries", type=Path, default=Path("data/grammar/auxiliary_rules.jsonl"))
    parser.add_argument("--genres", type=Path, default=Path("data/grammar/genre_rules.jsonl"))
    parser.add_argument("--grammar-rules", type=Path, default=Path("data/grammar/rules.jsonl"))
    parser.add_argument("--morph-examples", type=Path, default=Path("data/grammar/morph_examples.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aux_rows = load_jsonl(args.auxiliaries)
    genre_rows = load_jsonl(args.genres)
    grammar_rows = load_jsonl(args.grammar_rules)
    morph_rows = load_jsonl(args.morph_examples)

    aux_lemmas = {str(row["lemma"]) for row in aux_rows}
    missing = sorted(CORE_AUXILIARIES - aux_lemmas)
    if missing:
        print(f"missing_core_auxiliaries={missing}")
        raise SystemExit(1)

    bad_rows = []
    for row in aux_rows:
        forms = row.get("forms")
        connects_to = row.get("connects_to")
        if not isinstance(forms, dict) or not forms:
            bad_rows.append(f"{row.get('lemma')}: missing forms")
        if not isinstance(connects_to, list) or not connects_to:
            bad_rows.append(f"{row.get('lemma')}: missing connects_to")
    if bad_rows:
        for item in bad_rows:
            print(item)
        raise SystemExit(1)

    genre_categories = {str(row.get("category", "")) for row in genre_rows}
    required_genre_categories = {"meter", "makurakotoba", "kakekotoba", "engo", "ending"}
    missing_genre = sorted(required_genre_categories - genre_categories)
    if missing_genre:
        print(f"missing_genre_categories={missing_genre}")
        raise SystemExit(1)

    required_rentai_endings = {"く", "き", "む", "る"}
    for row in grammar_rows:
        if row.get("expected_form") == "連体形":
            endings = {str(value) for value in row.get("allowed_endings", [])}
            missing_endings = sorted(required_rentai_endings - endings)
            if missing_endings:
                print(f"{row.get('rule_id')}: missing common rentaikei endings {missing_endings}")
                raise SystemExit(1)

    for row in genre_rows:
        if row.get("category") == "kakekotoba" and not isinstance(row.get("sense_markers"), dict):
            print(f"{row.get('rule_id')}: kakekotoba needs sense_markers")
            raise SystemExit(1)

    for row in morph_rows:
        text = str(row.get("text", ""))
        tokens = row.get("tokens", [])
        surfaces = [str(token.get("surface", "")) for token in tokens if isinstance(token, dict)]
        if text == "いかでか知らむ。" and not {"知ら", "む"}.issubset(set(surfaces)):
            print("morph_examples: いかでか知らむ must annotate 知ら + む")
            raise SystemExit(1)
        if text == "帝おはします。" and "おはします" not in surfaces:
            print("morph_examples: 帝おはします must annotate おはします")
            raise SystemExit(1)

    print(f"auxiliary_rule_rows={len(aux_rows)} core_auxiliaries_covered={len(CORE_AUXILIARIES)}")
    print(f"genre_rule_rows={len(genre_rows)} categories={','.join(sorted(genre_categories))}")
    print(f"grammar_rule_rows={len(grammar_rows)} morph_examples={len(morph_rows)}")


if __name__ == "__main__":
    main()
