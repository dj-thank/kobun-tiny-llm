from __future__ import annotations

import argparse
import json
from pathlib import Path

from kobun_llm.morphology import annotate


EXAMPLES = [
    {"text": "春こそあはれなれ。", "rule_ids": ["kakari_koso_izen"], "style": "all"},
    {"text": "花ぞ咲く。", "rule_ids": ["kakari_zo_rentai"], "style": "all"},
    {"text": "人なむ来る。", "rule_ids": ["kakari_namu_rentai"], "style": "all"},
    {"text": "いかでか知らむ。", "rule_ids": ["kakari_ya_ka_rentai"], "style": "all"},
    {"text": "昔、男ありけり。", "rule_ids": ["aux_keri_inflection"], "style": "setsuwa"},
    {"text": "見し人は、今いづこにかあらむ。", "rule_ids": ["aux_ki_inflection"], "style": "all"},
    {"text": "知らぬ人の来たる。", "rule_ids": ["aux_zu_inflection"], "style": "all"},
    {"text": "御覧じたまふ。", "rule_ids": ["honorific_genji_style"], "style": "genji"},
    {"text": "聞こえたまひけり。", "rule_ids": ["honorific_genji_style", "aux_keri_inflection"], "style": "genji"},
    {"text": "帝おはします。", "rule_ids": ["honorific_genji_style"], "style": "genji"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build morphology-annotated grammar examples.")
    parser.add_argument("--out-jsonl", type=Path, default=Path("data/grammar/morph_examples.jsonl"))
    parser.add_argument("--out-text", type=Path, default=Path("data/grammar/morph_examples.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    text_blocks = []
    for example in EXAMPLES:
        tokens = [token.__dict__ | {"grammar_tags": list(token.grammar_tags)} for token in annotate(example["text"])]
        row = example | {"tokens": tokens}
        rows.append(row)
        lines = [
            f"形態素例 文体 {example['style']} 規則 {'、'.join(example['rule_ids'])}",
            example["text"],
            "形態素情報",
        ]
        for token in annotate(example["text"]):
            lines.append(
                f"{token.surface} は 語彙素 {token.lemma}、品詞 {token.pos}、"
                f"細分類 {token.subpos}、活用型 {token.conjugation_type or 'なし'}、"
                f"活用形 {token.conjugation_form or 'なし'}、文法 {'、'.join(token.grammar_tags)}。"
            )
        text_blocks.append("\n".join(lines))
    args.out_jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    args.out_text.write_text("\n\n".join(text_blocks) + "\n", encoding="utf-8")
    print(f"wrote {args.out_jsonl}")
    print(f"wrote {args.out_text}")


if __name__ == "__main__":
    main()
