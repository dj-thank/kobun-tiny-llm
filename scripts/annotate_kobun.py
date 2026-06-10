from __future__ import annotations

import argparse

from kobun_llm.morphology import annotate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate kobun text with local CHJ/UniDic-like fields.")
    parser.add_argument("text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("surface\tlemma\tpos\tsubpos\tconjugation_type\tconjugation_form\treading\tperiod\tstyle\tgrammar_tags")
    for token in annotate(args.text):
        print(token.to_chj_line())


if __name__ == "__main__":
    main()
