from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RULE_DIR = Path("data/rules")
REQUIRED_FILES = {
    "period_scope_policy.json",
    "genji_era_auxiliaries.json",
    "genji_era_kakari_musubi.json",
    "genji_era_honorifics.json",
    "waka_meter_rules.json",
}
REQUIRED_AUXILIARIES = {
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


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected JSON object")
    if payload.get("llm_generated_corpus_text") is not False:
        raise SystemExit(f"{path}: llm_generated_corpus_text must be false")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    missing = sorted(name for name in REQUIRED_FILES if not (RULE_DIR / name).exists())
    require(not missing, f"missing_rule_ssot_files={missing}")

    policy = load_json(RULE_DIR / "period_scope_policy.json")
    scope = policy.get("release_grammar_scope", {})
    require("genji" in str(scope.get("name", "")).lower(), "period policy must name Genji-era scope")
    forbidden = set(scope.get("forbidden_release_evidence_roles", []))
    require("llm_generated_corpus_text" in forbidden, "period policy must forbid LLM-generated corpus text")
    reference_only = set(scope.get("reference_only_roles", []))
    require("later_medieval_reference" in reference_only, "later medieval grammar must be reference-only")

    auxiliaries = load_json(RULE_DIR / "genji_era_auxiliaries.json")
    aux_set = set(auxiliaries.get("core_auxiliaries", []))
    missing_aux = sorted(REQUIRED_AUXILIARIES - aux_set)
    require(not missing_aux, f"missing_ssot_core_auxiliaries={missing_aux}")
    require(Path(str(auxiliaries.get("canonical_source", ""))).exists(), "missing auxiliary canonical source")

    kakari = load_json(RULE_DIR / "genji_era_kakari_musubi.json")
    triggers = kakari.get("required_triggers", {})
    require(triggers.get("ぞ") == "連体形", "ぞ must bind to 連体形")
    require(triggers.get("なむ") == "連体形", "なむ must bind to 連体形")
    require(triggers.get("や") == "連体形", "や must bind to 連体形")
    require(triggers.get("か") == "連体形", "か must bind to 連体形")
    require(triggers.get("こそ") == "已然形", "こそ must bind to 已然形")
    require(Path(str(kakari.get("canonical_source", ""))).exists(), "missing kakari canonical source")

    honorifics = load_json(RULE_DIR / "genji_era_honorifics.json")
    markers = set(honorifics.get("required_markers", []))
    required_markers = {"たまふ", "おはします", "思す", "聞こゆ"}
    missing_markers = sorted(required_markers - markers)
    require(not missing_markers, f"missing_honorific_markers={missing_markers}")
    for source in honorifics.get("canonical_sources", []):
        require(Path(str(source)).exists(), f"missing honorific canonical source: {source}")

    waka = load_json(RULE_DIR / "waka_meter_rules.json")
    require(waka.get("target_morae") == [5, 7, 5, 7, 7], "waka target_morae must be 5-7-5-7-7")
    categories = set(waka.get("required_categories", []))
    require({"meter", "makurakotoba", "kakekotoba", "engo", "ending"}.issubset(categories), "missing waka categories")
    require(Path(str(waka.get("canonical_source", ""))).exists(), "missing waka canonical source")

    print(f"rule_ssot_ok=true files={len(REQUIRED_FILES)} scope={scope.get('name')}")


if __name__ == "__main__":
    main()
