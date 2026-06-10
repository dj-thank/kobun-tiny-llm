# External Kobun Knowledge

This directory stores provenance and assistant-derived rule understanding from external references.

The external page prose is not training data. Modern Japanese explanations from external pages are forbidden for direct training.

## Files

- `sources.jsonl`: source metadata, licenses, retrieval date, and training-use policy.
- `waka_rhetoric_cards.jsonl`: structured waka rhetoric understanding for rule review.
- `classical_grammar_cards.jsonl`: structured grammar understanding for rule review.
- `training_allowlist.jsonl`: explicit classical-language surfaces or rule constraints that may be used for tuning/augmentation.

## Policy

- Use external sources to understand rules.
- Do not copy raw external prose into training data.
- Do not train on modern explanatory Japanese.
- Use only allowlisted classical surfaces, rule constraints, or project-authored paraphrases.
- Keep source IDs for all derived cards.
