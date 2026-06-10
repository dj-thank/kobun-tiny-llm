from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "data" / "rules" / "generation_diagnostic_policy.json"


def fail(message: str) -> None:
    raise SystemExit(f"generation_diagnostic_policy_error={message}")


def main() -> None:
    if not POLICY_PATH.exists():
        fail(f"missing:{POLICY_PATH}")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy.get("release_metric") is not False:
        fail("top_level_release_metric_must_be_false")
    if policy.get("metric_role") != "diagnostic_human_llm_review_only":
        fail("metric_role_must_be_diagnostic_only")

    allowed = set(policy.get("llm_use_policy", {}).get("allowed", []))
    forbidden = set(policy.get("llm_use_policy", {}).get("forbidden", []))
    required_forbidden = {
        "training corpus text generation",
        "replacement for checkpoint-bound test_lm_token_nll",
        "replacement for split leakage checks",
        "replacement for eval contamination checks",
        "replacement for provenance and source hash audits",
    }
    missing_forbidden = sorted(required_forbidden - forbidden)
    if missing_forbidden:
        fail("missing_forbidden_llm_uses:" + ",".join(missing_forbidden))
    if not allowed:
        fail("allowed_llm_uses_empty")

    forbidden_outputs = set(policy.get("sanitization", {}).get("forbidden_in_outputs", []))
    for required in ["train split text", "validation split text", "test split text", "full training logs", "optimizer state", "run snapshots"]:
        if required not in forbidden_outputs:
            fail(f"missing_sanitization_forbidden:{required}")

    dimensions = policy.get("dimensions", [])
    if len(dimensions) < 6:
        fail("too_few_dimensions")
    dimension_ids = set()
    for dimension in dimensions:
        dimension_id = dimension.get("id")
        if not dimension_id:
            fail("dimension_missing_id")
        if dimension_id in dimension_ids:
            fail(f"duplicate_dimension_id:{dimension_id}")
        dimension_ids.add(dimension_id)
        if dimension.get("release_metric") is not False:
            fail(f"dimension_release_metric_not_false:{dimension_id}")

    probes = policy.get("probes", [])
    if len(probes) < 6:
        fail("too_few_probes")
    probe_ids = set()
    for probe in probes:
        probe_id = probe.get("id")
        if not probe_id:
            fail("probe_missing_id")
        if probe_id in probe_ids:
            fail(f"duplicate_probe_id:{probe_id}")
        probe_ids.add(probe_id)
        if probe.get("trainable") is not False:
            fail(f"probe_trainable_not_false:{probe_id}")
        if probe.get("provenance") != "project-authored diagnostic probe":
            fail(f"probe_bad_provenance:{probe_id}")
        prompt = probe.get("prompt", "")
        if not isinstance(prompt, str) or not (2 <= len(prompt) <= 80):
            fail(f"probe_prompt_length_out_of_bounds:{probe_id}")
        unknown_dimensions = sorted(set(probe.get("dimension_ids", [])) - dimension_ids)
        if unknown_dimensions:
            fail(f"probe_unknown_dimensions:{probe_id}:{','.join(unknown_dimensions)}")

    print(
        "generation_diagnostic_policy_ok=true "
        f"dimensions={len(dimensions)} probes={len(probes)} "
        "release_metric=false llm_training_text=false"
    )


if __name__ == "__main__":
    main()
