from __future__ import annotations

from typing import Any, Literal, TypedDict


Backend = Literal["dml", "cuda", "unknown"]
RunStatus = Literal[
    "unknown",
    "running",
    "completed",
    "completed_unverified",
    "stale_incomplete",
    "launch_incomplete",
    "internal_evidence",
]
ReleaseStatus = Literal[
    "internal_failtest_evidence",
    "needs_post_run_quality",
    "non_release_artifact",
    "not_upload_ready",
    "training_active_or_unverified",
    "training_active_overfit_stop_recommended",
    "training_active_superseded_non_release",
    "upload_ready_not_exported",
]
RunNextAction = Literal[
    "await_explicit_hf_export_request",
    "fix_blockers_then_rerun_checks",
    "ignore_for_release",
    "investigate_failed_or_stuck_launch",
    "investigate_missing_train_exit_sentinel",
    "monitor",
    "run_or_fix_release_gate",
    "run_post_run_quality_checks",
    "stop_overfit_run",
    "supersede_non_release_run",
]
BoardActionName = Literal[
    "fix_blockers",
    "monitor",
    "prepare_next_fresh_run_after_static_gate_and_zero_base_reviews",
    "run_post_run_quality_checks",
    "stop_and_report_upload_ready",
    "stop_overfit_run",
    "supersede_non_release_run",
]


class StepSnapshot(TypedDict):
    step: int
    train_loss: float
    val_loss: float


class TrainingLogConfig(TypedDict, total=False):
    block_size: int
    device: str
    params: int
    vocab_size: int


class TrainingLogStatus(TypedDict, total=False):
    exists: bool
    path: str
    path_name: str
    steps: list[StepSnapshot]
    latest: StepSnapshot | None
    best: StepSnapshot | None
    completed: bool
    stop_reason: str
    non_improving_evals: int
    train_val_gap: float | None
    best_val_regression: float | None
    overfit_stop_signal: bool
    last_write_time: str
    config: TrainingLogConfig


class EvalMetrics(TypedDict, total=False):
    test_lm_token_nll: float | None
    grammar_score: float | None
    waka_score: float | None
    waka_static_score: float | None
    waka_generation_score: float | None
    morphology_score: float | None
    eval_contamination_hits: int | None
    eval_source_overlap_hits: int | None
    split_leaks: int | None
    waka_leaks: int | None
    tokenizer_leakage: int | None


class GovernanceMetrics(TypedDict, total=False):
    source_quality_present: bool
    source_quality_average: float | None
    source_quality_hard_blocker_rows: int | None
    generation_diagnostic_policy_present: bool
    generation_diagnostic_release_metric: bool | None
    generation_diagnostic_plan_present: bool


class RunClassification(EvalMetrics, GovernanceMetrics, total=False):
    run_id: str
    status: RunStatus | str
    backend: Backend | str
    params: int | None
    vocab_size: int | None
    block_size: int | None
    best_step: int | None
    best_val_loss: float | None
    latest_step: int | None
    latest_val_loss: float | None
    checkpoint: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    eval_json: str
    train_exit_sentinel: str
    active_lock: str
    active_lock_live: bool
    release_gate_verified: bool
    release_gate_error: str
    release_status: ReleaseStatus | str
    upload_ready: bool
    overall_score: float
    hard_blockers: list[str]
    soft_warnings: list[str]
    active_policy_issues: list[str]
    next_action: RunNextAction | str
    training_log: TrainingLogStatus
    stderr_bytes: int


class BoardRunRow(EvalMetrics, GovernanceMetrics, total=False):
    run_id: str
    status: RunStatus | str
    backend: Backend | str
    params: int | None
    vocab_size: int | None
    block_size: int | None
    best_step: int | None
    best_val_loss: float | None
    latest_step: int | None
    latest_val_loss: float | None
    checkpoint_sha256: str
    release_status: ReleaseStatus | str
    upload_ready: bool
    overall_score: float
    hard_blockers: list[str]
    soft_warnings: list[str]
    active_policy_issues: list[str]
    next_action: RunNextAction | str


class LeaseSummary(TypedDict, total=False):
    path: str
    run_id: str
    state: str
    train_pid: int
    launcher_pid: int
    train_live: bool
    launcher_live: bool
    lease_expires_at_utc: str
    artifact_root: str


class HealthStatus(TypedDict, total=False):
    exists: bool
    invalid_json: bool
    active: list[LeaseSummary]
    quarantined: list[str]
    errors: list[str]
    hard_blockers: list[str]


class ReleaseEvidenceState(TypedDict, total=False):
    state: str
    meaning: str
    post_run_required: list[str]


class AutonomousAction(TypedDict, total=False):
    action: BoardActionName | str
    run_id: str
    reason: str
    hard_blockers: list[str]
    overall_score: float | None
    details: dict[str, Any]


class EvaluationBoard(TypedDict, total=False):
    generated_at_utc: str
    project: str
    hf_export_policy: str
    runs: list[BoardRunRow]
    active_lock_health: HealthStatus
    colab_cuda_lease_health: HealthStatus
    startup_mutex_health: HealthStatus
    global_blockers: list[str]
    next_action: AutonomousAction
    governance_phase: str
    release_evidence_state: ReleaseEvidenceState
