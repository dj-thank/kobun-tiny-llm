# Autonomy Architecture

The autonomous architecture is a governance system around training. It is not an
LLM and it is not part of the model runtime.

## Typed Contracts

Public contracts live in `src/kobun_autonomy/types.py`.

Important contracts:

- `TrainingLogStatus`: parsed view of a local training log
- `RunClassification`: internal classification for one run
- `BoardRunRow`: public row written to the evaluation board
- `HealthStatus`: active-lock, lease, and startup health signals
- `AutonomousAction`: next action selected by the supervisor
- `EvaluationBoard`: top-level board payload

The contracts are intentionally JSON-shaped `TypedDict` definitions. This keeps
existing evidence files compatible while making the governance boundary visible
to type checkers and reviewers.

## Decision Flow

```text
logs/checkpoints -> run classifier -> evaluation board -> action selector
```

The selector may return actions such as:

- monitor an active run
- run post-run quality checks
- stop a clearly obsolete non-release run
- fix blockers before continuing
- prepare a fresh supervised run after gates pass
- report upload-ready evidence without exporting

## Non-Negotiables

The autonomous layer must fail closed when required evidence is missing. It must
not treat generated summaries, reviewer comments, or validation loss as a
substitute for checkpoint-bound release evidence.

HF export and public upload remain manual decisions. Automation can prepare
evidence, but it cannot publish a model by default.
