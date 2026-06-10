# Kobun Tiny LLM

Kobun Tiny LLM is a small research codebase for training a GPT-style language
model on classical Japanese text from scratch. The project is intentionally
modest: it is designed to make the training pipeline, tokenizer policy, source
provenance, evaluation checks, and release gates inspectable.

The repository currently publishes source code, metadata, rule tables, compact
evaluation cases, and governance tooling. It does not publish model weights,
training logs, raw corpora, derived training corpora, optimizer states, or
release packages.

## Project Status

There is no public release checkpoint in this repository yet. Existing local
training attempts are treated as internal evidence only. A public model release
requires a fresh supervised run, checkpoint-bound evaluation, clean provenance
evidence, and a manual export decision.

## Repository Layout

```text
src/kobun_llm/       Core LLM, tokenizer, generation, grammar, and training code
src/kobun_autonomy/  Typed contracts for autonomous governance and release gates
scripts/            Data preparation, evaluation, release-gate, and supervisor tools
data/               Public metadata, rule tables, source manifests, and eval cases
docs/               Public architecture, data policy, and audit notes
```

The separation between `kobun_llm` and `kobun_autonomy` is deliberate. The LLM
package should remain usable without the autonomous governance layer. Governance
code can observe and classify runs, but it must not become part of the model
runtime.

## Install

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

For a CPU-only smoke test:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\kobun_llm\model.py src\kobun_llm\tokenizer.py src\kobun_llm\train.py
```

## Training Boundary

Training data is not bundled in this public repository. The scripts preserve the
project's provenance and split policy, but users must reconstruct or provide
copyable source text under the documented data policy before training.

The model is intended to be trained from scratch. The project does not require
OpenAI APIs, hosted model outputs, or pretrained weights for the training corpus.

## Evaluation Boundary

The repository includes compact evaluation cases and rule tables that are safe
to inspect publicly. Release-quality evidence must be checkpoint-bound and must
include:

- independent test language-model loss
- grammar, morphology, and waka smoke/regression checks
- train/eval overlap checks
- split leakage checks
- source provenance hashes
- tokenizer scope checks

Validation loss alone is not treated as release evidence.

## Autonomous Governance

The autonomous layer is a supervisor and evidence system, not the model. It can:

- build an evaluation board
- classify run state
- recommend the next safe action
- enforce no-export defaults
- prepare sanitized review packets

It must not:

- train on LLM-generated corpus text
- silently export, package, or upload a model
- substitute reviewer text for checkpoint-bound metrics
- hide release blockers behind generated summaries

See [Autonomy Architecture](docs/AUTONOMY_ARCHITECTURE.md).

## Public Data Policy

This repository tracks metadata and small evaluation fixtures. It intentionally
does not track raw or cleaned corpus text. Source manifests describe where
public/copyable text came from, but users are responsible for respecting the
license terms of each upstream source.

See [Data and Release Policy](docs/DATA_AND_RELEASE_POLICY.md).

## Security

Do not commit credentials, local machine paths, private notes, raw logs, model
checkpoints, optimizer state, generated release packages, or personal browser
state. See [SECURITY.md](SECURITY.md).

## License

Code is licensed under the terms in [LICENSE](LICENSE). Source texts referenced
by manifests may have their own terms; consult the original sources before
redistribution.
