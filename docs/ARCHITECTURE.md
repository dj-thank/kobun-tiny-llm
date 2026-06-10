# Architecture

Kobun Tiny LLM is split into two layers:

- `kobun_llm`: model runtime, tokenizer, generation, grammar helpers, training
  loop, checkpoint IO, and release-policy primitives.
- `kobun_autonomy`: typed contracts for governance artifacts such as evaluation
  board rows, autonomous actions, health status, and run classifications.

The model package should not depend on the autonomous package. That keeps the
LLM usable as a normal Python package and prevents release governance from
becoming hidden model behavior.

## Core Model Layer

The core model layer owns:

- tokenizer construction and byte-fallback decoding
- GPT-style model definition
- training loop and checkpoint writing
- local generation utilities
- grammar and waka helper rules used by evaluation or decoding tools

The model layer does not decide whether a checkpoint is public-release quality.

## Governance Layer

The governance layer owns:

- run discovery
- active-lock health checks
- evaluation-board schema
- autonomous next-action selection
- no-export release defaults
- sanitized review/evidence packet boundaries

The governance layer may recommend actions. It does not silently export,
package, upload, or publish model artifacts.

## Scripts

The `scripts/` directory bridges the two layers. Some scripts call the model for
training or evaluation; other scripts read logs and artifacts to classify release
readiness. New scripts should make that boundary explicit in their names and
imports.
