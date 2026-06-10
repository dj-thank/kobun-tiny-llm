# Data and Release Policy

This repository is designed to be safe to publish as source code. It intentionally
does not include raw corpus text, cleaned corpus text, derived training text,
model checkpoints, optimizer states, raw logs, or release packages.

## What Is Tracked

- source metadata and provenance manifests
- compact evaluation fixtures
- grammar and waka rule tables authored for this project
- project-authored external-knowledge cards and surface-pattern constraints
- tokenizer public vocabulary metadata
- code for rebuilding, checking, and evaluating local artifacts

Tracked rule tables and compact evaluation examples are public source assets for
testing the pipeline. They are not redistributed upstream corpus text and should
not be treated as a substitute for rebuilding the training corpus from licensed
sources.

## What Is Not Tracked

- raw source downloads
- cleaned source text
- train, validation, and test corpora
- generated training snapshots
- run logs
- checkpoints
- release bundles
- private notes

## Source Responsibility

Some source manifests point to public or copyable classical Japanese texts. Those
manifests are not a license grant. Anyone rebuilding the corpus must check and
respect the license terms of each upstream source.

## Release Evidence

A release candidate must provide checkpoint-bound evidence:

- exact best checkpoint identity
- independent test loss
- source and split hashes
- tokenizer scope checks
- overlap/leakage checks
- grammar, morphology, and waka checks
- explicit no-export default

Upload-ready evidence means ready for a later manual release decision. It does
not mean the model has been exported or uploaded.
