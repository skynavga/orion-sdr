<!--
  Copyright (c) 2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Project Conventions

## Coding

- Rust edition 2024; `opt-level=3`, `lto=fat`, `codegen-units=1` in release
- `f32` throughout (no `f64`)
- No `unsafe`; no SIMD intrinsics
- Two Cargo features: `throughput` (gates throughput/SNR-sweep tests) and
  `extension-module` (gates the PyO3 Python bindings). No waveform — including
  OFDM — is gated behind its own feature flag; new waveforms compile in
  unconditionally, matching every existing mode
- Throughput and SNR-sweep tests are feature-gated: `#[cfg(feature = "throughput")]`
  in `tests/performance.rs`
- Benchmark assertions use `ORION_SDR_THROUGHPUT_MINSPS` env var as the floor
  (default per-test minimums are intentionally conservative)
- When code changes may affect throughput, SNR, BER, FER, CBER, et al, update
  `docs/performance.md` and the project's memory notes

## Documentation

- Markdown (`*.md`) files should be lint free (fix all found issues whether or no introduced in current session)
- Use `markdownlint-cli2` found on `$PATH` with project's `.markdownlint.json` configuration
- Use bracketed disablement of MD013 for MD tables with long line lengths

## Copyright and License Notice

All code, configuration, and documentation files must contain a copyright and license
header using the pattern found in the project. Include a year (for new files) or year
range (for files created in preceding years), e.g., `2026`, `2025-2026`, updating
ranges as needed. For example

```md
<!--
  Copyright (c) 2025-2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->
```

```python
# Copyright (c) 2025-2026 G & R Associates LLC
# SPDX-License-Identifier: MIT OR Apache-2.0
```

```rust
// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0
```

and similar patterns for other file types.

## Git

### Branches

Before creating or modifying files in a Git repository, check the current branch.
If the branch is `main` and the User has not explicitly requested the change on `main`,
do not modify files; instead, propose a new branch name to User for confirmation,
then create branch off of `main` and push to remote (`origin`) for tracking.

Branch names should use the format `{feature,fix,misc}/short-description` where
`short-description` should specify the main intent of the branch using kebab (lisp)
case, e.g., `feature/add-fancy-new-feature`, with a verb-object(s) pattern.

After a branch has been merged to `main`, delete it both locally and remotely,
then run `git remote prune origin`.

### Staging and Commits

Except for the staging and commit described in the /release-prep skill, all stagings
and commits are performed by the User unless explicitly requested.

### Co-Authored-By and other Trailers

Never include Claude trailers anywhere: commit messages, PR/Release descriptions, etc.

## Plans

Per-project plan documents previously lived under `~/.claude/plans/<project>/`. However,
they are no longer created; instead of plan files, the project now uses a combination of
ADRs, specs, and tickets created by the `/grill-with-docs`, `/to-spec`, and `/to-tickets`
skills. ADRs reside in `docs/adr`, while specs and tickets are create as Github issues.
