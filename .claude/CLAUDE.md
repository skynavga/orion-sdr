<!--
  Copyright (c) 2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

<!-- markdownlint-disable MD013 -->

# CLAUDE.md — orion-sdr

## Project

`orion-sdr` is a composable SDR/DSP library in Rust ed. 2024, targeting HF-through-EHF signal processing with Python bindings via PyO3.

## Project Docs

- [API reference](docs/api.md) — Python API class summary
- [Build and test commands](docs/commands.md) — cargo aliases, maturin
- [Conventions](docs/conventions.md) — code and documentation (language, safety, feature flags, sync rules, et al)
- [Demodulator usage](docs/demodulate.md) — usage patterns for all demodulators
- [Design patterns](docs/design.md) — Block trait, inner-loop style, IIR structure
- [DVB-T / NB-DVB-T design](docs/dvb.md) — DVB-T 2K structure, pilots, TPS, GI acquisition, TS payload
- [Features](docs/features.md) — capability list: blocks, chains, modulators/demodulators, codecs
- [Modulator usage](docs/modulate.md) — usage patterns for all modulators
- [OFDM / COFDM design](docs/ofdm.md) — OFDM PHY conventions and the coded frame (MAC) layer
- [Performance benchmarks](docs/performance.md) — throughput results, SNR curves, and how to run them
- [Python bindings](docs/python.md) — PyO3 native extension: installation, classes, usage
- [Source layout](docs/source.md) — module tree
- [Terminology](docs/terminology.md) — acronyms and glossary

## Git

Except for the staging and commit described in the `/release-prep` skill, all stagings and commits are performed by the User: stage, run `/commit-message`, review, then commit. Don't stage or commit on the User's behalf unless explicitly asked.

Never include Claude trailers (`Co-Authored-By` or similar) anywhere: commit messages, PR/release descriptions, GitHub issue/PR bodies and comments, etc.

## Agent skills

Issues live as GitHub issues in this repo, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

Default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

Single-context: `CONTEXT.md` + `docs/adr/`. See `docs/agents/domain.md`.
