<!--
  Copyright (c) 2025-2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

# orion-sdr

A composable SDR/DSP library in Rust with Python bindings, targeting HF
through EHF signal processing — analog modes, single-carrier digital modes
(BPSK, QPSK, QAM), FT8/FT4, PSK31, OFDM, a coded and framed COFDM link,
and a conformant DVB-T.

## Status

Pre-alpha (v0.0.67). See [CHANGELOG.md](CHANGELOG.md) for release history.

## Next Milestones

- DFT-s-OFDM (SC-FDMA) and OTFS, built on the same `multicarrier/`
  resource-grid foundation OFDM introduced

## Documentation

- [Features](docs/features.md) — what's implemented
- [Build and test commands](docs/commands.md) — cargo aliases, maturin
- [Source layout](docs/source.md) — module tree
- [Design patterns](docs/design.md) — Block trait, inner-loop style, IIR structure
- [OFDM / COFDM design](docs/ofdm.md) — OFDM PHY conventions, the coded frame (MAC) layer, and out-of-band spectral shaping
- [DVB-T / NB-DVB-T design](docs/dvb.md) — DVB-T 2K structure, pilots, TPS, GI acquisition, TS payload
- [Coding conventions](docs/conventions.md) — language, safety, feature flags
- [Performance benchmarks](docs/performance.md) — throughput results, SNR curves, and how to run them
- [Demodulator usage](docs/demodulate.md) — Rust examples for CW/AM/SSB/FM/PM, BPSK/QPSK/QAM, FT8/FT4, OFDM/COFDM, and DVB-T demodulators
- [Modulator usage](docs/modulate.md) — Rust examples for CW/AM/SSB/FM/PM, BPSK/QPSK/QAM, FT8/FT4, OFDM/COFDM, and DVB-T modulators
- [Python bindings](docs/python.md) — installation, type stubs, Python examples and round-trip demo
- [API reference](docs/api.md) — class summary, array types, graph schedulers, DSP primitives
- [Terminology](docs/terminology.md) — acronym expansions and glossary terms used across the source and docs
