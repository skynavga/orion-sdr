<!--
  Copyright (c) 2025-2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

# orion-sdr

A composable SDR/DSP library in Rust with Python bindings, targeting HF
through EHF signal processing — analog modes, single-carrier digital modes,
FT8/FT4, PSK31, and OFDM.

## Status

Pre-alpha (v0.0.43). See [CHANGELOG.md](CHANGELOG.md) for release history.

## Next Milestones

- DFT-s-OFDM (SC-FDMA) and OTFS, built on the same `multicarrier/`
  resource-grid foundation OFDM introduced

## Documentation

- [Features](docs/features.md) — what's implemented as of v0.0.43
- [Build and test commands](docs/commands.md) — cargo aliases, maturin
- [Source layout](docs/source.md) — module tree
- [Design patterns](docs/design.md) — Block trait, inner-loop style, IIR structure, multicarrier/OFDM pipeline
- [Coding conventions](docs/conventions.md) — language, safety, feature flags
- [Performance benchmarks](docs/performance.md) — throughput results, SNR curves, and how to run them
- [Demodulator usage](docs/demodulate.md) — Rust examples for CW/AM/SSB/FM/PM, BPSK/QPSK/QAM, FT8/FT4, and OFDM demodulators
- [Modulator usage](docs/modulate.md) — Rust examples for CW/AM/SSB/FM/PM, BPSK/QPSK/QAM, FT8/FT4, and OFDM modulators
- [Python bindings](docs/python.md) — installation, type stubs, Python examples and round-trip demo
- [API reference](docs/api.md) — class summary, array types, graph schedulers, DSP primitives
