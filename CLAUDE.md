# CLAUDE.md — orion-sdr

## Project

`orion-sdr` is a composable SDR/DSP library in Rust (crate name `orion-sdr`, lib name `orion_sdr`, v0.0.20), edition 2024, targeting HF–UHF signal processing with Python bindings via PyO3.

## Reference Docs

- [Build and test commands](docs/commands.md) — cargo aliases, maturin
- [Source layout](docs/source.md) — module tree
- [Design patterns](docs/design.md) — Block trait, inner-loop style, IIR structure
- [Coding conventions](docs/conventions.md) — language, safety, feature flags, sync rules
- [Performance benchmarks](docs/performance.md) — throughput results, SNR curves, and how to run them
