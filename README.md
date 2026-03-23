# orion-sdr

A composable SDR/DSP library in Rust with Python bindings.

## Status

Pre-alpha (v0.0.17). See [CHANGELOG.md](CHANGELOG.md) for release history.

## Next Milestones

- Expose full pipeline via Python, record/replay, UI, etc.

## Documentation

- [Features](docs/features.md) — what's implemented as of v0.0.17
- [Build and test commands](docs/commands.md) — cargo aliases, maturin
- [Source layout](docs/source.md) — module tree
- [Design patterns](docs/design.md) — Block trait, inner-loop style, IIR structure
- [Coding conventions](docs/conventions.md) — language, safety, feature flags
- [Performance benchmarks](docs/performance.md) — throughput results, SNR curves, and how to run them
- [Demodulator usage](docs/demodulate.md) — Rust examples for CW, AM, SSB, FM, PM demodulators
- [Modulator usage](docs/modulate.md) — Rust examples for CW, AM, SSB, FM, PM modulators
- [Python bindings](docs/python.md) — installation, type stubs, Python examples and round-trip demo
- [API reference](docs/api.md) — class summary, array types, graph schedulers, DSP primitives
