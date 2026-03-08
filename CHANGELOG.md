# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.0.15] - 2026-03-08

### Added

- BPSK, QPSK, QAM-16/64/256 modulators and demodulators (Rust + PyO3 bindings)
  - `BpskMapper`, `BpskMod`, `BpskDemod`, `BpskDecider` (1 bit/symbol)
  - `QpskMapper`, `QpskMod`, `QpskDemod`, `QpskDecider` (2 bits/symbol, Gray-coded)
  - `QamMapper<BITS>`, `QamMod`, `QamDecider<BITS>`, `QamDemod` (const-generic, 4/6/8 bits/symbol, Gray-coded, unit-energy normalized)
  - Python classes: `BpskMod`, `BpskDemod`, `QpskMod`, `QpskDemod`, `QamMod`, `QamDemod`
- `IqToIqChain<B>` graph scheduler for C32→C32 pipelines
- Throughput tests for all digital modes (BPSK ~253 Msps, QPSK ~317 Msps, QAM-16 ~209 Msps, QAM-64 ~92 Msps, QAM-256 ~73 Msps)
- Python unit and roundtrip tests for all digital modes (54 tests total)
- PEP 561 type stubs for 6 new digital classes
- Release-prep and release skills (`skills/release-prep/`, `skills/release/`)

### Changed

- Docs updated to cover digital modes: `docs/features.md`, `docs/design.md`, `docs/modulate.md`, `docs/demodulate.md`, `docs/throughput.md`, `docs/python.md`
- Fixed incorrect API examples in `docs/demodulate.md` (removed fictional `push_iq`, `push_audio`, `demod_mut`, `set_deemph_tau_us`, `set_limiter` calls)

## [0.0.14] - 2026-03-08

### Added

- GitHub Actions workflow (`publish.yml`) to build and publish wheels for
  Linux (x86-64, aarch64), macOS (x86-64, arm64), and Windows (x86-64)
  on every `v*` tag push, using `maturin-action` and OIDC trusted publishing

## [0.0.13] - 2026-03-08

### Added

- Restructured docs into `docs/` directory (source, design, conventions,
  throughput, features, commands, demodulate, modulate, python, api)
- `CHANGELOG.md` with git-accurate dates in Keep a Changelog format
- Mixed maturin Python package layout (`python/orion_sdr/`) with `__init__.py`,
  `py.typed`, and type stub (moved from root `orion_sdr.pyi`)
- `pyproject.toml` with maturin config and pytest settings
- pytest test suite: unit tests (shape, dtype, validation, setters, isolation)
  and roundtrip SNR tests for all 5 modes
- Cargo test aliases (`test-unit`, `test-roundtrip`, `test-throughput`)
- `.markdownlint.json` (MD024 `siblings_only`)
- `.venv` and `*.so`/`*.pyd` added to `.gitignore`

## [0.0.12] - 2026-03-01

### Added

- PyO3 bindings for mod/demod functionality
- Python type stubs (PEP 561)
- Claude configuration (`CLAUDE.md`)

### Changed

- Package name used by Python changed to `orion-sdr` / `orion_sdr`
- More optimizations on all mod/demod paths (fused `LpDcCascade`, phasor recurrence NCO, atan2 approximation, loop unrolling)
- Updated throughput results

## [0.0.11] - 2025-09-03

### Changed

- Optimized AM mod and demod path
- Updated throughput results

## [0.0.10] - 2025-09-01

### Changed

- Optimized SSB mod and demod path
- Added throughput results to README

## [0.0.9] - 2025-09-01

### Changed

- Subdivided modulator and demodulator code into per-mode modules

## [0.0.8] - 2025-09-01

### Added

- CW, SSB, FM, PM modulators

### Changed

- Reorganized source into module tree

## [0.0.7] - 2025-08-30

### Added

- Audio to IQ chain
- AM modulator with tests

### Changed

- Adopted `{Mode}{Approach}{Demod|Mod}` naming convention

## [0.0.6] - 2025-08-30

### Added

- FM and PM demodulators with tests

### Changed

- Updated changelog and README

## [0.0.5] - 2025-08-12

### Added

- Graph scheduler
- AGC, FIR decimator
- CW and AM demodulators with tests

## [0.0.4] - 2025-08-12

### Changed

- Updated roadmap

## [0.0.3] - 2025-08-12

### Changed

- Updated description

## [0.0.2] - 2025-08-12

### Added

- API implementation and basic test

## [0.0.1] - 2025-08-12

### Added

- Initial API, project structure, roadmap
