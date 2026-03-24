# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.0.22] - 2026-03-24

### Changed

- Replaced Hann-weighted integrate-and-dump (final quarter) in `Bpsk31Demod` and
  `Qpsk31Demod` with decision-feedback matched filtering over the full 256-sample
  symbol period: `corrected[n] = s[n] − prev_sym·(1−h[n])`, `sym = Σ h[n]·corrected[n] / Σ h[n]²`
- Improved sensitivity by ~1–2 dB: BPSK31 100% decode at −5 dB, QPSK31 at +13 dB SNR/2500 Hz
- Updated CI regression thresholds and `docs/performance.md` SNR table accordingly

### Added

- `HalfCosineMf` in `src/dsp/fir.rs`: complex-split FIR with unit-energy half-cosine taps,
  exported as `orion_sdr::dsp::HalfCosineMf`

## [0.0.21] - 2026-03-24

### Changed

- Replaced peak sampling in `Bpsk31Demod` and `Qpsk31Demod` with Hann-weighted
  integrate-and-dump over the final quarter of each symbol period (`n ∈
  [3·sps/4, sps)`), improving sensitivity by ~18 dB for both modes
- Updated CI regression thresholds: BPSK31 −4 dB, QPSK31 +14 dB SNR/2500 Hz
  (previously +14 dB and +32 dB); QPSK31 now correctly outperforms BPSK31
- Updated PSK31 SNR sweep ranges and `docs/performance.md` sensitivity table
  to reflect the improved demodulator

## [0.0.20] - 2026-03-24

### Added

- PSK31 SNR sensitivity characterisation: 50-trial Monte Carlo sweep for BPSK31
  and QPSK31, feature-gated (`src/tests/performance/snr/psk31.rs`)
- PSK31 SNR CI regression tests: fixed thresholds at 100% success level (BPSK31
  +14 dB, QPSK31 +32 dB, SNR/2500 Hz) (`src/tests/roundtrip/psk31_snr.rs`)
- SNR sensitivity table and explanatory notes added to `docs/performance.md`

## [0.0.19] - 2026-03-24

### Added

- PSK31 Python and Rust API reference in `docs/api.md`; fixed all compact table
  separator rows in the file to comply with MD060.

## [0.0.18] - 2026-03-23

### Added

- PSK31 full stack (BPSK31 + QPSK31) at 31.25 baud:
  - Varicode codec: IZ8BLY/G3PLX canonical table, `VaricodeEncoder`, `VaricodeDecoder`
    (`src/codec/varicode.rs`)
  - Convolutional codec: rate-1/2 K=5 (G0=25, G1=23) encoder and soft Viterbi decoder
    (`src/codec/psk31_conv.rs`)
  - BPSK31/QPSK31 modulators with Hann-windowed half-cosine crossfade pulse shaping
    (`src/modulate/psk31.rs`)
  - BPSK31/QPSK31 demodulators using peak-sampling differential detection
    (`src/demodulate/psk31.rs`)
  - Waterfall-based energy-persistence carrier sync (`psk31_sync`,
    `src/sync/psk31_sync.rs`)
  - PyO3 bindings: `VaricodeEncoder`, `VaricodeDecoder`, `Bpsk31Mod`, `Bpsk31Demod`,
    `Qpsk31Mod`, `Qpsk31Demod`, `psk31_sync` (`src/python/psk31.rs`)
  - 122 Rust tests (unit, roundtrip, throughput) and 20 new Python tests

## [0.0.17] - 2026-03-23

### Changed

- Synchronized release-prep and release skills from orion-sdr-view: added branch
  guard, PR-merge workflow, and removed co-author trailer from release commits
- Updated GitHub Actions workflow to use Node.js 24

## [0.0.16] - 2026-03-15

### Added

- FT8/FT4 full stack: CPFSK waveform mod/demod, CRC-14 + LDPC(174,91) + Gray codec,
  Costas-array frame sync with soft LLR extraction, and 77-bit message packing
  (standard Type 1/2, free text, telemetry, non-standard callsigns)
- Python bindings for the complete FT8/FT4 stack: `Ft8Mod`, `Ft8Demod`, `Ft8Codec`,
  `Ft4Mod`, `Ft4Demod`, `Ft4Codec`, `ft8_sync`, `ft4_sync`, `ft8_pack_standard`,
  `ft8_pack_free_text`, `ft8_pack_telemetry`, `ft8_unpack`
- SNR sensitivity tests: sweeping characterisation (feature-gated, always passes,
  prints curve) and fixed CI thresholds (FT8 −15 dB, FT4 −11 dB, SNR/2500 Hz)
- `performance/` test module replacing `throughput/`, with `throughput/` and `snr/`
  subdirectories; `cargo test-throughput` alias updated accordingly
- Acronym glossary in `docs/design.md`

### Changed

- `pyo3` and `numpy` are now optional dependencies activated only by the
  `extension-module` feature; `cargo test --release` no longer requires a Python
  interpreter at build time
- `docs/throughput.md` renamed to `docs/performance.md`; all references updated

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
