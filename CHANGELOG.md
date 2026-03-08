# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
