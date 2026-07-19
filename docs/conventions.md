<!--
  Copyright (c) 2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Coding Conventions

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
- When throughput numbers change, update `docs/performance.md` and the
  project's memory notes
