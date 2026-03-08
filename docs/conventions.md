# Coding Conventions

- Rust edition 2024; `opt-level=3`, `lto=fat`, `codegen-units=1` in release
- `f32` throughout (no `f64`)
- No `unsafe`; no SIMD intrinsics yet (guarded behind `simd` feature)
- Throughput tests are feature-gated: `#[cfg(feature = "throughput")]` in `src/tests/mod.rs`
- Benchmark assertions use `ORION_SDR_THROUGHPUT_MINSPS` env var as the floor
  (default per-test minimums are intentionally conservative)
- When throughput numbers change, update `docs/throughput.md` and `memory/MEMORY.md`
