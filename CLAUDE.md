# CLAUDE.md — orion-sdr

## Project

`orion-sdr` is a composable SDR/DSP library in Rust (crate name `orion-sdr`, lib name `sdr`, v0.0.12), edition 2024, targeting HF–UHF signal processing with planned Python bindings via PyO3.

## Source Layout

```
src/
  lib.rs              — crate root, public API re-exports
  core.rs             — Block trait, WorkReport, chain schedulers
                        (IqToAudioChain, IqToIqChain, AudioToIqChain, BasicChain)
  util.rs             — rms, tone, snr_db_at, atan2_approx, run_block helpers
  dsp/
    agc.rs            — AgcRms, AgcRmsIq
    dc.rs             — DcBlocker (1st-order HP: y = x - x1 + r·y1)
    decim.rs          — FirDecimator
    fir.rs            — FirLowpass
    iir.rs            — Biquad, LpCascade, LpDcCascade
    nco.rs            — Nco, mix_with_nco
    rotator.rs        — Rotator
  demodulate/
    am.rs             — AmEnvelopeDemod (PowerSqrt, AbsApprox)
    cw.rs             — CwEnvelopeDemod
    fm.rs             — FmQuadratureDemod
    pm.rs             — PmQuadratureDemod
    ssb.rs            — SsbProductDemod
  modulate/
    am.rs             — AmDsbMod
    cw.rs             — CwKeyedMod
    fm.rs             — FmPhaseAccumMod
    pm.rs             — PmDirectPhaseMod
    ssb.rs            — SsbPhasingMod
  tests/
    unit.rs           — unit tests
    roundtrip.rs      — mod→demod SNR roundtrip tests
    throughput.rs     — throughput benchmarks (feature-gated)
```

## Build and Test Commands

```bash
# Standard tests (always run before committing)
cargo test --release

# Throughput benchmarks
cargo test --release --features throughput -- --nocapture --test-threads=1

# Set a minimum throughput floor (Msps) for benchmark assertions
ORION_SDR_THROUGHPUT_MINSPS=50 cargo test --release --features throughput -- --nocapture --test-threads=1
```

Always use `--release` for throughput runs. Debug builds are ~10× slower and not representative.

## Key Design Patterns

### Block trait
All DSP nodes implement `Block` from `src/core.rs`:
```rust
fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport
```
`WorkReport` carries `{ in_read, out_written }`.

### Inner-loop style
Performance-critical `process` methods use:
- `#[inline(always)]` on the method
- Manual 4× unroll (`nn = n & !3`) with a scalar tail
- `mul_add` for FMA opportunities
- No intermediate `Vec` allocations inside the loop

### IIR filter structure
- `Biquad` — Transposed Direct Form II (TDF-II), two state vars `z1`/`z2`
- `LpCascade` — two cascaded Biquads (4th-order Butterworth / Linkwitz-Riley)
- `LpDcCascade` — fused `LpCascade` + `DcBlocker` in one struct with all 5 state vars inline; used by AM and SSB demod paths. `process_mapped(x, f)` applies a function (e.g. `f32::sqrt`) between the LP and DC stages, used by AM-PowerSqrt.

### atan2 approximation
`util::atan2_approx` — 5th-order minimax polynomial, max error ≈ 0.0005 rad. Used by FM and PM demodulators instead of `f32::atan2`.

### NCO / phasor recurrence
`Nco` and `FmPhaseAccumMod` use phasor multiplication (`z *= phasor`) instead of per-sample `cos`/`sin`.

## Throughput Reference (Apple M2 Pro, release, no SIMD) — 9-run mean ±stdev

| Mode         | Msps    |
|--------------|---------|
| CW           | 149 ±3  |
| AM-AbsApprox | 149 ±4  |
| AM-PowerSqrt | 147 ±2  |
| PM           | 127 ±3  |
| FM           | 117 ±3  |
| SSB-USB      | 117 ±4  |

## Coding Conventions

- Rust edition 2024; `opt-level=3`, `lto=fat`, `codegen-units=1` in release
- `f32` throughout (no `f64`)
- No `unsafe`; no SIMD intrinsics yet (guarded behind `simd` feature)
- Throughput tests are feature-gated: `#[cfg(feature = "throughput")]` in `src/tests/mod.rs`
- Benchmark assertions use `ORION_SDR_THROUGHPUT_MINSPS` env var as the floor (default per-test minimums are intentionally conservative)
- Keep `CLAUDE.md`, `README.md`, and `memory/MEMORY.md` throughput tables in sync when numbers change
