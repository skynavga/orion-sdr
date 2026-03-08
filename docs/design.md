# Design Patterns

## Key Design Patterns

### Block Trait

All DSP nodes implement `Block` from `src/core.rs`:

```rust
fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport
```

`WorkReport` carries `{ in_read, out_written }`.

### Inner-Loop Style

Performance-critical `process` methods use:

- `#[inline(always)]` on the method
- Manual 4× unroll (`nn = n & !3`) with a scalar tail
- `mul_add` for FMA opportunities
- No intermediate `Vec` allocations inside the loop

### IIR Filter Structure

- `Biquad` — Transposed Direct Form II (TDF-II), two state vars `z1`/`z2`
- `LpCascade` — two cascaded Biquads (4th-order Butterworth / Linkwitz-Riley)
- `LpDcCascade` — fused `LpCascade` + `DcBlocker` in one struct with all 5 state vars
  inline; used by AM and SSB demod paths. `process_mapped(x, f)` applies a function
  (e.g. `f32::sqrt`) between the LP and DC stages, used by AM-PowerSqrt.

### atan2 Approximation

`util::atan2_approx` — 5th-order minimax polynomial, max error ≈ 0.0005 rad.
Used by FM and PM demodulators instead of `f32::atan2`.

### NCO / Phasor Recurrence

`Nco` and `FmPhaseAccumMod` use phasor multiplication (`z *= phasor`) instead of
per-sample `cos`/`sin`.

## Throughput

See [throughput.md](throughput.md) for benchmark results and how to run them.
