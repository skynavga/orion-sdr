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

### Digital Modulation Pipeline

All digital modes follow a two-stage split that separates symbol mapping from waveform
generation, keeping each stage independently reusable:

```
[u8 bits] → Mapper → [C32 symbols] → Mod → [C32 IQ]
[C32 IQ]  → Demod  → [C32 soft]   → Decider → [u8 bits]
```

| Mode     | Bits/sym | Levels/axis | Mapper input          | Decider output  |
|----------|----------|-------------|-----------------------|-----------------|
| BPSK     | 1        | 2 (1D)      | 1 × u8 (LSB)          | 1 × u8          |
| QPSK     | 2        | 2           | 2 × u8 (LSBs)         | 2 × u8          |
| QAM-16   | 4        | 4           | 4 × u8 (LSBs)         | 4 × u8          |
| QAM-64   | 6        | 8           | 6 × u8 (LSBs)         | 6 × u8          |
| QAM-256  | 8        | 16          | 8 × u8 (LSBs)         | 8 × u8          |

**Gray coding.** Both axes of every constellation are independently Gray-coded
(`g → g ^ (g >> 1)`), so adjacent constellation points differ by exactly one bit.
This minimises bit errors under noise without requiring FEC.

**Normalization.** All mappers normalise to unit average symbol energy.  For M levels
per axis the scale factor is `1 / √(2(M²−1)/3)`:

| Mode    | M  | Scale (≈)       |
|---------|----|-----------------|
| BPSK    | —  | 1.0             |
| QPSK    | 2  | 1/√2 ≈ 0.7071   |
| QAM-16  | 4  | 1/√10 ≈ 0.3162  |
| QAM-64  | 8  | 1/√42 ≈ 0.1543  |
| QAM-256 | 16 | 1/√170 ≈ 0.0767 |

**QAM const-generic design.** `QamMapper<const BITS: usize>` and
`QamDecider<const BITS: usize>` are instantiated at compile time for `BITS ∈ {4, 6, 8}`.
The mapper stores a `[f32; 16]` amplitude table (stack-only); the decider stores a
`[f32; 15]` threshold table.  Both are populated in `new()` from `axis_scale(BITS)`.
The waveform stages (`QamMod`, `QamDemod`) are order-independent C32→C32 blocks shared
across all orders.

**Waveform stages.** `BpskMod`, `QpskMod`, and `QamMod` multiply each symbol by a
carrier phasor from an internal `Rotator` (phasor recurrence, no per-sample trig).
Setting `rf_hz = 0.0` gives baseband passthrough.

## Throughput

See [throughput.md](throughput.md) for benchmark results and how to run them.
