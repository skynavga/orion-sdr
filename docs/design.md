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

```text
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

See [performance.md](performance.md) for benchmark results and how to run them.

---

## Acronym Glossary

| Acronym | Expansion | Notes |
| ------- | --------- | ----- |
| AGC | Automatic Gain Control | `AgcRms`, `AgcRmsIq` in `dsp/agc.rs` |
| AM | Amplitude Modulation | DSB (double-sideband) variant implemented |
| AWGN | Additive White Gaussian Noise | Standard noise model used in tests |
| BP | Belief Propagation | Iterative sum-product algorithm used in LDPC decoder |
| BPSK | Binary Phase-Shift Keying | 1 bit/symbol |
| CLT | Central Limit Theorem | Used in AWGN generation (sum-of-uniforms approximation) |
| CPFSK | Continuous-Phase Frequency-Shift Keying | Phase continuity across symbol boundaries; used by FT8/FT4 |
| CRC | Cyclic Redundancy Check | CRC-14 (poly 0x2757) used by FT8/FT4 |
| CW | Continuous Wave | Morse-code keyed carrier |
| DC | Direct Current | Zero-frequency component; blocked by `DcBlocker` |
| DSB | Double-Sideband | Both sidebands transmitted; see AM |
| DSP | Digital Signal Processing | — |
| FEC | Forward Error Correction | LDPC is the FEC scheme in FT8/FT4 |
| FIR | Finite Impulse Response | `FirLowpass`, `FirDecimator` in `dsp/` |
| FM | Frequency Modulation | Quadrature (discriminator) demod |
| FMA | Fused Multiply-Add | `f32::mul_add`; used throughout inner loops |
| FSK | Frequency-Shift Keying | Base modulation for FT8 (8-FSK) and FT4 (4-FSK) |
| FT4 | Fast Telegraphy 4-FSK | 4-FSK weak-signal mode; 6-second transmit period |
| FT8 | Fast Telegraphy 8-FSK | 8-FSK weak-signal mode; 15-second transmit period |
| HF | High Frequency | 3–30 MHz; primary target band for FT8/FT4 |
| IF | Intermediate Frequency | `rf_hz` parameter in modulators |
| IIR | Infinite Impulse Response | `Biquad`, `LpCascade` in `dsp/` |
| IQ | In-phase / Quadrature | Complex baseband representation; `Complex32` throughout |
| LDPC | Low-Density Parity-Check | LDPC(174,91) code shared by FT8 and FT4 |
| LLR | Log-Likelihood Ratio | `log(P(bit=0)/P(bit=1))`; positive ↔ bit more likely 0 |
| LO | Local Oscillator | Receiver frequency reference; source of frequency offset |
| LP | Low-Pass | `FirLowpass`, `LpCascade` filter types |
| NCO | Numerically Controlled Oscillator | `Nco` in `dsp/nco.rs`; phasor recurrence |
| PM | Phase Modulation | Quadrature (dφ) demod |
| QAM | Quadrature Amplitude Modulation | 16/64/256-QAM implemented |
| QPSK | Quadrature Phase-Shift Keying | 2 bits/symbol |
| RF | Radio Frequency | Upconverted (non-baseband) signal |
| RMS | Root Mean Square | Used by AGC and test SNR helpers |
| SDR | Software-Defined Radio | — |
| SNR | Signal-to-Noise Ratio | Expressed in dB throughout |
| SSB | Single-Sideband | Phasing (Weaver) modulator; product demodulator |
| TDF-II | Transposed Direct Form II | Biquad filter state-variable structure |
| UHF | Ultra High Frequency | 300 MHz–3 GHz; secondary target band |
