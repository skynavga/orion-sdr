<!--
  Copyright (c) 2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

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

## Multicarrier / OFDM / COFDM

The OFDM physical layer (FFT normalization, carrier indexing, numerology, CFO
acquisition, channel estimation, the block-boundary contract) and the coded,
framed COFDM link built on it (concatenated FEC, interleaver domains, the
frame format, and the streaming receiver) have their own document:
[ofdm.md](ofdm.md).

## Throughput

See [performance.md](performance.md) for benchmark results and how to run them.

---

## Acronym Glossary

| Acronym | Expansion | Notes |
| ------- | --------- | ----- |
| AFC | Automatic Frequency Control | PSK31's symbol-rate decision-directed PLL, tracks residual carrier drift |
| AGC | Automatic Gain Control | `AgcRms`, `AgcRmsIq` in `dsp/agc.rs` |
| AM | Amplitude Modulation | DSB (double-sideband) variant implemented |
| AWGN | Additive White Gaussian Noise | Standard noise model used in tests |
| BCH | Bose–Chaudhuri–Hocquenghem | Binary block code (`fec/bch.rs`); COFDM outer FEC option |
| BFO | Beat Frequency Oscillator | `bfo_hz` parameter in `SsbProductDemod` |
| BM | Berlekamp–Massey | Error-locator algorithm in the BCH/RS decoders |
| BP | Belief Propagation | Iterative sum-product algorithm used in LDPC decoders |
| BPSK | Binary Phase-Shift Keying | 1 bit/symbol; the COFDM header's fixed modulation |
| CFO | Carrier Frequency Offset | TX/RX oscillator mismatch; corrected by `Rotator` before OFDM demod |
| CLT | Central Limit Theorem | Used in AWGN generation (sum-of-uniforms approximation) |
| COFDM | Coded OFDM | The concatenated-FEC, framed OFDM link; see [ofdm.md](ofdm.md) |
| CP | Cyclic Prefix | `CyclicPrefixInsert`/`CyclicPrefixRemove`; absorbs multipath delay spread up to `cp_len` |
| CPFSK | Continuous-Phase Frequency-Shift Keying | Phase continuity across symbol boundaries; used by FT8/FT4 |
| CRC | Cyclic Redundancy Check | CRC-14 (0x2757) for FT8/FT4; generic CRC-16/CRC-32 (`fec`) for COFDM frames |
| CW | Continuous Wave | Morse-code keyed carrier |
| DBPSK | Differential Binary Phase-Shift Keying | PSK31's BPSK31 modulation (differential detection, no absolute-phase reference) |
| DC | Direct Current | Zero-frequency component; blocked by `DcBlocker`; implicitly null in OFDM carrier plans |
| DQPSK | Differential Quadrature Phase-Shift Keying | PSK31's QPSK31 modulation, decoded via soft Viterbi |
| DSB | Double-Sideband | Both sidebands transmitted; see AM |
| DSP | Digital Signal Processing | — |
| EHF | Extremely High Frequency | 30–300 GHz; upper end of the OFDM target-band range |
| EVM | Error Vector Magnitude | Soft-vs-ideal constellation distance, in dB; `OfdmRxFrame::evm_db` |
| FEC | Forward Error Correction | LDPC in FT8/FT4; COFDM concatenates an inner (LDPC/convolutional) and outer (BCH/RS) code (`fec/`) |
| FIR | Finite Impulse Response | `FirLowpass`, `FirDecimator` in `dsp/` |
| GF | Galois Field | `Gf256` = GF(2^8) arithmetic (`fec/gf.rs`), foundation for BCH/RS |
| FM | Frequency Modulation | Quadrature (discriminator) demod |
| FMA | Fused Multiply-Add | `f32::mul_add`; used throughout inner loops |
| FSK | Frequency-Shift Keying | Base modulation for FT8 (8-FSK) and FT4 (4-FSK) |
| FT4 | Fast Telegraphy 4-FSK | 4-FSK weak-signal mode; 6-second transmit period |
| FT8 | Fast Telegraphy 8-FSK | 8-FSK weak-signal mode; 15-second transmit period |
| HF | High Frequency | 3–30 MHz; primary target band for FT8/FT4 |
| IF | Intermediate Frequency | `rf_hz` parameter in modulators |
| IIR | Infinite Impulse Response | `Biquad`, `LpCascade` in `dsp/` |
| IQ | In-phase / Quadrature | Complex baseband representation; `Complex32` throughout |
| LDPC | Low-Density Parity-Check | LDPC(174,91) in FT8/FT4 (`codec/ldpc.rs`); parameterized family (`fec/ldpc_codes.rs`) for COFDM |
| LEO | Low Earth Orbit | High-Doppler satellite case motivating OFDM's opt-in `PerSymbolPilotInterp` equalizer |
| LFSR | Linear-Feedback Shift Register | Basis of the `PnScrambler` whitener (`fec/scrambler.rs`) |
| LLR | Log-Likelihood Ratio | `log(P(bit=0)/P(bit=1))`; positive ↔ bit more likely 0 |
| LO | Local Oscillator | Receiver frequency reference; source of frequency offset |
| LP | Low-Pass | `FirLowpass`, `LpCascade` filter types |
| MAC | Medium Access Control | The COFDM frame layer (`FramePacket`, `OfdmFrameMod`/`OfdmFrameStreamDemod`) |
| MCS | Modulation and Coding Scheme | `McsTable` maps a per-frame index to (constellation, inner/outer FEC) |
| MLSE | Maximum-Likelihood Sequence Estimation | `viterbi_decode_coherent` in `codec/psk31.rs` |
| MMSE | Minimum Mean-Square Error | Noise-aware equalizer; a candidate remedy for high-order-QAM multipath (not yet implemented) |
| NBFM | Narrowband FM | Voice FM with a small deviation-to-audio-bandwidth ratio |
| NCO | Numerically Controlled Oscillator | `Nco` in `dsp/nco.rs`; phasor recurrence |
| OFDM | Orthogonal Frequency-Division Multiplexing | `multicarrier/` + `modulate`/`demodulate`/`sync::ofdm*`; VHF–EHF target bands |
| OTFS | Orthogonal Time Frequency Space | Planned future `multicarrier/`-based waveform |
| PLL | Phase-Locked Loop | PSK31's AFC loop is a first-order decision-directed PLL |
| PM | Phase Modulation | Quadrature (dφ) demod |
| PN | Pseudo-Noise | Deterministic pseudo-random sequence; the `PnScrambler` whitener and S&C preamble base sequence |
| QAM | Quadrature Amplitude Modulation | 16/64/256-QAM implemented |
| QPSK | Quadrature Phase-Shift Keying | 2 bits/symbol |
| RF | Radio Frequency | Upconverted (non-baseband) signal |
| RMS | Root Mean Square | Used by AGC and test SNR helpers |
| RS | Reed–Solomon | Byte-symbol block code (`fec/reed_solomon.rs`); DVB-T RS(204,188) t=8; COFDM outer FEC option |
| RX | Receive / Receiver | — |
| S&C | Schmidl & Cox | Repeated-segment preamble algorithm used by `ofdm_sync` for timing/CFO |
| SC-FDMA | Single-Carrier Frequency-Division Multiple Access | DFT-s-OFDM; planned future `multicarrier/`-based waveform |
| SDR | Software-Defined Radio | — |
| SNR | Signal-to-Noise Ratio | Expressed in dB throughout |
| SSB | Single-Sideband | Phasing (Weaver) modulator; product demodulator |
| TDF-II | Transposed Direct Form II | Biquad filter state-variable structure |
| TX | Transmit / Transmitter | — |
| UHF | Ultra High Frequency | 300 MHz–3 GHz; secondary target band |
| VHF | Very High Frequency | 30–300 MHz; lower end of the OFDM target-band range |
| WBFM | Wideband FM | Broadcast FM with a large deviation-to-audio-bandwidth ratio |
