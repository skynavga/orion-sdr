# API Reference

## Python API

All classes live in the flat `orion_sdr` namespace.

### Class Summary

#### Analog

| Class | Direction | Constructor |
|---|---|---|
| `CwEnvelopeDemod` | IQ → audio | `(sample_rate, tone_hz, env_bw_hz)` |
| `AmEnvelopeDemod` | IQ → audio | `(fs, audio_bw_hz, abs_approx=False)` |
| `SsbProductDemod` | IQ → audio | `(fs, bfo_hz, audio_bw_hz)` |
| `FmQuadratureDemod` | IQ → audio | `(fs, dev_hz, audio_bw_hz)` |
| `PmQuadratureDemod` | IQ → audio | `(fs, k, audio_bw_hz)` |
| `AmDsbMod` | audio → IQ | `(fs, rf_hz, carrier_level, modulation_index)` |
| `CwKeyedMod` | audio → IQ | `(sample_rate, tone_hz, rise_ms, fall_ms)` |
| `FmPhaseAccumMod` | audio → IQ | `(sample_rate, deviation_hz, rf_hz)` |
| `PmDirectPhaseMod` | audio → IQ | `(sample_rate, kp_rad_per_unit, rf_hz)` |
| `SsbPhasingMod` | audio → IQ | `(fs, audio_bw_hz, audio_if_hz, rf_hz, usb)` |

#### Digital

| Class | Direction | Constructor | Notes |
|---|---|---|---|
| `BpskMod` | bits → IQ | `(fs, rf_hz, gain)` | 1 bit/symbol |
| `QpskMod` | bits → IQ | `(fs, rf_hz, gain)` | 2 bits/symbol; input length must be even |
| `QamMod` | bits → IQ | `(order, fs, rf_hz, gain)` | order ∈ {16, 64, 256}; 4/6/8 bits/symbol |
| `BpskDemod` | IQ → bits | `(gain)` | 1 bit/symbol out |
| `QpskDemod` | IQ → bits | `(gain)` | 2 bits/symbol out |
| `QamDemod` | IQ → bits | `(order, gain)` | order ∈ {16, 64, 256}; raises `ValueError` otherwise |

Digital classes fuse the mapper/decider and waveform stage into a single `process()` call.
Input bits are `uint8` arrays (one bit per byte, LSB used). Output IQ is `complex64`; output bits are `uint8`.

### Array Types

| Domain | dtype | Notes |
|---|---|---|
| IQ | `numpy.ndarray[complex64]` | 1-D, C-contiguous |
| Audio | `numpy.ndarray[float32]` | 1-D, C-contiguous |

A wrong `dtype` or non-contiguous layout raises `ValueError`.

## Rust API

The Rust API is built around the `Block` trait from `src/core.rs`.
See [design.md](design.md) for the trait definition and [source.md](source.md)
for the full module layout.

### Graph Schedulers

| Type | Input | Output |
|---|---|---|
| `IqToAudioChain` | `Complex32` | `f32` |
| `IqToIqChain` | `Complex32` | `Complex32` |
| `AudioToIqChain` | `f32` | `Complex32` |
| `BasicChain` | generic | generic |

### DSP Primitives

| Type | Description |
|---|---|
| `AgcRms` / `AgcRmsIq` | RMS-based automatic gain control |
| `DcBlocker` | 1st-order high-pass (y = x − x₁ + r·y₁) |
| `FirDecimator` | FIR anti-alias + integer decimation |
| `FirLowpass` | FIR low-pass filter |
| `Biquad` | Transposed Direct Form II biquad |
| `LpCascade` | Two cascaded biquads (4th-order) |
| `LpDcCascade` | Fused `LpCascade` + `DcBlocker` |
| `Nco` | Numerically controlled oscillator (phasor recurrence) |
| `Rotator` | Continuous phase rotator |

### Analog Modulators / Demodulators

| Type | Description |
|---|---|
| `AmDsbMod` | Full-carrier AM (A3E) modulator |
| `CwKeyedMod` | CW keyed modulator with rise/fall shaping |
| `FmPhaseAccumMod` | Phase-accumulator FM modulator (phasor recurrence) |
| `PmDirectPhaseMod` | Direct-phase PM modulator |
| `SsbPhasingMod` | Weaver/phasing-method SSB modulator |
| `AmEnvelopeDemod` | AM envelope detector (PowerSqrt or AbsApprox) |
| `CwEnvelopeDemod` | CW tone envelope demodulator |
| `FmQuadratureDemod` | Quadrature FM discriminator (`atan2_approx`) |
| `PmQuadratureDemod` | Quadrature PM demodulator |
| `SsbProductDemod` | SSB product detector with BFO |

### Digital Modulators / Demodulators

| Type | Description |
|---|---|
| `BpskMapper` | u8 bits → C32 symbols (1 bit/symbol) |
| `BpskMod` | C32 symbols → C32 IQ (carrier upconversion) |
| `BpskDemod` | C32 IQ → C32 soft symbols |
| `BpskDecider` | C32 soft symbols → u8 bits |
| `QpskMapper` | u8 bits → C32 symbols (2 bits/symbol, Gray-coded, 1/√2 normalized) |
| `QpskMod` | C32 symbols → C32 IQ |
| `QpskDemod` | C32 IQ → C32 soft symbols |
| `QpskDecider` | C32 soft symbols → u8 bits (2 per symbol) |
| `QamMapper<BITS>` | u8 bits → C32 symbols; BITS ∈ {4,6,8} for 16/64/256-QAM |
| `QamMod` | C32 symbols → C32 IQ (order-independent) |
| `QamDemod` | C32 IQ → C32 soft symbols (order-independent) |
| `QamDecider<BITS>` | C32 soft symbols → u8 bits; BITS/2 I bits + BITS/2 Q bits per symbol |
| `Qam16Mapper` / `Qam16Decider` | Type aliases for `QamMapper<4>` / `QamDecider<4>` |
| `Qam64Mapper` / `Qam64Decider` | Type aliases for `QamMapper<6>` / `QamDecider<6>` |
| `Qam256Mapper` / `Qam256Decider` | Type aliases for `QamMapper<8>` / `QamDecider<8>` |
