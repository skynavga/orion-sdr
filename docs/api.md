# API Reference

## Python API

All classes live in the flat `orion_sdr` namespace.

### Class Summary

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
