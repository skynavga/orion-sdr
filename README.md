# orion-sdr

A composable SDR/DSP library in Rust with Python bindings.

## Change Log

- v0.0.12: more optimizations on all mod/demod paths; add claude configuration, update readme
- v0.0.11: optimize AM mod and demod path; add throughput results to readme
- v0.0.10: optimize SSB mod and demod path; add throughput results to readme
- v0.0.9: subdivide modulator and demodulator code into per-mode modules
- v0.0.8: add CW, SSB, FM, PM modulators; reorganize source into module tree
- v0.0.7: use {Mode}{Approach}{Demod|Mod} name convention; add Audio to IQ chain, AM modulator with tests
- v0.0.6: add FM and PM demodulators with tests; update changelog, readme
- v0.0.5: add graph scheduler; AGC, FIR decimator; CW and AM demodulators with tests
- v0.0.4: update roadmap
- v0.0.3: update description
- v0.0.2: add API implementation and basic test
- v0.0.1: placeholder API, project structure, roadmap

## Status

- Pre-alpha

## Features (as of v0.0.12)

- Core traits and runner ✅
- Basic, IQ->IQ, IQ->Audio, Audio->IQ graph schedulers ✅
- NCO, Phase Rotator, IIR  FIR low pass, DC blocker, FIR decimator, AGC, IIR cascade ✅
- CW, AM, SSB, FM, PM modulators and demodulators ✅
- Unit, roundtrip, and throughput tests ✅

## Next Milestones

- Expose full pipeline via Python, record/replay, UI, etc.

## Throughput Test Results (v0.0.12, optimized release build)

The following results were obtained using `cargo test --release --features throughput -- --nocapture` on Apple M2 Pro Silicon (sans SIMD), averaged over 9 runs:

| Mode         | Throughput (Msps) |
|--------------|-------------------|
| CW           | 149 ±3            |
| AM-AbsApprox | 149 ±4            |
| AM-PowerSqrt | 147 ±2            |
| PM           | 127 ±3            |
| FM           | 117 ±3            |
| SSB-USB      | 117 ±4            |

# Demodulator Usage

Below are usage patterns and examples for all demodulators currently available in **orion-sdr**: **CW, AM, SSB, FM, PM**.

All examples assume you have IQ samples as `Vec<num_complex::Complex32>` and show a minimal chain using `IqToAudioChain`. Adjust sample rates, bandwidths, and gains to your setup.

## CW Demodulation (Envelope)

Extract a CW tone at a chosen audio pitch (e.g., 600–800 Hz) from complex baseband IQ.

```rust
use orion_sdr::{
    core::IqToAudioChain,
    demodulate::CwEnvelopeDemod,
    dsp::{FirDecimator, AgcRms},
};
use num_complex::Complex32 as C32;

// IQ sample rate
let fs = 48_000.0;

// CW audio pitch & bandwidth
let pitch_hz = 700.0;
let audio_bw_hz = 300.0;

let mut chain = IqToAudioChain::new(CwEnvelopeDemod::new(fs, pitch_hz, audio_bw_hz));

// Optional: decimate IQ before demod to save CPU (design passband/transition for post-decim BW)
let m = 2; // decimate by 2
let cutoff = (fs / m as f32) * 0.45;
let trans  = (fs / m as f32) * 0.10;
chain.push_iq(FirDecimator::new(fs, m, cutoff, trans));

// Optional: audio AGC
chain.push_audio(AgcRms::new(fs, 0.2, 5.0, 200.0));

// Run
let iq: Vec<C32> = get_iq_block(); // your source of IQ samples
let audio: Vec<f32> = chain.process(iq);
```

## AM Demodulation (Envelope)

Simple envelope detector with post low-pass and DC removal.

```rust
use orion_sdr::{
    core::IqToAudioChain,
    demodulate::AmEnvelopeDemod,
    dsp::AgcRms,
};
use num_complex::Complex32 as C32;

let fs = 48_000.0;
let audio_bw_hz = 5_000.0; // narrow AM voice; raise for wider audio

let mut chain = IqToAudioChain::new(AmEnvelopeDemod::new(fs, audio_bw_hz));
chain.push_audio(AgcRms::new(fs, 0.2, 10.0, 300.0));

let iq: Vec<C32> = get_iq_block();
let audio = chain.process(iq);
```

## SSB Demodulation (Product)

Product detector with BFO; set BFO frequency and audio bandwidth to taste.

```rust
use orion_sdr::{
    core::IqToAudioChain,
    demodulate::SsbProductDemod,
    dsp::{FirDecimator, AgcRms},
};
use num_complex::Complex32 as C32;

let fs = 48_000.0;
let bfo_hz = 0.0;           // 0 = audio centered; use +/- offset to choose LSB/USB by tuning
let audio_bw_hz = 2_800.0;  // typical SSB audio bandwidth

let mut chain = IqToAudioChain::new(SsbProductDemod::new(fs, bfo_hz, audio_bw_hz));

// Optional: decimate IQ first
let m = 2;
let cutoff = (fs / m as f32) * 0.45;
let trans  = (fs / m as f32) * 0.10;
chain.push_iq(FirDecimator::new(fs, m, cutoff, trans));

// Optional: audio AGC
chain.push_audio(AgcRms::new(fs, 0.2, 5.0, 200.0));

let iq: Vec<C32> = get_iq_block();
let audio = chain.process(iq);
```

## FM Demodulation (Quadrature)

Phase-difference quadrature discriminator. Optional limiter and de-emphasis. Audio is scaled so roughly ±deviation → ±1.0.

```rust
use orion_sdr::{
    core::IqToAudioChain,
    demodulate::FmQuadratureDemod,
    dsp::AgcRms,
};
use num_complex::Complex32 as C32;

let fs = 48_000.0;         // IQ sample rate
let dev_hz = 2_500.0;      // peak deviation (e.g., 2.5k or 5k for NBFM)
let audio_bw_hz = 5_000.0; // post-demod audio low-pass

let mut chain = IqToAudioChain::new(FmQuadratureDemod::new(fs, dev_hz, audio_bw_hz));

// Optional: enable de-emphasis (try 300–750 µs for NBFM voice; 75 µs US WBFM, 50 µs EU WBFM)
// chain.demod_mut().set_deemph_tau_us(300.0);

// Optional: post audio AGC
chain.push_audio(AgcRms::new(fs, 0.2, 5.0, 200.0));

let iq: Vec<C32> = get_iq_block();
let audio = chain.process(iq);
```

## PM Demodulation (Quadrature)

Instantaneous phase (with unwrap). Set `pm_sense_rad` so that your expected phase deviation maps to ~±1.0 audio.

```rust
use orion_sdr::{
    core::IqToAudioChain,
    demodulate::PmQuadratureDemod,
};
use num_complex::Complex32 as C32;

let fs = 48_000.0;
let pm_sense_rad = 0.8;     // radians peak phase deviation → ~±1.0 audio
let audio_bw_hz = 5_000.0;

let mut chain = IqToAudioChain::new(PmQuadratureDemod::new(fs, pm_sense_rad, audio_bw_hz));

// Optional: disable amplitude limiter on the demod if desired
// chain.demod_mut().set_limiter(false);

let iq: Vec<C32> = get_iq_block();
let audio = chain.process(iq);
```

## Tips

- **Center your signal** in the complex baseband before demod (DDC/tuning not shown here).
- **Decimate early** to reduce CPU, but design anti-aliasing correctly (`FirDecimator` cutoff/transition relative to the *post-decim* rate).
- **AGC placement:** try IQ-domain AGC (before demod) *or* audio-domain AGC (after demod) depending on your preference and mode.
- **FM de-emphasis:** speech NBFM often benefits from 300–750 µs; broadcast WBFM uses 75 µs (US) or 50 µs (EU).
- **Block sizes:** feed consistent chunk sizes to keep latency predictable.

# Python Bindings

All 5 demodulators and all 5 modulators are available as a native Python extension via PyO3.

## Installation

Build and install the wheel with [maturin](https://www.maturin.rs/):

```bash
pip install maturin
maturin build --release
pip install target/wheels/orion_sdr-*.whl
```

Or, with an active virtualenv, install directly in editable/develop mode:

```bash
maturin develop --release
```

## API Overview

All classes live in the flat `sdr` namespace.

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

Input/output array types:
- IQ: `numpy.ndarray` with `dtype=complex64`
- Audio: `numpy.ndarray` with `dtype=float32`

Arrays must be 1-D and C-contiguous. A wrong dtype or non-contiguous layout raises `ValueError`.

## Demodulator Examples

```python
import sdr
import numpy as np

# Synthetic IQ block (replace with real samples from your SDR)
iq = np.zeros(4096, dtype=np.complex64)

# CW envelope demodulator — tracks amplitude of a tone near 700 Hz
cw = sdr.CwEnvelopeDemod(sample_rate=48_000, tone_hz=700, env_bw_hz=300)
cw.set_gain(1.5)
audio = cw.process(iq)   # → float32 ndarray, same length as input

# AM envelope demodulator (PowerSqrt, higher fidelity)
am = sdr.AmEnvelopeDemod(fs=48_000, audio_bw_hz=5_000)
audio = am.process(iq)

# AM envelope demodulator (AbsApprox, slightly faster, small amplitude error)
am_fast = sdr.AmEnvelopeDemod(fs=48_000, audio_bw_hz=5_000, abs_approx=True)
audio = am_fast.process(iq)

# SSB product demodulator — BFO at 0 Hz for a pre-tuned baseband signal
ssb = sdr.SsbProductDemod(fs=48_000, bfo_hz=0.0, audio_bw_hz=2_800)
audio = ssb.process(iq)

# NBFM quadrature demodulator — 2.5 kHz peak deviation, 5 kHz audio BW
fm = sdr.FmQuadratureDemod(fs=48_000, dev_hz=2_500, audio_bw_hz=5_000)
audio = fm.process(iq)

# PM quadrature demodulator — sensitivity k scales phase difference to audio
pm = sdr.PmQuadratureDemod(fs=48_000, k=0.8, audio_bw_hz=5_000)
audio = pm.process(iq)
```

## Modulator Examples

```python
import sdr
import numpy as np

audio = np.zeros(4096, dtype=np.float32)

# AM DSB modulator — full carrier (A3E), 80% modulation index, baseband output
am_mod = sdr.AmDsbMod(fs=48_000, rf_hz=0.0, carrier_level=1.0, modulation_index=0.8)
am_mod.set_gain(1.0)
am_mod.set_clamp(True)   # clamp modulated envelope to ±1
iq = am_mod.process(audio)   # → complex64 ndarray

# CW keyed modulator — input is a 0..1 keying envelope (not raw audio)
cw_mod = sdr.CwKeyedMod(sample_rate=48_000, tone_hz=700, rise_ms=5.0, fall_ms=5.0)
key = np.ones(4096, dtype=np.float32)   # key-down
iq = cw_mod.process(key)

# FM phase-accumulator modulator — 2.5 kHz deviation, baseband output
fm_mod = sdr.FmPhaseAccumMod(sample_rate=48_000, deviation_hz=2_500, rf_hz=0.0)
fm_mod.set_deviation(5_000)   # change deviation after construction
iq = fm_mod.process(audio)

# PM direct-phase modulator — kp maps ±1 audio to ±kp radians of phase
pm_mod = sdr.PmDirectPhaseMod(sample_rate=48_000, kp_rad_per_unit=0.8, rf_hz=0.0)
pm_mod.set_sensitivity(1.0)
iq = pm_mod.process(audio)

# SSB phasing modulator — USB, audio IF at 1.5 kHz, baseband RF output
ssb_mod = sdr.SsbPhasingMod(fs=48_000, audio_bw_hz=2_800, audio_if_hz=1_500,
                             rf_hz=0.0, usb=True)
iq = ssb_mod.process(audio)
```

## Round-trip Example

```python
import sdr
import numpy as np

fs = 48_000
n  = 8192

# Generate a 1 kHz test tone
t = np.arange(n, dtype=np.float32) / fs
audio_in = np.sin(2 * np.pi * 1_000 * t)

# Modulate → demodulate (SSB USB)
mod   = sdr.SsbPhasingMod(fs=fs, audio_bw_hz=2_800, audio_if_hz=1_500,
                           rf_hz=0.0, usb=True)
demod = sdr.SsbProductDemod(fs=fs, bfo_hz=0.0, audio_bw_hz=2_800)

iq        = mod.process(audio_in)
audio_out = demod.process(iq)

print(audio_in.shape, audio_out.shape, audio_out.dtype)
# (8192,) (8192,) float32
```

## Notes

- All `process()` calls are synchronous and hold the GIL. For high-throughput pipelines, call from a dedicated thread or release the GIL at a higher level.
- Each instance is stateful (filter state, NCO phase, AGC level). Create separate instances for independent signal paths.
- `rf_hz=0.0` produces baseband (DC-centered) IQ. Set a non-zero value to upconvert in one step.
- `CwKeyedMod` expects a keying envelope (0 = key up, 1 = key down), not audio. Derive the envelope from PTT state or a CW decoder.
