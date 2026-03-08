# Demodulator Usage

Usage patterns and examples for all demodulators in **orion-sdr**: CW, AM, SSB, FM, PM.

All examples assume IQ samples as `Vec<num_complex::Complex32>` and show a minimal
chain using `IqToAudioChain`. Adjust sample rates, bandwidths, and gains to your setup.

## CW Demodulation (Envelope)

Extract a CW tone at a chosen audio pitch (e.g., 600–800 Hz) from complex baseband IQ.

```rust
use orion_sdr::{
    core::IqToAudioChain,
    demodulate::CwEnvelopeDemod,
    dsp::{FirDecimator, AgcRms},
};
use num_complex::Complex32 as C32;

let fs = 48_000.0;
let pitch_hz = 700.0;
let audio_bw_hz = 300.0;

let mut chain = IqToAudioChain::new(CwEnvelopeDemod::new(fs, pitch_hz, audio_bw_hz));

// Optional: decimate IQ before demod to save CPU
let m = 2;
let cutoff = (fs / m as f32) * 0.45;
let trans  = (fs / m as f32) * 0.10;
chain.push_iq(FirDecimator::new(fs, m, cutoff, trans));

// Optional: audio AGC
chain.push_audio(AgcRms::new(fs, 0.2, 5.0, 200.0));

let iq: Vec<C32> = get_iq_block();
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
let audio_bw_hz = 5_000.0;

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
let bfo_hz = 0.0;
let audio_bw_hz = 2_800.0;

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

Phase-difference quadrature discriminator. Audio is scaled so roughly ±deviation → ±1.0.

```rust
use orion_sdr::{
    core::IqToAudioChain,
    demodulate::FmQuadratureDemod,
    dsp::AgcRms,
};
use num_complex::Complex32 as C32;

let fs = 48_000.0;
let dev_hz = 2_500.0;
let audio_bw_hz = 5_000.0;

let mut chain = IqToAudioChain::new(FmQuadratureDemod::new(fs, dev_hz, audio_bw_hz));

// Optional: de-emphasis (300–750 µs NBFM voice; 75 µs US WBFM; 50 µs EU WBFM)
// chain.demod_mut().set_deemph_tau_us(300.0);

// Optional: post audio AGC
chain.push_audio(AgcRms::new(fs, 0.2, 5.0, 200.0));

let iq: Vec<C32> = get_iq_block();
let audio = chain.process(iq);
```

## PM Demodulation (Quadrature)

Instantaneous phase (with unwrap). Set `pm_sense_rad` so that expected phase deviation
maps to ~±1.0 audio.

```rust
use orion_sdr::{
    core::IqToAudioChain,
    demodulate::PmQuadratureDemod,
};
use num_complex::Complex32 as C32;

let fs = 48_000.0;
let pm_sense_rad = 0.8;
let audio_bw_hz = 5_000.0;

let mut chain = IqToAudioChain::new(PmQuadratureDemod::new(fs, pm_sense_rad, audio_bw_hz));

// Optional: disable amplitude limiter
// chain.demod_mut().set_limiter(false);

let iq: Vec<C32> = get_iq_block();
let audio = chain.process(iq);
```

## Tips

- **Center your signal** in the complex baseband before demod (DDC/tuning not shown here).
- **Decimate early** to reduce CPU, but design anti-aliasing correctly (`FirDecimator`
  cutoff/transition relative to the *post-decim* rate).
- **AGC placement:** try IQ-domain AGC (before demod) *or* audio-domain AGC (after demod)
  depending on your preference and mode.
- **FM de-emphasis:** speech NBFM often benefits from 300–750 µs; broadcast WBFM uses
  75 µs (US) or 50 µs (EU).
- **Block sizes:** feed consistent chunk sizes to keep latency predictable.
