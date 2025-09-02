# orion-sdr

A composable SDR/DSP library in Rust with Python bindings.

## Change Log

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

## Features (as of v0.0.9)

- Core traits and runner ✅
- Basic, IQ->IQ, IQ->Audio, Audio->IQ graph schedulers ✅
- NCO, Tone generator, FIR low pass, DC blocker, FIR decimator, AGC ✅
- CW, AM, SSB, FM, PM modulators and demodulators ✅
- Basic DSP and Demod tests ✅

## Next Milestones

- PyO3 binding for SSB and simple Python process ✅
- Expose full pipeline via Python, record/replay, UI, etc.

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

let mut chain = IqToAudioChain::new(PmQuadratureDemod::new(pm_sense_rad, audio_bw_hz, fs));

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
