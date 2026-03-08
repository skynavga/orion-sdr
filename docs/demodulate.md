# Demodulator Usage

Usage patterns and examples for all demodulators in **orion-sdr**: CW, AM, SSB, FM, PM,
BPSK, QPSK, QAM-16/64/256.

Analog demodulator examples assume IQ samples as `Vec<num_complex::Complex32>` and show
a minimal chain using `IqToAudioChain`. Adjust sample rates, bandwidths, and gains to
your setup.

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

## Tips (analog modes)

- **Center your signal** in the complex baseband before demod (DDC/tuning not shown here).
- **Decimate early** to reduce CPU, but design anti-aliasing correctly (`FirDecimator`
  cutoff/transition relative to the *post-decim* rate).
- **AGC placement:** try IQ-domain AGC (before demod) *or* audio-domain AGC (after demod)
  depending on your preference and mode.
- **FM de-emphasis:** speech NBFM often benefits from 300–750 µs; broadcast WBFM uses
  75 µs (US) or 50 µs (EU).
- **Block sizes:** feed consistent chunk sizes to keep latency predictable.

---

## Digital Demodulators

All digital demodulation pipelines use two stages:

1. A **soft-symbol estimator** (`BpskDemod`, `QpskDemod`, or `QamDemod`) applies gain
   normalization and passes the complex sample through — it is the coherent decision
   metric before slicing.
2. A **hard decider** (`BpskDecider`, `QpskDecider`, or `QamDecider<BITS>`) slices the
   soft symbol into bits and writes one bit per output byte (value 0 or 1 in the LSB).

Both stages implement `Block` and can be driven directly or wrapped in `IqToIqChain` /
`IqToAudioChain` as needed.

### BPSK

```rust
use orion_sdr::{
    core::Block,
    demodulate::{BpskDemod, BpskDecider},
};
use num_complex::Complex32 as C32;

let iq: Vec<C32> = receive_iq_block();        // coherent, carrier-removed baseband
let mut soft     = vec![C32::default(); iq.len()];
let mut bits_out = vec![0u8; iq.len()];

BpskDemod::new(1.0).process(&iq, &mut soft);
BpskDecider::new().process(&soft, &mut bits_out);
// bits_out[i] ∈ {0, 1}: Re(soft) ≥ 0 → 0, Re(soft) < 0 → 1
```

### QPSK

Each input symbol produces two output bytes: `bits_out[2k]` (I decision) and
`bits_out[2k+1]` (Q decision).

```rust
use orion_sdr::{
    core::Block,
    demodulate::{QpskDemod, QpskDecider},
};
use num_complex::Complex32 as C32;

let iq: Vec<C32> = receive_iq_block();
let n = iq.len();
let mut soft     = vec![C32::default(); n];
let mut bits_out = vec![0u8; n * 2];

QpskDemod::new(1.0).process(&iq, &mut soft);
QpskDecider::new().process(&soft, &mut bits_out);
```

### QAM-16 / QAM-64 / QAM-256

`QamDecider<BITS>` emits `BITS` output bytes per input symbol, laid out as `BITS/2`
I-axis bits followed by `BITS/2` Q-axis bits (MSB-first within each axis Gray index),
exactly mirroring the `QamMapper<BITS>` input layout.

Type aliases `Qam16Decider`, `Qam64Decider`, `Qam256Decider` are provided.

```rust
use orion_sdr::{
    core::Block,
    demodulate::{QamDemod, Qam16Decider, Qam64Decider, Qam256Decider},
};
use num_complex::Complex32 as C32;

let iq: Vec<C32> = receive_iq_block();
let n = iq.len();
let mut soft = vec![C32::default(); n];
QamDemod::new(1.0).process(&iq, &mut soft);

// QAM-16: 4 bits/symbol
let mut bits16 = vec![0u8; n * 4];
Qam16Decider::new().process(&soft, &mut bits16);

// QAM-64: 6 bits/symbol
let mut bits64 = vec![0u8; n * 6];
Qam64Decider::new().process(&soft, &mut bits64);

// QAM-256: 8 bits/symbol
let mut bits256 = vec![0u8; n * 8];
Qam256Decider::new().process(&soft, &mut bits256);
```

### Carrier removal

The soft-symbol stages expect carrier-removed baseband IQ.  If the received signal is
at an IF, down-mix first using `Rotator`:

```rust
use orion_sdr::{
    core::Block,
    dsp::Rotator,
    demodulate::{BpskDemod, BpskDecider},
};
use num_complex::Complex32 as C32;

let fs = 2_400_000.0;
let if_hz = 100_000.0;
let mut rot = Rotator::new(-if_hz, fs);   // negative frequency → down-mix

let iq_if: Vec<C32>  = receive_iq_block();
let mut iq_bb  = vec![C32::default(); iq_if.len()];
let mut soft   = vec![C32::default(); iq_if.len()];
let mut bits   = vec![0u8; iq_if.len()];

rot.rotate_block(&iq_if, &mut iq_bb);
BpskDemod::new(1.0).process(&iq_bb, &mut soft);
BpskDecider::new().process(&soft, &mut bits);
```

### Notes

- **This is v1 — no timing or carrier recovery.**  The soft-symbol stages are coherent
  passthrough blocks.  Symbol timing must be established externally (1 sample per symbol,
  correct sampling instant).  Carrier phase must be removed before the demod stage.
- **Gain normalization:** pass `gain = 1.0` when the received symbols are already
  normalized.  Adjust when the channel introduces amplitude scaling.
- **Soft decisions for FEC:** `BpskDemod` / `QpskDemod` / `QamDemod` output the raw
  `Complex32` metric before slicing — feed `Re(soft)` (and `Im(soft)` for QPSK/QAM)
  directly into a soft-decision Viterbi or LDPC decoder when one is available.
