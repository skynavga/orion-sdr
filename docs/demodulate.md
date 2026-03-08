# Demodulator Usage

Usage patterns and examples for all demodulators in **orion-sdr**: CW, AM, SSB, FM, PM,
BPSK, QPSK, QAM-16/64/256.

Analog demodulator examples assume IQ samples as `Vec<num_complex::Complex32>` and show
a minimal chain using `IqToAudioChain`. Adjust sample rates, bandwidths, and gains to
your setup.

## CW Demodulation (Envelope)

Extract a CW tone at a chosen audio pitch (e.g., 600–800 Hz) from complex baseband IQ.
`audio_bw_hz` sets the one-pole envelope LP bandwidth (e.g. 300 Hz for narrow CW).

```rust
use orion_sdr::{
    core::IqToAudioChain,
    demodulate::CwEnvelopeDemod,
};
use num_complex::Complex32 as C32;

let fs = 48_000.0;
let pitch_hz = 700.0;
let audio_bw_hz = 300.0;

let mut chain = IqToAudioChain::new(CwEnvelopeDemod::new(fs, pitch_hz, audio_bw_hz));

let iq: Vec<C32> = get_iq_block();
let audio: Vec<f32> = chain.process(iq);
```

## AM Demodulation (Envelope)

Envelope detector with fused 4th-order LP and DC blocker.  Default method is
`PowerSqrt` (√(I²+Q²)); call `.with_abs_approx(0.9475, 0.3925)` on the demod
for a faster abs-approximation variant.

```rust
use orion_sdr::{
    core::IqToAudioChain,
    demodulate::AmEnvelopeDemod,
};
use num_complex::Complex32 as C32;

let fs = 48_000.0;
let audio_bw_hz = 5_000.0;

let mut chain = IqToAudioChain::new(AmEnvelopeDemod::new(fs, audio_bw_hz));

let iq: Vec<C32> = get_iq_block();
let audio = chain.process(iq);
```

## SSB Demodulation (Product)

Product detector with BFO and fused 4th-order LP + DC blocker.  Set `bfo_hz` to
the carrier offset of the signal within the baseband (0.0 if already centred).

```rust
use orion_sdr::{
    core::IqToAudioChain,
    demodulate::SsbProductDemod,
};
use num_complex::Complex32 as C32;

let fs = 48_000.0;
let bfo_hz = 0.0;
let audio_bw_hz = 2_800.0;

let mut chain = IqToAudioChain::new(SsbProductDemod::new(fs, bfo_hz, audio_bw_hz));

let iq: Vec<C32> = get_iq_block();
let audio = chain.process(iq);
```

## FM Demodulation (Quadrature)

Phase-difference quadrature discriminator using `atan2_approx`.  Audio is scaled
so ±`dev_hz` of frequency deviation maps to roughly ±1.0.  Use `.with_translate(freq_hz)`
to shift the signal before demodulation when it is not already centred.

```rust
use orion_sdr::{
    core::IqToAudioChain,
    demodulate::FmQuadratureDemod,
};
use num_complex::Complex32 as C32;

let fs = 48_000.0;
let dev_hz = 2_500.0;
let audio_bw_hz = 5_000.0;

let mut chain = IqToAudioChain::new(FmQuadratureDemod::new(fs, dev_hz, audio_bw_hz));

let iq: Vec<C32> = get_iq_block();
let audio = chain.process(iq);
```

## PM Demodulation (Quadrature)

Phase-difference quadrature discriminator, scaled by `k`.  Set `k` to match the
modulator's `kp_rad_per_unit` so that the expected phase deviation maps to ~±1.0 audio.

```rust
use orion_sdr::{
    core::IqToAudioChain,
    demodulate::PmQuadratureDemod,
};
use num_complex::Complex32 as C32;

let fs = 48_000.0;
let k = 0.8;            // match PmDirectPhaseMod kp_rad_per_unit
let audio_bw_hz = 5_000.0;

let mut chain = IqToAudioChain::new(PmQuadratureDemod::new(fs, k, audio_bw_hz));

let iq: Vec<C32> = get_iq_block();
let audio = chain.process(iq);
```

## Tips (analog modes)

- **Center your signal** before demodulation.  Use `Rotator` for DDC/frequency
  translation, or `FmQuadratureDemod::with_translate(freq_hz)` for FM.
- **Decimation:** run `FirDecimator` on the IQ stream before the demod block to
  reduce CPU, sizing cutoff and transition relative to the post-decimation rate.
- **AGC:** apply `AgcRmsIq` to the IQ stream before demod, or `AgcRms` to the audio
  output after demod — both are available in `orion_sdr::dsp`.
- **FM de-emphasis:** no built-in de-emphasis filter yet; apply a first-order IIR
  post-demod (τ = 75 µs for US WBFM, 50 µs for EU WBFM, 300–750 µs for NBFM voice).
- **Block sizes:** feed consistent chunk sizes to keep filter state and latency predictable.

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
