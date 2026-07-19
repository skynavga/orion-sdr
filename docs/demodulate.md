<!--
  Copyright (c) 2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Demodulator Usage

Usage patterns and examples for all demodulators in **orion-sdr**: CW, AM, SSB, FM, PM,
BPSK, QPSK, QAM-16/64/256, FT8/FT4, and OFDM.

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

---

## FT8 / FT4 Demodulation

FT8 and FT4 demodulation has two distinct use cases:

1. **Pre-aligned** — you have a single frame at a known time/frequency offset and
   want tone decisions. Use `Ft8Demod` / `Ft4Demod` directly.
2. **Search** — you have a raw capture and need to find frames. Use `ft8_sync` /
   `ft4_sync`, which also return soft LLRs for LDPC decoding.

### Direct demodulation (aligned input)

`Ft8Demod` uses the same Goertzel dot-product correlator as the sync waterfall but
operates on a single pre-aligned frame.  It returns hard tone decisions, which
`Ft8Codec::decode_hard` can decode directly.

```rust
use orion_sdr::demodulate::Ft8Demod;
use orion_sdr::codec::ft8::Ft8Codec;
use num_complex::Complex32 as C32;

let iq: Vec<C32> = get_ft8_frame();   // 151 680 samples, tone 0 at base_hz

let demod = Ft8Demod::new(12_000.0, 1_000.0);
if let Some(frame) = demod.demodulate(&iq) {
    if let Some(payload) = Ft8Codec::decode_hard(&frame) {
        // payload: [u8; 10]  — 77-bit message
    }
}
```

For FT4, substitute `Ft4Demod` and `Ft4Codec`; the input must be at least 60 480 samples.

### Sync search (unknown timing / frequency)

`ft8_sync` computes a symbol-rate magnitude waterfall over the supplied IQ buffer,
searches for Costas-array matches, and extracts soft LLRs for each candidate.
Pass `llr` to `Ft8Codec::decode_soft` for robust decoding in noise.

```rust
use orion_sdr::sync::ft8_sync;
use orion_sdr::codec::ft8::Ft8Codec;
use num_complex::Complex32 as C32;

let iq: Vec<C32> = get_received_block();   // arbitrary length

// Search for FT8 frames with tone-0 between 1000 and 1200 Hz,
// anywhere in the buffer, return up to 5 candidates.
let candidates = ft8_sync(
    &iq,
    12_000.0,  // fs
    1_000.0,   // base_hz
    1_200.0,   // max_hz
    0,         // t_min (symbol offset)
    0,         // t_max (0 = end of buffer)
    5,         // max_cand
);

for c in &candidates {
    if let Some(payload) = Ft8Codec::decode_soft(&c.llr) {
        println!("found: time={} freq={} score={:.1}", c.time_sym, c.freq_bin, c.score);
        // payload: [u8; 10]
    }
}
```

Use `ft4_sync` for FT4; the signature is identical.

---

## OFDM Demodulation

OFDM's RX pipeline is deliberately split into composable stages — unlike
`OfdmMod`'s fused TX chain — so [`OfdmEqualizer`](#ofdm-channel-equalization)
can be inserted, swapped, or skipped independently. `OfdmDemod` itself covers
`CyclicPrefixRemove → FftBlock → GridExtract` (known packet start, no CFO,
flat channel); wire `OfdmEqualizer` in manually for multipath channels, and
[`ofdm_sync`](#ofdm-packet-sync--cfo-acquisition) first when the packet start
and carrier offset aren't already known.

### Known-start, flat-channel demodulation

```rust
use orion_sdr::{
    core::Block,
    demodulate::{OfdmDecider, OfdmDemod},
    modulate::{ConstellationOrder, OfdmConfig, OfdmMod},
    multicarrier::CarrierPlan,
};
use num_complex::Complex32 as C32;

let n_fft = 64;
let cp_len = 8;
let half = (n_fft / 2) as i32;
let data_carriers: Vec<i32> = (1..half).chain(-(half - 1)..0).collect();
let plan = CarrierPlan::new(n_fft, cp_len).with_data_carriers(data_carriers);
let cfg = OfdmConfig::new(plan, 48_000.0, 0.0, 1.0, ConstellationOrder::Qpsk);

let mut modstage = OfdmMod::new(&cfg);
let bits_in = vec![0u8; cfg.bits_per_ofdm_symbol()];
let iq = modstage.modulate(&bits_in);

let mut demod = OfdmDemod::new(&cfg);
let mut decider = OfdmDecider::new(&cfg);
let mut soft = vec![C32::default(); demod.num_data_carriers()];
let mut bits_out = vec![0u8; cfg.bits_per_ofdm_symbol()];

demod.process(&iq, &mut soft);
decider.process(&soft, &mut bits_out);
assert_eq!(bits_in, bits_out);
```

### OFDM channel equalization

`OfdmEqualizer` sits between `FftBlock` and `GridExtract`. Its default
method, `TrainingSymbolHold`, estimates the channel once from a training
symbol and holds the estimate for the rest of the packet — the correct
default (not just the simplest) for OFDM's line-of-sight VHF–EHF target
bands, where multipath is static or slowly varying across a packet.
`PerSymbolPilotInterp` re-estimates every data symbol via frequency-domain
linear interpolation between `CarrierGrid`'s pilot bins — the explicit
opt-in for genuinely time-varying (fast-moving/Doppler) channels.

```rust
use orion_sdr::{
    core::Block,
    demodulate::{EqualizerMethod, OfdmDecider, OfdmEqualizer},
    multicarrier::{CarrierGrid, CyclicPrefixRemove, FftBlock, GridExtract},
};
use num_complex::Complex32 as C32;

let n_fft = cfg.carrier_plan.n_fft();
let cp_len = cfg.carrier_plan.cp_len();
let grid = CarrierGrid::from_plan(&cfg.carrier_plan);

let mut cp_remove = CyclicPrefixRemove::new(n_fft, cp_len);
let mut fft = FftBlock::new(n_fft);
let mut eq = OfdmEqualizer::new(&cfg, EqualizerMethod::TrainingSymbolHold);
let mut grid_extract = GridExtract::new(grid);
let mut decider = OfdmDecider::new(&cfg);

// Once per packet: FFT the received training symbol and hand it to the
// equalizer (training-symbol IQ obtained via `generate_ofdm_preamble` /
// `ofdm_sync`, see below).
let mut training_time = vec![C32::default(); n_fft];
cp_remove.process(&training_symbol_iq, &mut training_time);
let mut training_freq = vec![C32::default(); n_fft];
fft.process(&training_time, &mut training_freq);
eq.estimate_from_training_symbol(&training_freq);

// Per data symbol: CP-remove → FFT → equalize → grid-extract → decide.
let mut time = vec![C32::default(); n_fft];
cp_remove.process(&data_symbol_iq, &mut time);
let mut freq = vec![C32::default(); n_fft];
fft.process(&time, &mut freq);
let mut equalized = vec![C32::default(); n_fft];
eq.process(&freq, &mut equalized);
let mut soft = vec![C32::default(); grid_extract.num_data_carriers()];
grid_extract.process(&equalized, &mut soft);
let mut bits = vec![0u8; cfg.bits_per_ofdm_symbol()];
decider.process(&soft, &mut bits);
```

This release's per-bin equalizer models delay spreads up to `cp_len` — a
longer channel impulse response causes inter-symbol interference the model
doesn't capture.

### Soft (LLR) demapping

`OfdmSoftDemod` is a separate type from `OfdmDecider` (not a mode flag),
producing max-log LLRs per bit instead of hard decisions. No mandatory FEC
ships with this crate — the LLRs are the deliverable for an
external/user-supplied FEC layer. Positive LLR means the bit is more likely 0,
matching the crate-wide convention.

```rust
use orion_sdr::{core::Block, demodulate::OfdmSoftDemod};

let mut soft_demod = OfdmSoftDemod::new(&cfg);
let mut llrs = vec![0.0f32; cfg.bits_per_ofdm_symbol()];
soft_demod.process(&soft, &mut llrs);
```

### RX diagnostics (`OfdmRxFrame`)

`build_ofdm_rx_frame` assembles per-packet diagnostics from demodulated soft
symbols and hard-decided bits. `Option` fields make "not yet measured at
this pipeline stage" explicit — `evm_db` only needs the soft/hard pair, so
it's always populated; `cfo_hz`/`timing_offset_samples` require having run
`ofdm_sync` first, and `channel_mse` isn't computed by the current equalizer.

```rust
use orion_sdr::demodulate::build_ofdm_rx_frame;

let frame = build_ofdm_rx_frame(&cfg, &soft, bits_out);
println!("EVM: {:?} dB", frame.evm_db);
```

### OFDM packet sync + CFO acquisition

`ofdm_sync` searches for a Schmidl & Cox-style repeated-segment preamble,
recovering coarse timing and fractional CFO (unambiguous within ±½ the
subcarrier spacing). Add a training symbol via `OfdmPreamble::with_training_symbol`
to additionally recover integer-multiple-of-spacing CFO, extending the
capture range to the full oscillator-error budget of the upper target bands
(the same training symbol also feeds `OfdmEqualizer::estimate_from_training_symbol`
above).

```rust
use orion_sdr::{
    dsp::Rotator,
    sync::{OfdmPreamble, generate_ofdm_preamble, ofdm_sync},
};
use num_complex::Complex32 as C32;

let n_fft = cfg.carrier_plan.n_fft();
let cp_len = cfg.carrier_plan.cp_len();
let preamble = OfdmPreamble::new(4, 32).with_training_symbol(n_fft, cp_len);
let preamble_iq = generate_ofdm_preamble(&preamble, &cfg);

// Prepend to the data symbols, transmit, then on receive:
let results = ofdm_sync(&received_iq, cfg.fs, &preamble, 0, received_iq.len());
let best = &results[0];   // sorted by descending score

// Total CFO = fractional + integer·subcarrier_spacing; correct with Rotator
// before handing samples to OfdmDemod/OfdmEqualizer.
let subcarrier_spacing_hz = cfg.fs / n_fft as f32;
let total_cfo_hz = best.cfo_hz + best.integer_cfo_bins as f32 * subcarrier_spacing_hz;
let mut correction = Rotator::new(-total_cfo_hz, cfg.fs);
let mut corrected = vec![C32::default(); received_iq.len()];
correction.rotate_block(&received_iq, &mut corrected);

let training_start = best.start_sample + preamble.num_repeats * preamble.repeat_len;
let data_start = best.start_sample + preamble.total_len();
```
