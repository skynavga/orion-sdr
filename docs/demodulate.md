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

The per-bin equalizer models delay spreads up to `cp_len` — a longer channel
impulse response causes inter-symbol interference the model doesn't capture.

### RX FFT-window back-off

By default the receiver's FFT window is the last `n_fft` samples of each symbol
— pinned at the cyclic-prefix boundary, with the whole guard ahead of it and
nothing behind. `rx_window_backoff` slides it `b` samples *earlier* into the
guard, leaving slack on **both** sides of the useful part:

```rust
// Set once on the shared config; SymbolFft applies it on every RX path.
let cfg = cfg.with_rx_window_backoff(cp_len / 2);
// DVB-T sets it on the demod objects instead:
// DvbTFrameDemod::new(params).with_rx_window_backoff(cp_len / 2)
```

Two reasons to want it. It buys tolerance to **pre-echo and small timing error**
(standard receiver practice), and it is what makes the transmitter's spectral
shaping transparent — the taper and mask in
[modulate.md](modulate.md#out-of-band-spectral-shaping-tx) live in exactly the
guard samples a backed-off window discards. Sample count and symbol boundaries
are unchanged, so a strided receiver's cursor is unaffected.

**It requires an equalizer.** Sliding the window by `b` multiplies bin `k` by
`exp(-j2πkb/n_fft)` (the FFT shift theorem). That is harmless only where the
channel estimate is measured at the *same* back-off and divides the ramp back
out — the streaming COFDM demod and the DVB-T scattered path. On a bare,
unequalized `OfdmDemod` or batch `OfdmFrameDemod`, a nonzero back-off leaves the
ramp uncorrected and corrupts the decode; leave it `0` there.

**How large it can be depends on the equalizer, not just the guard**, and the
answer is the opposite of the intuitive one:

- `TrainingSymbolHold` (the COFDM default) measures every bin from the training
  symbol, so it absorbs any `b` the guard allows — verified up to `b = cp_len`.
- `PerSymbolPilotInterp` (the DVB-T scattered path) only samples the channel
  every `pilot_spacing` carriers. The ramp advances `θ = 2π·b·pilot_spacing/n_fft`
  per gap, and past `π` the interpolation aliases: `b < n_fft/(2·pilot_spacing)`
  (`SymbolFft::max_pilot_safe_backoff`), which is **85 samples for DVB-T 2K
  regardless of guard interval** (`DVB_T_MAX_RX_WINDOW_BACKOFF`).

So holding one full-resolution estimate is the *stronger* option here, and
pilot interpolation is what costs budget.

And it costs it earlier than the aliasing bound suggests. The interpolation is
*linear*, so it approximates the ramp's arc by a chord and is wrong by
`1 − cos(θ/2)` between pilots — graded, not a cliff. Measured on DVB-T 2K:
`b = 32` (θ = 68°) is free, `b = 42` (θ = 90°) costs ~1 dB, `b = 64` costs ~6 dB,
and at `b = 85` the link does not close at any SNR. Budget against
**`b ≤ n_fft/(4·pilot_spacing)`**, not the aliasing cap; see
[performance.md](performance.md#the-rx-window-back-off-costs-sensitivity-well-before-it-aliases).
Noiseless round trips pass well past this, which is exactly why it is easy to
miss.

Picking `b` is step 1 of the TX-side sizing recipe, since it sets the slack the
taper and mask then share:
[modulate.md → Choosing the numbers](modulate.md#choosing-the-numbers).

### Soft (LLR) demapping

`OfdmSoftDemod` is a separate type from `OfdmDecider` (not a mode flag),
producing max-log LLRs per bit instead of hard decisions. These LLRs feed the
COFDM frame layer's inner FEC decoder (see [ofdm.md](ofdm.md)), or any
external/user-supplied FEC. Positive LLR means the bit is more likely 0,
matching the crate-wide convention.

```rust
use orion_sdr::{core::Block, demodulate::OfdmSoftDemod};

let mut soft_demod = OfdmSoftDemod::new(&cfg);
let mut llrs = vec![0.0f32; cfg.bits_per_ofdm_symbol()];
soft_demod.process(&soft, &mut llrs);
```

### RX diagnostics (`OfdmRxFrame`)

`build_ofdm_rx_frame` assembles per-packet diagnostics from demodulated soft
symbols and hard-decided bits. `Option` fields make "not measured at this
pipeline stage" explicit: a field is `None` where the stage that would produce
it did not run, never where the measurement was zero.

```rust
use orion_sdr::demodulate::build_ofdm_rx_frame;

let frame = build_ofdm_rx_frame(&cfg, &soft, bits_out);
println!("EVM: {:?} dB", frame.evm_db);
```

Via this entry point only `evm_db` is populated — it needs just the soft/hard
pair. The rest come from the frame layer, which has run acquisition and the FEC
chain:

| Field | Source |
| --- | --- |
| `cfo_hz`, `timing_offset_samples` | `ofdm_sync`, on the streaming path |
| `sync_score` | the S&C score the frame was acquired at (`None` on the batch path, which never acquires) |
| `evm_db` | measured on the payload's own hard decisions |
| `inner_fec_ok`, `outer_fec_ok` | the two FEC stages, reported **separately** |
| `channel_estimate` | opt-in, `with_channel_estimate` |
| `channel_ber`, `inner_ber` | opt-in, `with_error_rates` |
| `channel_mse` | not computed — see below |

`channel_mse` stays `None` deliberately. A scalar mean-square error needs a
reference to measure against, and a single-shot training estimate has none;
deriving one means separating channel from noise, which is an estimator rather
than an exposure. `channel_estimate` carries strictly more — the per-bin channel
`H[k] = received[k] / known[k]`, whose inverse FFT is a power delay profile, and
from that delay spread and whether echoes fall inside the guard. It is the
*channel*, not the raw received training bins: the known pattern is
crate-internal, so a caller could not divide it out.

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

## COFDM Frame Demodulation

The COFDM frame layer recovers a whole `FramePacket` — running the inverse of
`OfdmFrameMod`'s chain (soft-demap → deinterleave → concatenated FEC decode →
CRC) — through two entry points.

**Batch, known start.** `OfdmFrameDemod` decodes one frame whose IQ begins at the
first post-preamble sample (the caller has already synchronized and, if needed,
equalized). It is the exact counterpart of `OfdmFrameMod`: construct it once from
the same `OfdmConfig` and `McsTable`, then `decode` each frame.

```rust
use orion_sdr::demodulate::OfdmFrameDemod;

// `iq` is a full modulated frame (preamble + training + header + payload); the
// batch decoder wants the samples after the preamble+training.
let body = &iq[modu.preamble().total_len()..];

// The demodulator owns a persistent CodecCache warmed across every `decode`; use
// `OfdmFrameDemod::with_cache(cfg, table, cache)` to share one `Arc<CodecCache>`
// with a modulator so each FEC code is built once between them.
let demod = OfdmFrameDemod::new(cfg, table);
match demod.decode(body) {
    Ok(frame) => {
        assert_eq!(frame.metadata.mcs_index, 1);
        // frame.payload — the recovered bytes; frame.metadata.sequence_num, etc.
    }
    Err(e) => eprintln!("frame decode failed: {e:?}"),
}
```

**Streaming, unknown start.** `OfdmFrameStreamDemod` accumulates raw IQ across
`feed` calls, locates preambles with `ofdm_sync`, corrects total CFO, estimates
the channel from the training symbol, decodes each frame, and drains its
samples — yielding any frames (or typed errors) that completed. A frame split
across `feed` calls is held until a later call completes it. It mirrors
`Ft8StreamDecoder`'s accumulate-and-drain shape (`feed`/`flush`/`clear`).

```rust
use orion_sdr::demodulate::OfdmFrameStreamDemod;

// Both diagnostics below are off by default — each costs per decoded frame.
let mut rx = OfdmFrameStreamDemod::new(cfg, table, preamble)
    .with_error_rates(true)      // true CBER/IBER, one re-encode per frame
    .with_channel_estimate(true); // per-bin H, an n_fft-sized allocation

// `feed` returns the frames that completed on this call (failed decodes are
// omitted; `feed_with_errors` surfaces the reasons).
for result in rx.feed(&received_iq) {
    match result {
        Ok(rx_frame) => {
            let d = &rx_frame.diagnostics;
            // Acquisition confidence, and the two FEC stages kept apart: an
            // inner failure beside an outer success is a link running hot but
            // still delivering, which a folded flag cannot express.
            println!(
                "score {:?}  CBER {:?}  IBER {:?}  inner {:?} outer {:?}",
                d.sync_score, d.channel_ber, d.inner_ber, d.inner_fec_ok, d.outer_fec_ok
            );
            let payload = &rx_frame.packet.payload;
            let _ = payload;
        }
        Err(e) => eprintln!("frame error: {e:?}"),
    }
}
let _tail = rx.flush(); // final pass over the residual buffer at end of stream
```

Both entry points **invert whatever coding chain the config and MCS table
describe** — non-default FEC, interleavers, and scrambler included — so the
receiver must be constructed from the *same* `OfdmConfig` and `McsTable` as the
transmitter. Those parameters (interleaver dimensions, scrambler poly/width/seed
mode, the MCS mapping) are shared out-of-band link state; only the per-frame
scrambler *seed* and `mcs_index` are signaled in the header. See
[modulate.md](modulate.md#configuring-the-coding-chain-non-default-fec--interleave--scramble)
for building such a config; decode is symmetric — construct `OfdmFrameDemod` or
`OfdmFrameStreamDemod` from the identical `cfg`/`table`.

The receiver's LDPC inner decoder uses the check-node rule selected by
`OfdmConfig::with_ldpc_decode_rule` (exact sum-product by default; opt-in
min-sum / scaled-min-sum for ~2× decode at a sub-0.3 dB coding-gain cost on the
payload — the header always uses sum-product). See
[performance.md](performance.md) for the measured trade.

## DVB-T Frame Demodulation (conformant, preamble-less)

`DvbTFrameDemod` is the exact inverse of `modulate::DvbTFrameMod`: it acquires the
symbol grid from the guard interval (no preamble), equalizes each symbol from the
scattered/continual pilots, soft-demaps the Figure-9a constellation, recovers the
TPS word from the 17 TPS carriers, runs the payload FEC decode, and undoes the TS
energy dispersal — returning both the payload and the TPS-signalled parameters.
Construct it with the cold-start MCS `params` (as real receivers acquire on
assumptions), which the recovered TPS then verifies. See [dvb.md](dvb.md).

```rust
use orion_sdr::demodulate::DvbTFrameDemod;

// `params` is the assumed MCS (a DvbTFrameParams); `iq` the received samples;
// `n_symbols` / `payload_len` come from the paired DvbTFrameMod's DvbTFrame.
let rx = DvbTFrameDemod::new(params)
    .decode(iq, n_symbols, payload_len)
    .expect("conformant DVB-T frame decode");

// rx.payload — the recovered TS payload; rx.tps — the parameters read off the TPS
// carriers (constellation, code_rate_hp, guard, frame_number, cell_id), which a
// receiver can check against its acquisition assumptions.
let _ = (&rx.payload, rx.tps.constellation);
```

This is a batch, single-frame receiver (the buffer holds one whole frame plus
enough lead-in for the guard-interval search). The **super-frame** and
**streaming** receivers below wrap it. It recovers only the *fractional* CFO from
the guard interval; if a real front end may be off by whole subcarriers, enable
**integer**-CFO correction with the builder flag (below).

### Receiving a spectrally-shaped DVB-T signal

When the transmitter applies symbol windowing or a baseband mask (see
[modulate.md](modulate.md#spectral-shaping-on-dvb-t)), the receiver's only job is
to supply the back-off those levers spend. Nothing about the *decoding* changes:
the scattered-pilot estimate is measured at the same back-off, so it divides out
both the phase ramp the shift induces and the mask's own frequency response,
like any other channel.

```rust
use orion_sdr::demodulate::{DvbTFrameDemod, DvbTFrameStreamDemod, DvbTSuperFrameDemod};
use orion_sdr::waveform::dvb_t::DVB_T_MAX_RX_WINDOW_BACKOFF;

// The one DVB-T-specific rule: cp_len/2 is the COFDM answer, but DVB-T's
// pilot-interpolated equalizer makes the back-off cost sensitivity. 32 is free,
// 42 costs ~1 dB, 64 costs ~6 dB, and 85 (the aliasing cap) never decodes — so
// clamping to DVB_T_MAX_RX_WINDOW_BACKOFF is NOT the right move.
let _aliasing_cap = DVB_T_MAX_RX_WINDOW_BACKOFF;             // 85 — a ceiling, not a target
let backoff = 32usize;                                       // free; use 42 if you need the slack

// Set the same value on whichever receiver the link uses.
let _batch = DvbTFrameDemod::new(params).with_rx_window_backoff(backoff);
let _super = DvbTSuperFrameDemod::new(sf_params).with_rx_window_backoff(backoff);
let _stream = DvbTFrameStreamDemod::new(params, n_symbols, payload_len)
    .with_rx_window_backoff(backoff);
```

The back-off must match the transmitter's *budget*, not any single TX value —
there is no TX/RX pair to keep numerically equal the way `roll_off` and `b` are
often quoted together. What has to hold is
`roll_off + group_delay ≤ min(cp_len − b, b)`, and `b` is one term in it.
[modulate.md](modulate.md#choosing-the-numbers) works the choice through.

Two failure modes are worth recognising, because neither reports itself as a
back-off problem:

- **Back-off sized against the aliasing cap.** This is the easy mistake, because
  a noiseless round trip still passes at `b = 64` or `b = 85` — the FEC has the
  margin to absorb the interpolation error when there is no noise to spend it on.
  Under noise it does not: `b = 64` costs ~6 dB and `b = 85` never closes. Keep
  `b ≤ 32` (free) or `≤ 42` (~1 dB).
- **Nothing else.** Shaping used to have a second failure mode here — a tapered
  frame beginning at sample 0 of the buffer would lock onto the wrong symbol —
  which is now handled inside the estimator (below).

#### Acquiring a shaped signal with no lead-in

Symbol windowing biases the guard-interval ML timing estimate **early**, by
roughly a third of `roll_off`: the taper attenuates each symbol's leading
cyclic-prefix samples but not their unwindowed copies in the symbol's interior,
so the correlation peaks slightly before the true boundary. A long baseband mask
does the same by smearing the correlation.

That bias is harmless where the receiver has lead-in — the peak simply lands a
few samples early, which a backed-off window absorbs. It is not harmless where
the frame begins at sample 0: the search range is `[0, period)`, a negative phase
is not representable, and the argmax surfaces at `period − δ` instead. Right
phase, wrong symbol — nearly a whole symbol late.

Zero lead-in is not a corner case. `DvbTSuperFrameDemod` slices every constituent
frame that way, and `DvbTFrameStreamDemod` re-acquires inside the slice it just
acquired, which leaves that inner search almost none.

`dvb_t_gi_sync` therefore reports the period boundary at or before the peak when
two conditions hold: the peak sits within `cp_len/2` below that boundary, *and*
the boundary's own **single-symbol** correlation reaches half the peak's. Both
are needed — see `GiSyncConfig::origin_score_ratio`, which documents why the
score rather than the ML metric, and why a single symbol rather than the
accumulated one. Set `origin_score_ratio = 0.0` for the plain argmax.

### DVB-T super-frame demodulation

`DvbTSuperFrameDemod` is the inverse of `modulate::DvbTSuperFrameMod`: it decodes
the four frames, verifies the frame-number sequence `0,1,2,3` (which implies the
correct alternating TPS sync word), reassembles the 16-bit cell id from its byte
halves, and concatenates the payloads. `symbols_per_frame` and `frame_payload_lens`
come from the paired modulator's `DvbTSuperFrame`.

```rust
use orion_sdr::demodulate::DvbTSuperFrameDemod;

// `params` is a DvbTSuperFrameParams; `sf` is the paired modulator's DvbTSuperFrame
// (its `symbols_per_frame` / `frame_payload_lens` describe the re-slice).
let rx = DvbTSuperFrameDemod::new(params)
    .decode(&sf.iq, sf.symbols_per_frame, sf.frame_payload_lens)
    .expect("super-frame decode");
// rx.payload — the four frames' payloads concatenated; rx.cell_id — the reassembled
// 16-bit cell id.
let _ = (&rx.payload, rx.cell_id);
```

### DVB-T streaming reception (`feed`/`flush`)

`DvbTFrameStreamDemod` decodes a **continuous run of frames** as their samples
arrive, mirroring `OfdmFrameStreamDemod`. It accumulates IQ across `feed` calls,
guard-interval-acquires the next frame at the front of the buffer, decodes it,
drains its samples, and loops — holding a partially-arrived frame until a later
`feed` completes it. The frame geometry (`n_symbols`, `payload_len`) is fixed at
construction (as the batch demod takes it); `feed` is chunk-boundary-invariant.

```rust
use orion_sdr::demodulate::DvbTFrameStreamDemod;

// `params`, `n_symbols`, `payload_len` fix the frame geometry (as the batch demod
// takes them); `stream` is a run of incoming IQ chunked arbitrarily.
let mut rx = DvbTFrameStreamDemod::new(params, n_symbols, payload_len);
// Add `.with_rx_window_backoff(b)` here when the transmitter is shaped.
let mut frames = Vec::new();
for chunk in stream.chunks(4096) {
    // `feed` returns the frames that completed on this call (decode errors are
    // Err entries; `flush` runs a final pass over the residual buffer).
    frames.extend(rx.feed(chunk));
}
frames.extend(rx.flush());
```

### DVB-T integer-CFO correction (a builder flag)

The guard-interval acquisition resolves the CFO only within ±½ a subcarrier. A
capture with a larger front-end offset is shifted by whole subcarriers, and the
frame will not demap until that integer offset is removed. This is a
**link-constant, RX-only** property, so it is a set-once builder flag on the demod:
`DvbTFrameDemod::new(params).with_integer_cfo_correction(true)`. When enabled, the
demod estimates the whole-subcarrier offset from the 45 continual pilots (fixed
positions, boosted) right after its own guard-interval acquisition and rotates it
out internally — the caller just decodes as usual. Off by default: a clean link
needs no correction, and the estimate/rotate is skipped entirely.

The super-frame and streaming receivers carry the same flag —
`DvbTSuperFrameDemod::new(params).with_integer_cfo_correction(true)` (delegated to
each constituent frame) and
`DvbTFrameStreamDemod::new(params, n_symbols, payload_len)` gains it via
`.with_integer_cfo_correction(true)`:

```rust
use orion_sdr::demodulate::{DvbTFrameDemod, DvbTFrameStreamDemod};

// Single frame: enable correction once at construction, then decode raw IQ
// (`raw` may carry a whole-subcarrier offset the demod removes internally).
let rx = DvbTFrameDemod::new(params)
    .with_integer_cfo_correction(true)
    .decode(&raw, n_symbols, payload_len);

// Streaming: the whole run shares the front-end offset, so set the flag once.
let mut stream = DvbTFrameStreamDemod::new(params, n_symbols, payload_len)
    .with_integer_cfo_correction(true);
let frames = stream.feed(&raw);
```

For the **super-frame**, build
`DvbTSuperFrameDemod::new(sf_params).with_integer_cfo_correction(true)` and decode
as usual; the flag applies to every constituent frame. A fixed front-end offset is
constant across a capture, which is exactly why the flag is set once rather than
per call. Under noise the pilot peak is modest (45 of 1705 carriers, boosted
~1.78×), so the demod accumulates several symbols' pilot energy before estimating.
Always-on, the correction costs on the order of a few percent of the decode
(continual-pilot search per frame; see [performance.md](performance.md)).
