<!--
  Copyright (c) 2026 G & R Associates LLC
  SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Modulator Usage

Usage patterns and examples for all modulators in **orion-sdr**: CW, AM, SSB, FM, PM,
BPSK, QPSK, QAM-16/64/256, FT8/FT4, and OFDM.

## CW Keyed Modulator

Input is a keying envelope (0.0 = key up, 1.0 = key down), not raw audio.
Rise/fall shaping is applied automatically.

```rust
use orion_sdr::modulate::CwKeyedMod;

let fs = 48_000.0;
let tone_hz = 700.0;
let rise_ms = 5.0;
let fall_ms = 5.0;

let mut mod_ = CwKeyedMod::new(fs, tone_hz, rise_ms, fall_ms);
```

## AM DSB Modulator

Full carrier (A3E) AM modulator. Input is audio (`f32`), output is baseband IQ (`Complex32`).

```rust
use orion_sdr::modulate::AmDsbMod;

let fs = 48_000.0;
let rf_hz = 0.0;            // baseband output; set non-zero to upconvert
let carrier_level = 1.0;
let modulation_index = 0.8; // 80% modulation

let mut mod_ = AmDsbMod::new(fs, rf_hz, carrier_level, modulation_index);
```

## SSB Phasing Modulator

Weaver/phasing-method SSB modulator. Set `usb = true` for upper sideband, `false` for lower.

```rust
use orion_sdr::modulate::SsbPhasingMod;

let fs = 48_000.0;
let audio_bw_hz = 2_800.0;
let audio_if_hz = 1_500.0;
let rf_hz = 0.0;
let usb = true;

let mut mod_ = SsbPhasingMod::new(fs, audio_bw_hz, audio_if_hz, rf_hz, usb);
```

## FM Phase-Accumulator Modulator

Phase-accumulator FM modulator using phasor recurrence (no per-sample `cos`/`sin`).

```rust
use orion_sdr::modulate::FmPhaseAccumMod;

let fs = 48_000.0;
let deviation_hz = 2_500.0;
let rf_hz = 0.0;

let mut mod_ = FmPhaseAccumMod::new(fs, deviation_hz, rf_hz);
```

## PM Direct-Phase Modulator

Maps ±1.0 audio directly to ±`kp` radians of instantaneous phase.

```rust
use orion_sdr::modulate::PmDirectPhaseMod;

let fs = 48_000.0;
let kp_rad_per_unit = 0.8;
let rf_hz = 0.0;

let mut mod_ = PmDirectPhaseMod::new(fs, kp_rad_per_unit, rf_hz);
```

---

## Digital Modulators

All digital modulation pipelines use two stages:

1. A **mapper** converts a flat `&[u8]` bit stream (one bit per byte, LSB used) into
   `&[Complex32]` constellation symbols.
2. A **waveform stage** (`BpskMod`, `QpskMod`, or `QamMod`) multiplies each symbol by
   a carrier phasor and applies gain.  Set `rf_hz = 0.0` for baseband output.

Both stages implement `Block` and can be driven directly or wrapped in `IqToIqChain`.

### BPSK

Gray-coded; 1 bit per symbol.  Constellation: `(+1, 0)` for bit 0, `(−1, 0)` for bit 1.

```rust
use orion_sdr::{
    core::Block,
    modulate::{BpskMapper, BpskMod},
};
use num_complex::Complex32 as C32;

let bits: Vec<u8> = vec![0, 1, 0, 0, 1, 1, 0, 1]; // one bit per byte (LSB)
let mut syms = vec![C32::default(); bits.len()];
let mut iq   = vec![C32::default(); bits.len()];

BpskMapper::new().process(&bits, &mut syms);

// baseband, unit gain
BpskMod::new(1.0, 0.0, 1.0).process(&syms, &mut iq);
```

### QPSK

Gray-coded; 2 bits per symbol.  Input consumed in pairs `[b0, b1]`; normalized to
unit energy (each axis at ±1/√2).

```rust
use orion_sdr::{
    core::Block,
    modulate::{QpskMapper, QpskMod},
};
use num_complex::Complex32 as C32;

// 8 bits → 4 symbols
let bits: Vec<u8> = vec![0,0, 0,1, 1,0, 1,1];
let mut syms = vec![C32::default(); 4];
let mut iq   = vec![C32::default(); 4];

QpskMapper::new().process(&bits, &mut syms);
QpskMod::new(1.0, 0.0, 1.0).process(&syms, &mut iq);
```

### QAM-16 / QAM-64 / QAM-256

Const-generic `QamMapper<const BITS: usize>` where `BITS` is the bits per symbol:
`4` → QAM-16, `6` → QAM-64, `8` → QAM-256.  Type aliases are provided for convenience.

Input is consumed `BITS` bytes per symbol: the first `BITS/2` bytes encode the I axis
(MSB-first within the axis Gray index), the next `BITS/2` bytes encode the Q axis.
Constellation is normalized to unit average symbol energy.

`QamMod` is order-independent and shared across all QAM variants.

```rust
use orion_sdr::{
    core::Block,
    modulate::{Qam16Mapper, Qam64Mapper, Qam256Mapper, QamMod},
};
use num_complex::Complex32 as C32;

let n_syms = 64;

// QAM-16: 4 bits/symbol
let bits16: Vec<u8> = vec![0u8; n_syms * 4];
let mut syms = vec![C32::default(); n_syms];
let mut iq   = vec![C32::default(); n_syms];
Qam16Mapper::new().process(&bits16, &mut syms);
QamMod::new(1.0, 0.0, 1.0).process(&syms, &mut iq);

// QAM-64: 6 bits/symbol — same pattern, different mapper
let bits64: Vec<u8> = vec![0u8; n_syms * 6];
Qam64Mapper::new().process(&bits64, &mut syms);
QamMod::new(1.0, 0.0, 1.0).process(&syms, &mut iq);

// QAM-256: 8 bits/symbol
let bits256: Vec<u8> = vec![0u8; n_syms * 8];
Qam256Mapper::new().process(&bits256, &mut syms);
QamMod::new(1.0, 0.0, 1.0).process(&syms, &mut iq);
```

---

## FT8 / FT4 Modulators

FT8 and FT4 are weak-signal digital modes used on HF amateur radio.  The
modulator takes a frame of pre-encoded tone indices and produces a
phase-continuous CPFSK IQ waveform at 12 kHz.  Costas synchronisation arrays
are inserted automatically at fixed positions.

### FT8

8-FSK, 79 symbols total (58 data + 21 Costas), 1 920 samples/symbol → 151 680 samples/frame.

```rust
use orion_sdr::modulate::{Ft8Mod, Ft8Frame};
use orion_sdr::codec::ft8::{Ft8Codec, Ft8Bits};

let payload: Ft8Bits = [0u8; 10];   // 77-bit payload
let frame: Ft8Frame = Ft8Codec::encode(&payload);

let modulator = Ft8Mod::new(
    12_000.0,   // fs: sample rate (Hz)
    1_000.0,    // base_hz: frequency of tone 0
    0.0,        // rf_hz: upconversion (0 = baseband)
    1.0,        // gain
);
let iq = modulator.modulate(&frame);   // Vec<Complex32>, len = 151_680
```

### FT4

4-FSK, 105 symbols total (87 data + 18 Costas/ramps), 576 samples/symbol → 60 480 samples/frame.

```rust
use orion_sdr::modulate::{Ft4Mod, Ft4Frame};
use orion_sdr::codec::ft4::{Ft4Codec, Ft4Bits};

let payload: Ft4Bits = [0u8; 10];
let frame: Ft4Frame = Ft4Codec::encode(&payload);

let modulator = Ft4Mod::new(12_000.0, 1_000.0, 0.0, 1.0);
let iq = modulator.modulate(&frame);   // Vec<Complex32>, len = 60_480
```

Both modulators maintain phase continuity across symbol boundaries (CPFSK).
The inner loop uses a 4-sample unrolled phasor recurrence with periodic
renormalisation — no per-sample `sin`/`cos` calls.

### Carrier upconversion

All waveform stages accept an `rf_hz` parameter.  When non-zero the symbols are
rotated onto that carrier using an internal `Rotator` (phasor recurrence, no per-sample
trig):

```rust
use orion_sdr::{core::Block, modulate::{BpskMapper, BpskMod}};
use num_complex::Complex32 as C32;

let fs = 2_400_000.0;   // 2.4 MHz sample rate
let rf_hz = 100_000.0;  // 100 kHz IF carrier

let bits: Vec<u8> = (0..256).map(|i| (i & 1) as u8).collect();
let mut syms = vec![C32::default(); 256];
let mut iq   = vec![C32::default(); 256];

BpskMapper::new().process(&bits, &mut syms);
BpskMod::new(fs, rf_hz, 1.0).process(&syms, &mut iq);
```

---

## OFDM Modulator

`OfdmMod` fuses the whole TX chain — symbol mapper (BPSK/QPSK/QAM, reused
verbatim) → resource-grid mapping → IFFT → cyclic prefix → optional RF
upconversion — into a single `Block<In = u8, Out = C32>`. Numerology
(`n_fft`, `cp_len`, carrier layout) is entirely caller-owned via `CarrierPlan`;
see [ofdm.md](ofdm.md) for the FFT
normalization and carrier-indexing conventions.

```rust
use orion_sdr::{
    core::Block,
    modulate::{ConstellationOrder, OfdmConfig, OfdmMod},
    multicarrier::CarrierPlan,
};

let n_fft = 64;
let cp_len = 8;

// Signed carrier indices (bin 0 = DC); a contiguous band on both sides of
// DC, leaving DC itself null (opt in explicitly to use bin 0).
let half = (n_fft / 2) as i32;
let data_carriers: Vec<i32> = (1..half).chain(-(half - 1)..0).collect();
let plan = CarrierPlan::new(n_fft, cp_len).with_data_carriers(data_carriers);

let cfg = OfdmConfig::new(
    plan,
    48_000.0,               // fs
    0.0,                    // rf_hz (0 = baseband)
    1.0,                    // gain
    ConstellationOrder::Qpsk,
);

let mut modstage = OfdmMod::new(&cfg);
let bits = vec![0u8; cfg.bits_per_ofdm_symbol()];
let mut iq = vec![num_complex::Complex32::default(); cfg.samples_per_ofdm_symbol()];
modstage.process(&bits, &mut iq);

// Or use the convenience wrapper, which zero-pads a final partial symbol
// and handles multi-symbol batches:
let bits_batch = vec![0u8; 5 * cfg.bits_per_ofdm_symbol()];
let iq_batch = modstage.modulate(&bits_batch);
```

`OfdmMod` consumes whole `bits_per_ofdm_symbol()`-sized bit chunks and
produces whole `samples_per_ofdm_symbol()`-sized IQ chunks per `process()`
call; a partial trailing chunk is a no-op (`WorkReport::default()`), with no
cross-call buffering — the same contract every `multicarrier::` primitive
follows. `OfdmMod` deliberately does **not** route its sub-blocks through
`IqToIqChain`/`AudioToIqChain`, since those chains assume near-1:1 sample
flow and would silently truncate the IFFT+CP's rate expansion.

Prepend a packet-sync preamble with `generate_ofdm_preamble` (see
[demodulate.md](demodulate.md#ofdm-packet-sync--cfo-acquisition)) when the
receiver doesn't already know the packet start time and carrier offset.

## COFDM Frame Modulator

`OfdmFrameMod` builds on the OFDM PHY to serialize a whole `FramePacket`
(metadata + opaque payload bytes) into a flat IQ stream —
`[preamble + training][header][payload]` — applying the concatenated FEC chain
configured on `OfdmConfig` and selected per frame by an `McsTable`. The FEC
stages (CRC, outer + inner code, two interleavers, PN scrambler) all default
off and are enabled with `with_*` builders; see [ofdm.md](ofdm.md) for the chain
order and design conventions.

```rust
use orion_sdr::{
    fec::{FrameMetadata, FramePacket},
    modulate::{ConstellationOrder, OfdmConfig, OfdmFrameMod, McsTable},
    multicarrier::CarrierPlan,
    sync::OfdmPreamble,
};

let n_fft = 64;
let cp_len = 8;
let half = (n_fft / 2) as i32;
let data: Vec<i32> = (1..half).chain(-(half - 1)..0).collect();
let plan = CarrierPlan::new(n_fft, cp_len).with_data_carriers(data);

// The payload's inner/outer FEC and constellation come from the MCS table
// (below), selected per frame by `mcs_index` — not from the config's own
// `inner_fec`/`outer_fec`. The config carries the link-wide settings: CRCs,
// interleavers, scrambler, header format. `McsTable::default_ladder` pairs a
// rate-1/2 LDPC inner code with a BCH(t=8) outer code across a BPSK→QAM-64
// ladder.
let cfg = OfdmConfig::new(plan, 48_000.0, 0.0, 1.0, ConstellationOrder::Bpsk)
    .with_payload_crc(orion_sdr::fec::CrcKind::Crc32)
    .with_header_crc(orion_sdr::fec::CrcKind::Crc16);

// A Schmidl & Cox preamble with a training symbol sized to the plan, so the
// receiver can acquire timing/CFO and estimate the channel.
let preamble = OfdmPreamble::new(4, 16)
    .with_training_symbol(cfg.carrier_plan.n_fft(), cfg.carrier_plan.cp_len());

let table = McsTable::default_ladder();
let modu = OfdmFrameMod::new(cfg, table, preamble);

// Modulate a frame. `mcs_index` selects the payload's constellation + FEC
// from the table; `sequence_num` and the payload are carried end to end.
let payload: Vec<u8> = (0..96).map(|i| (i * 37 + 11) as u8).collect();
let frame = FramePacket::new(FrameMetadata::new(/*sequence_num*/ 1, /*mcs_index*/ 1), payload);
let iq = modu.modulate_frame(&frame, /*per_frame_seed*/ 0);
```

`OfdmFrameMod` holds a per-link `CodecCache`, so a stream of frames builds each
FEC code (the LDPC parity-check matrix especially) only once. To share one set
of built codes between a modulator and a demodulator, construct both with
`OfdmFrameMod::with_cache`/`OfdmFrameStreamDemod::with_cache` and the same
`Arc<CodecCache>`. The receiver's LDPC decode rule is a *config* choice
(`with_ldpc_decode_rule`) read on the RX side; it does not affect modulation.

### Configuring the coding chain (non-default FEC / interleave / scramble)

The example above leans on `McsTable::default_ladder()`. To exercise the full
chain, split the two kinds of configuration: **per-frame coding** (the
constellation with its inner/outer FEC) lives in the `McsTable`, selected by
`mcs_index`; **link-wide settings** (interleavers, scrambler, CRCs, header
format) live on `OfdmConfig`.
The chain order is `payload → CRC → [scramble] → outer → outer-interleave →
inner → inner-interleave → [scramble] → map` (see [ofdm.md](ofdm.md)); the two
scrambler positions are `BeforeOuterFec` (the default, DVB energy-dispersal
placement) and `AfterInnerFec`.

```rust
use orion_sdr::{
    fec::{InnerFec, InterleaverKind, LdpcCode, OuterFec, PunctureRate,
          ScramblerKind, ScramblerPos, SeedMode},
    modulate::{ConstellationOrder, Mcs, McsTable, OfdmConfig},
};

// A custom MCS table: each entry is (constellation, inner FEC, outer FEC),
// selected per frame by `mcs_index`. Here a DVB-style pairing — punctured
// convolutional inner + Reed–Solomon outer — and an LDPC + BCH alternative.
let table = McsTable::new(vec![
    Mcs::new(
        ConstellationOrder::Qpsk,
        InnerFec::Convolutional { rate: PunctureRate::R1_2 },
        OuterFec::ReedSolomon { n: 60, n_parity: 8 },   // RS(60,52), t=4
    ),
    Mcs::new(
        ConstellationOrder::Qam16,
        InnerFec::Ldpc(LdpcCode::N512R34),
        OuterFec::Bch { t: 8 },
    ),
]);

// Link-wide settings on the config: both block interleavers, a fixed-seed
// additive PN scrambler placed after the inner FEC, and CRCs. (`outer_fec` /
// `inner_fec` on the config are unused when an MCS table drives the payload.)
let cfg = OfdmConfig::new(plan, 48_000.0, 0.0, 1.0, ConstellationOrder::Bpsk)
    .with_outer_interleaver(InterleaverKind::Block { rows: 4, cols: 8 })
    .with_inner_interleaver(InterleaverKind::Block { rows: 8, cols: 16 })
    .with_scrambler(ScramblerKind::Additive {
        poly: 0b1001,           // x^7 + x^4 + 1 (802.11-style)
        width: 7,
        seed: SeedMode::Fixed(0x7F),
    })
    .with_scrambler_pos(ScramblerPos::AfterInnerFec)
    .with_payload_crc(orion_sdr::fec::CrcKind::Crc32)
    .with_header_crc(orion_sdr::fec::CrcKind::Crc16);

cfg.validate().expect("consistent frame config");
```

The **receiver must be built from the same `OfdmConfig` and `McsTable`** — the
interleaver dimensions, scrambler parameters, and MCS mapping are all shared
state, not signaled in band (only the per-frame scrambler *seed* and `mcs_index`
travel in the header). `OfdmConfig::validate()` rejects inconsistent
combinations (e.g. a zero interleaver dimension, or a `PerFrameRandom` scrambler
seed with `HeaderFormat::NoHeader`, which has no header to carry the seed).

For a **per-frame-random scrambler**, set `seed: SeedMode::PerFrameRandom` and
pass the drawn seed as `modulate_frame`'s second argument — it is recorded in
the header so the receiver rebuilds the descrambler:

```rust
let cfg = cfg.with_scrambler(ScramblerKind::Additive {
    poly: 0b1001,
    width: 7,
    seed: SeedMode::PerFrameRandom,
});
// ... build modu ...
let per_frame_seed = 0xABCD_1234;               // draw a fresh one per frame
let iq = modu.modulate_frame(&frame, per_frame_seed);
```

## Out-of-band spectral shaping (TX)

Plain OFDM's out-of-band spectrum decays only as `~1/f`, so the transmitted
signal carries a wide skirt beyond its occupied band. Three independent,
**off-by-default** levers reduce it; they compose, and the full stack beats any
pair. [ofdm.md](ofdm.md) has the geometry and the transparency arguments — this
section is the TX-side API surface.

| Lever | Where it is set | Takes effect in | Chains |
| --- | --- | --- | --- |
| Edge-carrier guard | `CarrierPlan::with_contiguous_data` | any mapping (incl. bare `OfdmMod`) | COFDM only |
| Symbol-window taper | `OfdmConfig::with_symbol_window` | `OfdmFrameMod::modulate_frame` | both |
| Baseband mask | `OfdmConfig::with_tx_lowpass*` | `OfdmFrameMod::modulate_frame` | both |

Note the middle column: the taper and the mask are **post-passes over an
assembled frame**, so a bare `OfdmMod` (single symbols, no frame) applies
neither. The edge guard is different — it only changes which carriers exist, so
every path inherits it.

```rust
use orion_sdr::{
    modulate::{ConstellationOrder, OfdmConfig},
    multicarrier::{CarrierPlan, TxLowpass},
};

let (n_fft, cp_len) = (256usize, 64usize);

// 1. Edge guard: a contiguous data span leaving 31 null carriers per edge. This
//    also creates the unoccupied bandwidth the mask below needs to filter into.
let plan = CarrierPlan::new(n_fft, cp_len).with_contiguous_data(31, false);
let occupied = plan.occupied_half_carriers();       // 96 of 128

// 2 + 3. Taper and mask share ONE guard budget with the RX window back-off:
//        roll_off + group_delay <= min(cp_len - b, b), maximized at b = cp_len/2.
let roll_off = 16usize;
let taps = TxLowpass::taps_for_null_band(n_fft, occupied, 60.0);  // suggested length
let mask = TxLowpass::for_null_band(n_fft, occupied, taps, 60.0);
assert!(mask.fits_guard(cp_len, roll_off, cp_len / 2));

let cfg = OfdmConfig::new(plan, 240_000.0, 0.0, 1.0, ConstellationOrder::Qpsk)
    .with_symbol_window(roll_off)          // or with_symbol_window_beta_guard(0.25)
    .with_tx_lowpass(mask)                 // or with_tx_lowpass_null_band(taps, 60.0)
    .with_rx_window_backoff(cp_len / 2);   // RX-side; see demodulate.md
```

**The receiver is not free of this.** Neither the taper nor the mask changes how
a receiver *decodes*, but both spend guard samples the receiver has to be
discarding — which is what `with_rx_window_backoff` arranges (see
[demodulate.md](demodulate.md#rx-fft-window-back-off)). Configure the two
together, or the shaping is not transparent. `TxLowpass::fits_guard` is the
check. The edge guard needs no RX change at all.

### Choosing the numbers

Five knobs — `edge_guard`, `backoff`, `roll_off`, `num_taps`, `stopband_db` —
and two independent budgets between them. Work in this order; each step
constrains the next.

**1. Back-off first.** It is the enabler, not an afterthought: both TX levers
spend guard samples, and the back-off is what puts those samples outside the
receiver's FFT window. The slack it yields is `min(cp_len − b, b)`, which peaks
at `b = cp_len/2` giving `cp_len/2` samples. Below that you are wasting guard;
above it you are eating into the useful part.

On COFDM that is the whole story — `TrainingSymbolHold` measures every bin, so it
absorbs any `b` the guard allows. **On DVB-T it is not.** The scattered-pilot
equalizer interpolates linearly between pilots 12 carriers apart, and the
back-off's phase ramp advances `2π·b·12/2048` per gap, so a chord approximates an
arc with a `1 − cos(θ/2)` magnitude error in between. The result is a *graded*
cost, not a cliff at the aliasing limit:

| `b` (DVB-T 2K) | Cost |
| --- | --- |
| ≤ 32 | free |
| ≤ 42 (`n_fft/(4·spacing)`, θ ≤ 90°) | ≤ 1 dB |
| 64 | ~6 dB |
| 85 (`DVB_T_MAX_RX_WINDOW_BACKOFF`, θ = 180°) | link does not close at any SNR |

So **keep DVB-T's `b` at 32, or 42 if you need the slack** — the 85-sample
aliasing cap is a hard ceiling, not a target. Measured in
[performance.md](performance.md#the-rx-window-back-off-costs-sensitivity-well-before-it-aliases).

**2. Split the slack between the two TX levers.** They share it:

```text
roll_off + group_delay ≤ min(cp_len − b, b)      where group_delay = (num_taps − 1)/2
```

`TxLowpass::fits_guard(cp_len, roll_off, backoff)` is the check
(`OfdmConfig::tx_lowpass_fits_guard` and `dvb_t_tx_lowpass_fits_guard` from
Python). Overrunning it degrades gradually — a little inter-symbol leakage the
equalizer cannot invert — rather than failing abruptly, but it is the budget to
design against.

**3. Edge guard (COFDM only) — a separate, frequency-domain budget.** This one
does not touch the guard interval at all; it costs *throughput*, since each
nulled carrier removes `bits_per_ofdm_symbol` capacity.
`edge_guard ≈ ceil(0.02 … 0.05 · n_fft)` per edge is the useful range. Its
second job matters as much as its first: it creates the null band the mask
filters into. DVB-T needs none — 343 of its 2048 bins are already inactive — and
could not use one anyway, since its extreme carriers are mandatory continual
pilots.

**4. Mask length: start from the suggestion, then check the guard.**
`TxLowpass::taps_for_null_band` (Python: `tx_lowpass_suggested_taps`,
`dvb_t_tx_lowpass_suggested_taps`) returns the *shortest* filter whose
transition reaches the stop band inside the null band. Two independent
constraints meet here, and the suggestion only satisfies the first:

| Constraint | Says | Fails when |
| --- | --- | --- |
| `transition_fits` | long enough to reach the stop band before Nyquist | the null band is too narrow — widen `edge_guard` |
| `fits_guard` | short enough that its group delay fits the guard | the CP is too short — lengthen it, or lower `stopband_db` |

They pull in opposite directions, which is the whole sizing problem. A wider
null band needs a *shorter* filter; a deeper stop band needs a longer one.

**5. Taper: whatever slack is left.** With no mask, `roll_off = cp_len/2` is the
maximum transparent taper (`with_symbol_window_beta_guard(0.5)`). With a mask,
`roll_off = slack − group_delay`. Near the band edge the taper is the better of
the two levers — the mask leaves its own transition deliberately unattenuated —
so it is worth keeping some.

**6. On a preamble-bearing frame, check acquisition too.** The mask filters the
whole burst, preamble included, and Schmidl & Cox repetition only survives where
the taps see repeated samples: keep `group_delay ≪ repeat_len`. Preamble-less
waveforms (DVB-T, which acquires from the CP) are bound only by the guard
budget.

Two worked configurations, both measured in
[performance.md](performance.md#out-of-band-emission-spectral-shaping):

| | COFDM | DVB-T (G1/8) |
| --- | --- | --- |
| `n_fft` / `cp_len` | 256 / 64 | 2048 / 256 |
| `backoff` | 32 (`cp_len/2`) | 32 (the free-cost limit, not `cp_len/2` = 128) |
| slack | 32 | 32 |
| `edge_guard` | 31 (12% of `n_fft`) | n/a |
| `num_taps` / group delay | 45 / 22 | 45 / 22 |
| `roll_off` | 8 | 8 |
| budget used | 22 + 8 = 30 ≤ 32 | 22 + 8 = 30 ≤ 32 |

The DVB-T column is not a typo: its 256-sample guard buys it no more shaping room
than COFDM's 64-sample one, because the binding constraint is the pilot grid, not
the guard. Measured cost of that configuration: ~0.5 dB, all of it the back-off's.

**Defaults are off**, so a config that asks for none of this emits exactly what
it emitted before.

## DVB-T Frame Modulator (conformant, preamble-less)

`DvbTFrameMod` emits a fully conformant DVB-T on-air frame (ETSI EN 300 744): no
Schmidl & Cox preamble and no `OrionSdr` header — the transmission parameters ride
on the TPS carriers, and the receiver acquires from the guard interval. It is a
dedicated per-standard assembler (not `OfdmFrameMod`, which is preamble + header
oriented): construct it once with the link's `DvbTFrameParams`, then call
`modulate` per frame. The RX is `demodulate::DvbTFrameDemod`. See [dvb.md](dvb.md)
for the waveform design.

```rust
use orion_sdr::{
    fec::PunctureRate,
    modulate::{ConstellationOrder, DvbTFrameMod},
    waveform::dvb_t::{DvbTFrameParams, DvbTLinkParams, GuardInterval, NbBandwidth},
};

// The transmission parameters — everything the TPS word signals. The link set
// (guard, constellation, code rate) is shared with the super-frame via
// DvbTLinkParams. NB-DVB-T is a pure fs-scaling of the fixed 2K structure, so the
// bandwidth mode (NbBandwidth::Bw333kHz / Bw1MHz / Bw2MHz) only sets the sample
// rate for RF upconversion; the baseband frame is identical.
let params = DvbTFrameParams {
    link: DvbTLinkParams {
        guard: GuardInterval::G1_8,
        constellation: ConstellationOrder::Qpsk,   // QPSK / 16-QAM / 64-QAM
        code_rate: PunctureRate::R1_2,             // K=7 punctured conv rate
    },
    frame_number: 0,                                // 0..=3 within the super-frame
    cell_id: 0,                                     // this frame's cell-id byte
};
let modulator = DvbTFrameMod::new(params);

// `payload` is treated as the MPEG-TS payload bytes (packetized + energy-
// dispersed inside). One frame spans at least 68 symbols (a full TPS block).
let payload: Vec<u8> = (0..184).map(|i| (i * 37 + 11) as u8).collect();
let frame = modulator.modulate(&payload);

// `frame.iq` is baseband time-domain IQ; `frame.n_symbols` / `samples_per_symbol`
// describe its layout. To place it on air at a bandwidth mode's sample rate, use
// `NbBandwidth::Bw1MHz.fs()` as the DAC/resampler rate.
let _fs = NbBandwidth::Bw1MHz.fs();
```

### Spectral shaping on DVB-T

Two of the three levers are available here as builders. The edge guard is not:
DVB-T's extreme carriers (active indices `0` and `1704`) are *mandatory*
continual pilots that conformant receivers rely on, so they cannot be nulled or
moved inward. In exchange DVB-T needs no edge guard — the standard already
leaves 343 of 2048 bins inactive, which is the room a mask filters into.

Two other things differ from COFDM, both because the frame is preamble-less:
*every* symbol is CP-bearing and therefore tapered (there is no Schmidl & Cox
region to skip), and there is no acquisition budget to respect — only the guard
budget. Work through [Choosing the numbers](#choosing-the-numbers) above; the
DVB-T-specific constraint is the back-off ceiling in step 1.

```rust
use orion_sdr::modulate::DvbTFrameMod;
use orion_sdr::multicarrier::TxLowpass;

// 1. Back-off. cp_len/2 = 128 would be the COFDM answer, but DVB-T's pilot-
//    interpolated equalizer makes the back-off cost sensitivity: 32 is free,
//    42 costs ~1 dB, 64 costs ~6 dB. Take 32 and accept 32 samples of slack —
//    the 256-sample guard buys nothing extra here.
let cp_len = GuardInterval::G1_8.cp_len_2k();                    // 256
let backoff = 32usize;
let slack = (cp_len - backoff).min(backoff);                     // 32

// 2. Mask length. The suggestion is the *shortest* filter that reaches its stop
//    band inside DVB-T's null band — and on DVB-T that shortest filter is also
//    the practical one, since 45 taps' group delay of 22 is most of the slack.
let taps = TxLowpass::taps_for_null_band(2048, 852, 60.0);       // 45
let mask = DvbTFrameMod::tx_lowpass_for_2k(taps, 60.0);          // group delay 22
let roll_off = 8usize;
assert!(mask.fits_guard(cp_len, roll_off, backoff));             // 22 + 8 <= 32
assert!(slack >= roll_off + mask.group_delay());

let frame = DvbTFrameMod::new(params)
    .with_symbol_window(roll_off) // raised-cosine taper, samples per edge
    .with_tx_lowpass(mask)        // cutoff placed against the fixed ±852 band edge
    .modulate(&payload);
```

Pair with `DvbTFrameDemod::with_rx_window_backoff(backoff)` — the same value, on
the same link.

**Do not size DVB-T shaping against the 85-sample aliasing cap.**
`DVB_T_MAX_RX_WINDOW_BACKOFF` is where the pilot interpolation *aliases*, not
where it stops working: at 85 the link does not close at any SNR, and at 64 it
costs ~6 dB. The usable slack is 32 samples for free, 42 for about a dB, and it
saturates there from G1/16 onward regardless of guard interval. A longer guard
buys delay-spread tolerance, not shaping room. See
[dvb.md](dvb.md#choosing-a-guard-interval-for-datv) and the measurements in
[performance.md](performance.md#the-rx-window-back-off-costs-sensitivity-well-before-it-aliases).

Nothing about the *carriers* changes: the taper touches only time-domain guard
samples and the mask is a linear channel the scattered-pilot equalizer absorbs,
so the continual, scattered, and TPS pilots and the payload all come through
untouched. Measured on a conformant frame: the null band drops 66 dB with in-band
power unchanged to within 0.1 dB.

## DVB-T Super-Frame Modulator (four frames)

`DvbTSuperFrameMod` sequences **four** consecutive conformant frames into one
super-frame (§4.4/§4.6): the TPS sync word alternates each frame, the 16-bit cell
id is split across them (b15..b8 in frames 1 & 3, b7..b0 in 2 & 4), and the frame
number counts 0..3. It drives the single-frame modulator above; the RX is
`demodulate::DvbTSuperFrameDemod`. See [dvb.md](dvb.md) (Frame transport) for the
structure.

```rust
use orion_sdr::{
    fec::PunctureRate,
    modulate::{ConstellationOrder, DvbTSuperFrameMod, DvbTSuperFrameParams},
    waveform::dvb_t::{DvbTLinkParams, GuardInterval},
};

// Same shared link set as DvbTFrameParams, but with the FULL 16-bit cell id
// (split across the four frames) and no per-frame number — the driver derives it.
let params = DvbTSuperFrameParams {
    link: DvbTLinkParams {
        guard: GuardInterval::G1_8,
        constellation: ConstellationOrder::Qpsk,
        code_rate: PunctureRate::R1_2,
    },
    cell_id: 0xBEEF,
};

// The payload is split into four contiguous parts, one per frame.
let payload: Vec<u8> = (0..700).map(|i| (i * 37 + 11) as u8).collect();
let sf = DvbTSuperFrameMod::new(params).modulate(&payload);

// `sf.iq` is the four frames concatenated; `sf.symbols_per_frame` /
// `sf.samples_per_symbol` and `sf.frame_payload_lens` (the per-frame byte counts)
// are what the RX needs to re-slice and trim.
let _n_symbols = sf.n_symbols(); // 4 · symbols_per_frame
```

Both shaping builders carry over — `DvbTSuperFrameMod::with_symbol_window` and
`with_tx_lowpass` — but they apply at different scopes, deliberately. The taper
is per-symbol, so it propagates to each constituent frame; the mask is a
spectral filter and runs **once over the concatenated four frames**. Filtering
each frame separately would leave the filter's edge transient at all three
interior seams, which are continuous on air.

> **Receiver limitation on the super-frame path.** `DvbTSuperFrameDemod` hands
> each frame a sub-buffer starting exactly at that frame's first sample, and the
> last frame's sub-buffer is exactly one frame long — so it has *no* tolerance
> for a guard-interval acquisition that lands off zero. A symbol taper biases
> that acquisition a few samples early (~`roll_off/2`), and since the search
> range cannot express a negative offset, the argmax wraps to nearly a whole
> symbol late. A long mask reaches the same place by smearing the cyclic-prefix
> correlation. In practice today: **shape the super-frame with a short mask**
> (45 taps is measured-safe; 89 is not), and use the taper on the single-frame
> path, where any lead-in ahead of the frame — as a real receiver always has —
> absorbs the bias. `python/tests/test_spectral_shaping.py`'s
> `TestDvbTZeroLeadInAcquisition` pins the boundary.
