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
see [design.md](design.md#multicarrier--ofdm-pipeline) for the FFT
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
