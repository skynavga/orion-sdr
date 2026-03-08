# Modulator Usage

Usage patterns and examples for all modulators in **orion-sdr**: CW, AM, SSB, FM, PM.

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
