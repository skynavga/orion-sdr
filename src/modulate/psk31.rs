// src/modulate/psk31.rs
//
// PSK31 modulators: BPSK31 and QPSK31.
//
// Both modes share:
//   - 31.25 baud rate
//   - Raised-cosine (α=1) pulse shaping via Hann-windowed cross-fade
//   - Differential phase encoding (0 bit = phase change, 1 bit = no phase change)
//   - Varicode character encoding
//   - Preamble (0-bits) and postamble (1-bits)
//
// QPSK31 additionally uses a rate-1/2, K=5 convolutional encoder.
//
// Reference: Peter Martinez G3PLX, "PSK31: A New Radio-Teletype Mode" (1998).

use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::Rotator;
use crate::codec::varicode::VaricodeEncoder;

// ── Shared constants ──────────────────────────────────────────────────────────

/// PSK31 symbol rate (baud).
pub const PSK31_BAUD: f32 = 31.25;

/// Samples per symbol at 8 kHz.
pub const PSK31_SPS_8000: usize = 256;

/// Samples per symbol at 12 kHz.
pub const PSK31_SPS_12000: usize = 384;

/// Default preamble length in bits.
pub const PSK31_PREAMBLE_BITS: usize = 32;

/// Default postamble length in bits.
pub const PSK31_POSTAMBLE_BITS: usize = 32;

/// Compute the number of samples per PSK31 symbol for a given sample rate.
pub fn psk31_sps(fs: f32) -> usize {
    (fs / PSK31_BAUD).round() as usize
}

/// Precompute the half-cosine crossfade window for PSK31 pulse shaping.
///
/// `hann[n] = 0.5 − 0.5·cos(π·n / (sps − 1))`, n ∈ [0, sps).
///
/// This is a one-sided raised-cosine (half-Hann) that goes from 0 at n=0
/// to 1 at n=sps−1, providing a smooth crossfade from the previous phasor
/// to the current one over one symbol period.
fn make_hann(sps: usize) -> Vec<f32> {
    if sps == 0 { return Vec::new(); }
    if sps == 1 { return vec![1.0]; }
    let denom = (sps - 1) as f32;
    (0..sps)
        .map(|i| {
            let t = std::f32::consts::PI * (i as f32) / denom;
            0.5 - 0.5 * t.cos()
        })
        .collect()
}


/// Write one pulse-shaped symbol into `out[0..sps]`.
///
/// Computes a per-sample Hann-windowed linear interpolation between phasors
/// `p0` (previous symbol's phasor) and `p1` (current symbol's phasor):
///
///   `sample[n] = gain · (p0 + hann[n] · (p1 − p0))`
///
/// For BPSK31: p0/p1 ∈ {(1,0), (−1,0)}.
/// For QPSK31: p0/p1 ∈ {(1,0), (0,1), (−1,0), (0,−1)}.
#[inline(always)]
fn write_symbol(out: &mut [C32], p0: C32, p1: C32, gain: f32, hann: &[f32]) {
    let dr = p1.re - p0.re;
    let di = p1.im - p0.im;
    let n = out.len().min(hann.len());
    let mut i = 0;
    let nn = n & !3;
    while i < nn {
        let h0 = hann[i];
        let h1 = hann[i+1];
        let h2 = hann[i+2];
        let h3 = hann[i+3];
        out[i]   = C32::new(gain * (p0.re + h0 * dr), gain * (p0.im + h0 * di));
        out[i+1] = C32::new(gain * (p0.re + h1 * dr), gain * (p0.im + h1 * di));
        out[i+2] = C32::new(gain * (p0.re + h2 * dr), gain * (p0.im + h2 * di));
        out[i+3] = C32::new(gain * (p0.re + h3 * dr), gain * (p0.im + h3 * di));
        i += 4;
    }
    while i < n {
        let h = hann[i];
        out[i] = C32::new(gain * (p0.re + h * dr), gain * (p0.im + h * di));
        i += 1;
    }
}

// ── BPSK31 modulator ──────────────────────────────────────────────────────────

/// BPSK31 modulator.
///
/// Accepts a stream of differential bits (0 = phase flip, 1 = no phase change)
/// and produces pulse-shaped IQ samples at the configured sample rate.
#[derive(Debug, Clone)]
pub struct Bpsk31Mod {
    fs: f32,
    sps: usize,
    rf_hz: f32,
    gain: f32,
    current_phase: f32,  // +1.0 or -1.0
    hann: Vec<f32>,
}

impl Bpsk31Mod {
    /// Create a new BPSK31 modulator.
    ///
    /// * `fs`    — sample rate (Hz)
    /// * `rf_hz` — IF upconversion carrier (0.0 = baseband)
    /// * `gain`  — output amplitude scale
    pub fn new(fs: f32, rf_hz: f32, gain: f32) -> Self {
        let sps = psk31_sps(fs);
        Self {
            fs,
            sps,
            rf_hz,
            gain,
            current_phase: 1.0,
            hann: make_hann(sps),
        }
    }

    pub fn set_gain(&mut self, g: f32) { self.gain = g; }

    pub fn reset(&mut self) {
        self.current_phase = 1.0;
    }

    /// Modulate a text string with preamble and postamble → IQ samples.
    pub fn modulate_text(
        &mut self,
        text: &[u8],
        preamble_bits: usize,
        postamble_bits: usize,
    ) -> Vec<C32> {
        let mut enc = VaricodeEncoder::new();
        enc.push_preamble(preamble_bits);
        for &b in text {
            enc.push_byte(b);
        }
        enc.push_postamble(postamble_bits);
        let bits = enc.drain_bits();
        self.modulate_bits(&bits)
    }

    /// Modulate raw differential bits (0 = flip, 1 = no flip) → IQ samples.
    /// Output length = `bits.len() * sps`.
    pub fn modulate_bits(&mut self, bits: &[u8]) -> Vec<C32> {
        let total = bits.len() * self.sps;
        let mut out = vec![C32::new(0.0, 0.0); total];
        let mut prev_p = C32::new(self.current_phase, 0.0);

        for (k, &bit) in bits.iter().enumerate() {
            // DBPSK: bit 0 → flip phase; bit 1 → no change.
            if bit == 0 {
                self.current_phase = -self.current_phase;
            }
            let cur_p = C32::new(self.current_phase, 0.0);
            let base = k * self.sps;
            write_symbol(&mut out[base..base + self.sps], prev_p, cur_p, self.gain, &self.hann);
            prev_p = cur_p;
        }

        if self.rf_hz != 0.0 {
            let mut rot = Rotator::new(self.rf_hz, self.fs);
            rot.rotate_block(&out.clone(), &mut out);
        }

        out
    }
}

impl Block for Bpsk31Mod {
    type In  = u8;
    type Out = C32;

    fn process(&mut self, input: &[u8], output: &mut [C32]) -> WorkReport {
        let max_bits = output.len() / self.sps;
        let n = input.len().min(max_bits);
        if n == 0 { return WorkReport { in_read: 0, out_written: 0 }; }
        let iq = self.modulate_bits(&input[..n]);
        let written = iq.len().min(output.len());
        output[..written].copy_from_slice(&iq[..written]);
        WorkReport { in_read: n, out_written: written }
    }
}

// ── QPSK31 modulator ──────────────────────────────────────────────────────────

// QPSK31 phase-step phasors indexed by dibit = g0*2 + g1.
//
// Sign convention: positive soft value = coded bit likely 0
// (matches Viterbi decoder expectation).
//
// After differential detection d = sym × conj(prev):
//   Re(d) soft-demodulates g0: Re>0 → g0=0, Re<0 → g0=1
//   Im(d) soft-demodulates g1: Im>0 → g1=0, Im<0 → g1=1
//
// Mapping chosen to satisfy this convention:
//   (g0=0, g1=0): d = ( 1,  0)  → step = ( 1,  0)
//   (g0=0, g1=1): d = ( 0, -1)  → step = ( 0, -1)  [Im<0 → g1=1]
//   (g0=1, g1=0): d = ( 0, +1)  → step = ( 0, +1)  [Im>0 → g1=0]
//   (g0=1, g1=1): d = (-1,  0)  → step = (-1,  0)
const QPSK31_PHASE_STEP: [C32; 4] = [
    C32 { re:  1.0, im:  0.0 }, // dibit 0: g0=0, g1=0 →   0°
    C32 { re:  0.0, im: -1.0 }, // dibit 1: g0=0, g1=1 → -90° (270°)
    C32 { re:  0.0, im:  1.0 }, // dibit 2: g0=1, g1=0 →  90°
    C32 { re: -1.0, im:  0.0 }, // dibit 3: g0=1, g1=1 → 180°
];

/// QPSK31 modulator.
///
/// Convolutional-encodes input bits at rate-1/2 (G0=25, G1=23), maps the
/// resulting dibits to DQPSK symbols with Hann-windowed pulse shaping.
#[derive(Debug, Clone)]
pub struct Qpsk31Mod {
    fs: f32,
    sps: usize,
    rf_hz: f32,
    gain: f32,
    current_phase: C32,  // current phasor ∈ {(1,0),(0,1),(−1,0),(0,−1)}
    hann: Vec<f32>,
    enc_sr: u8,           // convolutional encoder shift register (4 bits)
}

impl Qpsk31Mod {
    /// Create a new QPSK31 modulator.
    pub fn new(fs: f32, rf_hz: f32, gain: f32) -> Self {
        let sps = psk31_sps(fs);
        Self {
            fs,
            sps,
            rf_hz,
            gain,
            current_phase: C32::new(1.0, 0.0),
            hann: make_hann(sps),
            enc_sr: 0,
        }
    }

    pub fn set_gain(&mut self, g: f32) { self.gain = g; }

    pub fn reset(&mut self) {
        self.current_phase = C32::new(1.0, 0.0);
        self.enc_sr = 0;
    }

    /// Modulate text with preamble and postamble → IQ samples.
    pub fn modulate_text(
        &mut self,
        text: &[u8],
        preamble_bits: usize,
        postamble_bits: usize,
    ) -> Vec<C32> {
        let mut enc = VaricodeEncoder::new();
        enc.push_preamble(preamble_bits);
        for &b in text {
            enc.push_byte(b);
        }
        enc.push_postamble(postamble_bits);
        let bits = enc.drain_bits();
        self.modulate_bits(&bits)
    }

    /// Modulate raw information bits → IQ samples.
    /// Each input bit → 2 coded bits (rate-1/2) → 1 DQPSK symbol.
    /// Output length = `bits.len() * sps`.
    pub fn modulate_bits(&mut self, bits: &[u8]) -> Vec<C32> {
        let coded = conv_encode_stateful(bits, &mut self.enc_sr);
        // coded is interleaved [g0_0, g1_0, g0_1, g1_1, ...]
        let n_syms = coded.len() / 2;
        let total = n_syms * self.sps;
        let mut out = vec![C32::new(0.0, 0.0); total];
        let mut prev_p = self.current_phase;

        for k in 0..n_syms {
            let g0 = coded[k * 2];
            let g1 = coded[k * 2 + 1];
            let dibit = (g0 & 1) * 2 + (g1 & 1); // 0..3
            let step = QPSK31_PHASE_STEP[dibit as usize];
            // Differential: new_phasor = current_phase * step
            let np = C32::new(
                self.current_phase.re * step.re - self.current_phase.im * step.im,
                self.current_phase.im * step.re + self.current_phase.re * step.im,
            );
            self.current_phase = np;
            let base = k * self.sps;
            write_symbol(&mut out[base..base + self.sps], prev_p, np, self.gain, &self.hann);
            prev_p = np;
        }

        if self.rf_hz != 0.0 {
            let mut rot = Rotator::new(self.rf_hz, self.fs);
            rot.rotate_block(&out.clone(), &mut out);
        }

        out
    }
}

impl Block for Qpsk31Mod {
    type In  = u8;
    type Out = C32;

    fn process(&mut self, input: &[u8], output: &mut [C32]) -> WorkReport {
        let max_bits = output.len() / self.sps;
        let n = input.len().min(max_bits);
        if n == 0 { return WorkReport { in_read: 0, out_written: 0 }; }
        let iq = self.modulate_bits(&input[..n]);
        let written = iq.len().min(output.len());
        output[..written].copy_from_slice(&iq[..written]);
        WorkReport { in_read: n, out_written: written }
    }
}

// ── Stateful convolutional encoder helper ─────────────────────────────────────

/// Run the convolutional encoder but maintain external shift-register state.
/// Used by `Qpsk31Mod` to encode across successive `modulate_bits` calls.
fn conv_encode_stateful(bits: &[u8], sr: &mut u8) -> Vec<u8> {
    fn parity(x: u8) -> u8 {
        let x = x ^ (x >> 4);
        let x = x ^ (x >> 2);
        let x = x ^ (x >> 1);
        x & 1
    }
    let mut out = Vec::with_capacity(bits.len() * 2);
    for &b in bits {
        let window = ((b & 1) << 4) | (*sr & 0x0F);
        let g0 = parity(window & 0b10101);
        let g1 = parity(window & 0b10011);
        out.push(g0);
        out.push(g1);
        *sr = (*sr >> 1) | ((b & 1) << 3);
    }
    out
}
