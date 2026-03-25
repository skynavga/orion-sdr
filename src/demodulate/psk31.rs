// src/demodulate/psk31.rs
//
// PSK31 demodulators: BPSK31 and QPSK31.
//
// Both modes use decision-feedback matched filtering over the full symbol
// period.  The half-cosine crossfade mixes the previous and current phasors:
//
//   s[n] = p0·(1−h[n]) + p1·h[n],   h[n] = 0.5 − 0.5·cos(π·n/(sps−1))
//
// By subtracting the known contribution of the previous symbol estimate
// (`prev_sym`) and weighting by h[n], we recover a clean estimate of p1:
//
//   corrected[n] = s[n] − prev_sym·(1−h[n])
//   sym = Σ h[n]·corrected[n] / Σ h[n]²
//       = p1   (exactly, in the noiseless case)
//
// This integrates all sps samples, maximising SNR.  A first-order
// decision-directed PLL then corrects residual carrier phase at each
// symbol boundary (AFC):
//
//   sym_c     = sym · exp(−j·phase_acc)          (phase correction)
//   decision  = sym_c · conj(prev_sym)            (differential detection)
//   phase_err = Im(decision · conj(decided)) / |decision|
//   phase_acc += K · phase_err                   (loop update, K = 0.05)
//
// BPSK31: Re(decision) → soft bit (positive = bit 1, no phase change).
// QPSK31: [Re(decision), Im(decision)] → soft dibit for Viterbi.

use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::Rotator;
use crate::codec::psk31_conv::viterbi_decode;
use crate::modulate::psk31::psk31_sps;

fn make_hann(sps: usize) -> Vec<f32> {
    if sps == 0 { return Vec::new(); }
    if sps == 1 { return vec![1.0]; }
    let denom = (sps - 1) as f32;
    (0..sps)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::PI * i as f32 / denom).cos())
        .collect()
}

const BPSK31_LOOP_GAIN: f32 = 0.05;
const QPSK31_LOOP_GAIN: f32 = 0.05;

/// BPSK31 hard decision on a differential symbol.
/// Returns ±1.0 based on the sign of the real component.
#[inline(always)]
pub(crate) fn hard_decide_dbpsk(d_re: f32) -> f32 {
    if d_re >= 0.0 { 1.0 } else { -1.0 }
}

/// QPSK31 hard decision on a differential symbol.
/// Returns the nearest unit-axis phasor: (±1,0) or (0,±1).
#[inline(always)]
pub(crate) fn hard_decide_dqpsk(d_re: f32, d_im: f32) -> (f32, f32) {
    if d_re.abs() >= d_im.abs() {
        if d_re >= 0.0 { (1.0, 0.0) } else { (-1.0, 0.0) }
    } else {
        if d_im >= 0.0 { (0.0, 1.0) } else { (0.0, -1.0) }
    }
}

// ── BPSK31 demodulator ────────────────────────────────────────────────────────

/// BPSK31 decision-feedback matched-filter demodulator.
///
/// Input: `C32` IQ samples.
/// Output: one `f32` per symbol = `Re(sym · conj(prev_sym))`.
///   Positive → bit 1 (no phase change), negative → bit 0 (phase flip).
#[derive(Debug, Clone)]
pub struct Bpsk31Demod {
    sps:         usize,
    gain:        f32,
    count:       usize,
    acc:         C32,
    hann:        Vec<f32>,
    hann_sq_sum: f32,     // Σ h[n]²
    prev_sym:    C32,
    rot:         Option<Rotator>,
    phase_acc:   f32,     // accumulated phase correction (rad)
    loop_gain:   f32,     // first-order loop gain K
}

impl Bpsk31Demod {
    /// Create a new BPSK31 demodulator.
    /// `rf_hz` = 0.0 → baseband input; non-zero → down-mix by `rf_hz`.
    pub fn new(fs: f32, rf_hz: f32, gain: f32) -> Self {
        Self::new_with_offset(fs, rf_hz, gain, 0)
    }

    /// As `new` but start integration at a given sample offset (for sync alignment).
    pub fn new_with_offset(fs: f32, rf_hz: f32, gain: f32, offset: usize) -> Self {
        let sps = psk31_sps(fs);
        let rot = if rf_hz != 0.0 { Some(Rotator::new(-rf_hz, fs)) } else { None };
        let count = if offset % sps == 0 { 0 } else { sps - (offset % sps) };
        let hann = make_hann(sps);
        let hann_sq_sum: f32 = hann.iter().map(|&h| h * h).sum();
        Self {
            sps,
            gain,
            count,
            acc: C32::new(0.0, 0.0),
            hann,
            hann_sq_sum,
            prev_sym: C32::new(1.0, 0.0),
            rot,
            phase_acc: 0.0,
            loop_gain: BPSK31_LOOP_GAIN,
        }
    }

    pub fn set_gain(&mut self, g: f32) { self.gain = g; }

    pub fn reset(&mut self) {
        self.count    = 0;
        self.acc      = C32::new(0.0, 0.0);
        self.prev_sym = C32::new(1.0, 0.0);
        self.phase_acc = 0.0;
        if let Some(r) = &mut self.rot { r.reset_phase(); }
    }
}

impl Block for Bpsk31Demod {
    type In  = C32;
    type Out = f32;

    fn process(&mut self, input: &[C32], output: &mut [f32]) -> WorkReport {
        let mut in_pos  = 0;
        let mut out_pos = 0;

        while in_pos < input.len() && out_pos < output.len() {
            let mut s = input[in_pos];
            // Down-mix if carrier offset configured.
            if let Some(rot) = &mut self.rot {
                let r = rot.next();
                let re = s.re * r.re - s.im * r.im;
                let im = s.im * r.re + s.re * r.im;
                s = C32::new(re, im);
            }
            // Decision-feedback cancellation: remove prev_sym·(1−h[n]) contribution.
            let h = self.hann[self.count];
            let one_minus_h = 1.0 - h;
            let corr_re = s.re - self.prev_sym.re * one_minus_h;
            let corr_im = s.im - self.prev_sym.im * one_minus_h;
            // Accumulate h[n]·corrected[n].
            self.acc.re += h * corr_re;
            self.acc.im += h * corr_im;
            self.count += 1;

            // Dump at the end of each symbol period.
            if self.count == self.sps {
                let scale = self.gain / self.hann_sq_sum;
                let sym = C32::new(self.acc.re * scale, self.acc.im * scale);
                self.acc   = C32::new(0.0, 0.0);
                self.count = 0;

                // Phase correction: rotate sym by -phase_acc.
                let (sin_pa, cos_pa) = self.phase_acc.sin_cos();
                let sym_re = sym.re * cos_pa + sym.im * sin_pa;
                let sym_im = sym.im * cos_pa - sym.re * sin_pa;

                // Differential detection: Re(sym_c · conj(prev_sym)).
                let d_re = sym_re * self.prev_sym.re + sym_im * self.prev_sym.im;
                output[out_pos] = d_re;
                out_pos += 1;

                // Decision-directed phase error.
                let dec_re = hard_decide_dbpsk(d_re);
                let d_im   = sym_im * self.prev_sym.re - sym_re * self.prev_sym.im;
                let cross_im  = d_im * dec_re;   // dec_im = 0
                let mag_sq    = d_re * d_re + d_im * d_im;
                let phase_err = if mag_sq > 1e-6 { cross_im / mag_sq.sqrt() } else { 0.0 };
                self.phase_acc += self.loop_gain * phase_err;
                if self.phase_acc >  std::f32::consts::PI { self.phase_acc -= std::f32::consts::TAU; }
                if self.phase_acc < -std::f32::consts::PI { self.phase_acc += std::f32::consts::TAU; }

                self.prev_sym = C32::new(sym_re, sym_im);
            }

            in_pos += 1;
        }

        WorkReport { in_read: in_pos, out_written: out_pos }
    }
}

// ── BPSK31 hard decider ───────────────────────────────────────────────────────

/// Hard decision on BPSK31 soft values.
/// Input: `f32` soft bits (positive = bit 1 / no phase change).
/// Output: `u8` bits (0 or 1).
#[derive(Debug, Clone, Copy, Default)]
pub struct Bpsk31Decider;

impl Bpsk31Decider {
    pub fn new() -> Self { Self }
}

impl Block for Bpsk31Decider {
    type In  = f32;
    type Out = u8;

    #[inline(always)]
    fn process(&mut self, input: &[f32], output: &mut [u8]) -> WorkReport {
        let n = input.len().min(output.len());
        let mut i = 0;
        let nn = n & !3;
        while i < nn {
            output[i]   = u8::from(input[i]   >= 0.0);
            output[i+1] = u8::from(input[i+1] >= 0.0);
            output[i+2] = u8::from(input[i+2] >= 0.0);
            output[i+3] = u8::from(input[i+3] >= 0.0);
            i += 4;
        }
        while i < n {
            output[i] = u8::from(input[i] >= 0.0);
            i += 1;
        }
        WorkReport { in_read: n, out_written: n }
    }
}

// ── QPSK31 demodulator ────────────────────────────────────────────────────────

/// QPSK31 decision-feedback matched-filter demodulator.
///
/// Input: `C32` IQ samples.
/// Output: two `f32` per symbol = `[Re(d), Im(d)]` where `d = sym·conj(prev)`.
/// These are the soft dibit values for Viterbi decoding.
#[derive(Debug, Clone)]
pub struct Qpsk31Demod {
    sps:         usize,
    gain:        f32,
    count:       usize,
    acc:         C32,
    hann:        Vec<f32>,
    hann_sq_sum: f32,
    prev_sym:    C32,
    rot:         Option<Rotator>,
    phase_acc:   f32,     // accumulated phase correction (rad)
    loop_gain:   f32,     // first-order loop gain K
}

impl Qpsk31Demod {
    pub fn new(fs: f32, rf_hz: f32, gain: f32) -> Self {
        let sps = psk31_sps(fs);
        let rot = if rf_hz != 0.0 { Some(Rotator::new(-rf_hz, fs)) } else { None };
        let hann = make_hann(sps);
        let hann_sq_sum: f32 = hann.iter().map(|&h| h * h).sum();
        Self {
            sps,
            gain,
            count: 0,
            acc: C32::new(0.0, 0.0),
            hann,
            hann_sq_sum,
            prev_sym: C32::new(1.0, 0.0),
            rot,
            phase_acc: 0.0,
            loop_gain: QPSK31_LOOP_GAIN,
        }
    }

    pub fn set_gain(&mut self, g: f32) { self.gain = g; }

    pub fn reset(&mut self) {
        self.count    = 0;
        self.acc      = C32::new(0.0, 0.0);
        self.prev_sym = C32::new(1.0, 0.0);
        self.phase_acc = 0.0;
        if let Some(r) = &mut self.rot { r.reset_phase(); }
    }
}

impl Block for Qpsk31Demod {
    type In  = C32;
    type Out = f32;

    /// Output: pairs of soft values `[Re(d), Im(d)]` per symbol.
    fn process(&mut self, input: &[C32], output: &mut [f32]) -> WorkReport {
        let mut in_pos  = 0;
        let mut out_pos = 0;

        while in_pos < input.len() && out_pos + 1 < output.len() {
            let mut s = input[in_pos];
            if let Some(rot) = &mut self.rot {
                let r = rot.next();
                let re = s.re * r.re - s.im * r.im;
                let im = s.im * r.re + s.re * r.im;
                s = C32::new(re, im);
            }
            // Decision-feedback cancellation + Hann-weighted accumulation.
            let h = self.hann[self.count];
            let one_minus_h = 1.0 - h;
            let corr_re = s.re - self.prev_sym.re * one_minus_h;
            let corr_im = s.im - self.prev_sym.im * one_minus_h;
            self.acc.re += h * corr_re;
            self.acc.im += h * corr_im;
            self.count += 1;
            in_pos += 1;

            // Dump at the end of each symbol period.
            if self.count == self.sps {
                let scale = self.gain / self.hann_sq_sum;
                let sym = C32::new(self.acc.re * scale, self.acc.im * scale);
                self.acc   = C32::new(0.0, 0.0);
                self.count = 0;

                // Phase correction: rotate sym by -phase_acc.
                let (sin_pa, cos_pa) = self.phase_acc.sin_cos();
                let sym_re = sym.re * cos_pa + sym.im * sin_pa;
                let sym_im = sym.im * cos_pa - sym.re * sin_pa;

                // Differential: d = sym_c · conj(prev_sym)
                let d_re = sym_re * self.prev_sym.re + sym_im * self.prev_sym.im;
                let d_im = sym_im * self.prev_sym.re - sym_re * self.prev_sym.im;
                output[out_pos]   = d_re;
                output[out_pos+1] = d_im;
                out_pos += 2;

                // Decision-directed phase error.
                let (dec_re, dec_im) = hard_decide_dqpsk(d_re, d_im);
                let cross_im  = d_im * dec_re - d_re * dec_im;
                let mag_sq    = d_re * d_re + d_im * d_im;
                let phase_err = if mag_sq > 1e-6 { cross_im / mag_sq.sqrt() } else { 0.0 };
                self.phase_acc += self.loop_gain * phase_err;
                if self.phase_acc >  std::f32::consts::PI { self.phase_acc -= std::f32::consts::TAU; }
                if self.phase_acc < -std::f32::consts::PI { self.phase_acc += std::f32::consts::TAU; }

                self.prev_sym = C32::new(sym_re, sym_im);
            }
        }

        WorkReport { in_read: in_pos, out_written: out_pos }
    }
}

// ── QPSK31 Viterbi decider ────────────────────────────────────────────────────

/// QPSK31 Viterbi decoder.
///
/// Buffers soft dibits `[Re, Im]` from `Qpsk31Demod` and runs the Viterbi
/// algorithm when `flush()` is called (or when the block is fully consumed).
///
/// The `Block::process` implementation buffers all input and emits 0 bytes
/// until `flush()` is called.
#[derive(Debug, Clone, Default)]
pub struct Qpsk31Decider {
    soft_buf: Vec<f32>,
}

impl Qpsk31Decider {
    pub fn new() -> Self { Self::default() }

    /// Run Viterbi on the accumulated soft dibits and append decoded bits to `output`.
    pub fn flush(&mut self, output: &mut Vec<u8>) {
        if !self.soft_buf.is_empty() {
            let decoded = viterbi_decode(&self.soft_buf);
            output.extend_from_slice(&decoded);
            self.soft_buf.clear();
        }
    }
}

impl Block for Qpsk31Decider {
    type In  = f32;
    type Out = u8;

    fn process(&mut self, input: &[f32], _output: &mut [u8]) -> WorkReport {
        self.soft_buf.extend_from_slice(input);
        WorkReport { in_read: input.len(), out_written: 0 }
    }
}
