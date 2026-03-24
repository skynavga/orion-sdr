// src/demodulate/psk31.rs
//
// PSK31 demodulators: BPSK31 and QPSK31.
//
// Both modes use Hann-weighted integrate-and-dump over the final quarter of
// each symbol period (n ∈ [3·sps/4, sps)) to form the symbol estimate, then
// apply differential detection.  By the 75% point of the half-cosine crossfade,
// hann[n] ≥ 0.85, so the phasor has settled close to p1 and the weighted
// average is a reliable complex symbol estimate for both BPSK31 and QPSK31.
//
//   sym = (Σ hann[n] · s[n]) / Σ hann[n]   over n ∈ [3·sps/4, sps)
//
//   decision = sym · conj(prev_sym)
//
// BPSK31: Re(decision) → soft bit (positive = bit 1, no phase change).
// QPSK31: [Re(decision), Im(decision)] → soft dibit for Viterbi.

use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::Rotator;
use crate::codec::psk31_conv::viterbi_decode;
use crate::modulate::psk31::{psk31_sps, make_hann_demod};

// ── BPSK31 demodulator ────────────────────────────────────────────────────────

/// BPSK31 integrate-and-dump + differential demodulator.
///
/// Input: `C32` IQ samples.
/// Output: one `f32` per symbol = `Re(sym · conj(prev_sym))`.
///   Positive → bit 1 (no phase change), negative → bit 0 (phase flip).
#[derive(Debug, Clone)]
pub struct Bpsk31Demod {
    sps: usize,
    sps_start: usize,   // 3*sps/4 — accumulation starts here
    gain: f32,
    count: usize,
    acc: C32,
    hann_sum: f32,      // Σ hann[n] for n ∈ [sps_start, sps)
    hann: Vec<f32>,
    prev_sym: C32,
    rot: Option<Rotator>,
}

impl Bpsk31Demod {
    /// Create a new BPSK31 demodulator.
    /// `rf_hz` = 0.0 → baseband input; non-zero → down-mix by `rf_hz` before integration.
    pub fn new(fs: f32, rf_hz: f32, gain: f32) -> Self {
        Self::new_with_offset(fs, rf_hz, gain, 0)
    }

    /// As `new` but start integration at a given sample offset (for sync alignment).
    pub fn new_with_offset(fs: f32, rf_hz: f32, gain: f32, offset: usize) -> Self {
        let sps = psk31_sps(fs);
        let rot = if rf_hz != 0.0 { Some(Rotator::new(-rf_hz, fs)) } else { None };
        // Set count so the first dump occurs after `sps - offset` samples.
        let count = if offset % sps == 0 { 0 } else { sps - (offset % sps) };
        let hann = make_hann_demod(sps);
        let sps_start = 3 * sps / 4;
        let hann_sum: f32 = hann[sps_start..].iter().sum();
        Self {
            sps,
            sps_start,
            gain,
            count,
            acc: C32::new(0.0, 0.0),
            hann_sum,
            hann,
            prev_sym: C32::new(1.0, 0.0),
            rot,
        }
    }

    pub fn set_gain(&mut self, g: f32) { self.gain = g; }

    pub fn reset(&mut self) {
        self.count = 0;
        self.acc = C32::new(0.0, 0.0);
        self.prev_sym = C32::new(1.0, 0.0);
        if let Some(r) = &mut self.rot { r.reset_phase(); }
    }
}

impl Block for Bpsk31Demod {
    type In  = C32;
    type Out = f32;

    fn process(&mut self, input: &[C32], output: &mut [f32]) -> WorkReport {
        let mut in_pos = 0;
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
            // Accumulate only the settled final quarter of the symbol period.
            if self.count >= self.sps_start {
                let w = self.hann[self.count];
                self.acc.re += w * s.re;
                self.acc.im += w * s.im;
            }
            self.count += 1;

            // Dump at the end of each symbol period.
            if self.count == self.sps {
                let scale = self.gain / self.hann_sum;
                let sym = C32::new(self.acc.re * scale, self.acc.im * scale);
                self.acc = C32::new(0.0, 0.0);
                self.count = 0;
                // Differential detection: sym * conj(prev_sym).
                let d_re = sym.re * self.prev_sym.re + sym.im * self.prev_sym.im;
                output[out_pos] = d_re;
                out_pos += 1;
                self.prev_sym = sym;
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

/// QPSK31 integrate-and-dump + differential demodulator.
///
/// Input: `C32` IQ samples.
/// Output: two `f32` per symbol = `[Re(d), Im(d)]` where `d = sym·conj(prev)`.
/// These are the soft dibit values for Viterbi decoding.
#[derive(Debug, Clone)]
pub struct Qpsk31Demod {
    sps: usize,
    sps_start: usize,
    gain: f32,
    count: usize,
    acc: C32,
    hann_sum: f32,
    hann: Vec<f32>,
    prev_sym: C32,
    rot: Option<Rotator>,
}

impl Qpsk31Demod {
    pub fn new(fs: f32, rf_hz: f32, gain: f32) -> Self {
        let sps = psk31_sps(fs);
        let rot = if rf_hz != 0.0 { Some(Rotator::new(-rf_hz, fs)) } else { None };
        let hann = make_hann_demod(sps);
        let sps_start = 3 * sps / 4;
        let hann_sum: f32 = hann[sps_start..].iter().sum();
        Self {
            sps,
            sps_start,
            gain,
            count: 0,
            acc: C32::new(0.0, 0.0),
            hann_sum,
            hann,
            prev_sym: C32::new(1.0, 0.0),
            rot,
        }
    }

    pub fn set_gain(&mut self, g: f32) { self.gain = g; }

    pub fn reset(&mut self) {
        self.count = 0;
        self.acc = C32::new(0.0, 0.0);
        self.prev_sym = C32::new(1.0, 0.0);
        if let Some(r) = &mut self.rot { r.reset_phase(); }
    }
}

impl Block for Qpsk31Demod {
    type In  = C32;
    type Out = f32;

    /// Output: pairs of soft values `[Re(d), Im(d)]` per symbol.
    fn process(&mut self, input: &[C32], output: &mut [f32]) -> WorkReport {
        let mut in_pos = 0;
        let mut out_pos = 0;

        while in_pos < input.len() && out_pos + 1 < output.len() {
            let mut s = input[in_pos];
            if let Some(rot) = &mut self.rot {
                let r = rot.next();
                let re = s.re * r.re - s.im * r.im;
                let im = s.im * r.re + s.re * r.im;
                s = C32::new(re, im);
            }
            // Accumulate only the settled final quarter of the symbol period.
            if self.count >= self.sps_start {
                let w = self.hann[self.count];
                self.acc.re += w * s.re;
                self.acc.im += w * s.im;
            }
            self.count += 1;
            in_pos += 1;

            // Dump at the end of each symbol period.
            if self.count == self.sps {
                let scale = self.gain / self.hann_sum;
                let sym = C32::new(self.acc.re * scale, self.acc.im * scale);
                self.acc = C32::new(0.0, 0.0);
                self.count = 0;
                // Differential: d = sym * conj(prev_sym)
                let d_re = sym.re * self.prev_sym.re + sym.im * self.prev_sym.im;
                let d_im = sym.im * self.prev_sym.re - sym.re * self.prev_sym.im;
                output[out_pos]   = d_re;
                output[out_pos+1] = d_im;
                out_pos += 2;
                self.prev_sym = sym;
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
