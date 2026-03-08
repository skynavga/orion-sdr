// src/demodulate/qam.rs
use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};

// ── compile-time validation ──────────────────────────────────────────────────

const fn check_bits(bits: usize) {
    assert!(bits == 4 || bits == 6 || bits == 8,
        "QamDecider: BITS must be 4 (QAM-16), 6 (QAM-64), or 8 (QAM-256)");
}

// ── decision threshold table ─────────────────────────────────────────────────

fn axis_scale(bits: usize) -> f32 {
    let m = 1usize << (bits / 2);
    let avg_e_total = 2.0 * ((m * m - 1) as f64) / 3.0;
    (1.0 / avg_e_total.sqrt()) as f32
}

/// Build the M-1 decision thresholds for one axis, in normalized units.
///
/// Unnormalized levels are odd integers {-(M-1), …, -1, +1, …, +(M-1)}.
/// Midpoints between adjacent levels are even integers {-(M-2), …, 0, …, +(M-2)}.
/// Multiplied by the same normalization scale used in the mapper.
///
/// Table has 15 entries (enough for up to M=16).  Only the first M-1 are used.
/// Entries are in ascending order so the threshold loop is a simple scan.
const fn build_threshold_table(bits: usize, scale: f32) -> [f32; 15] {
    let k = bits / 2;
    let m = 1usize << k;
    let mut table = [0.0f32; 15];
    // Midpoints in unnormalized space: -(M-2), -(M-4), …, 0, …, +(M-2)
    // i.e. even integers from -(M-2) to +(M-2), that's M-1 values.
    let mut j = 0usize;
    while j < m - 1 {
        let mid_unnorm = (2 * j) as f32 - (m - 2) as f32; // -(M-2)+2j
        table[j] = mid_unnorm * scale;
        j += 1;
    }
    table
}

// ── QamDemod ─────────────────────────────────────────────────────────────────

/// QAM soft-symbol estimator: C32 IQ → C32 soft symbol.
/// Re(output) is the I decision metric, Im(output) the Q decision metric.
/// Coherent passthrough with gain normalization; order-independent.
#[derive(Debug, Clone, Copy)]
pub struct QamDemod {
    gain: f32,
}

impl QamDemod {
    pub fn new(gain: f32) -> Self { Self { gain } }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
}

impl Block for QamDemod {
    type In  = C32;
    type Out = C32;

    #[inline(always)]
    fn process(&mut self, input: &[C32], output: &mut [C32]) -> WorkReport {
        let n = input.len().min(output.len());
        let g = self.gain;
        let mut i = 0;
        let nn = n & !3;
        while i < nn {
            output[i]   = C32::new(g * input[i].re,   g * input[i].im);
            output[i+1] = C32::new(g * input[i+1].re, g * input[i+1].im);
            output[i+2] = C32::new(g * input[i+2].re, g * input[i+2].im);
            output[i+3] = C32::new(g * input[i+3].re, g * input[i+3].im);
            i += 4;
        }
        while i < n {
            output[i] = C32::new(g * input[i].re, g * input[i].im);
            i += 1;
        }
        WorkReport { in_read: n, out_written: n }
    }
}

// ── QamDecider ───────────────────────────────────────────────────────────────

/// QAM hard decision: C32 soft symbol → `BITS` u8 bits (LSBs).
///
/// Matches `QamMapper<BITS>` Gray coding.  For each axis:
///   1. Count how many of the M-1 thresholds the value exceeds → natural index.
///   2. Gray-encode the natural index → Gray index.
///   3. Emit `BITS/2` bits MSB-first from the Gray index.
///
/// Output layout: `BITS/2` I-bits then `BITS/2` Q-bits per symbol,
/// matching the input layout of `QamMapper<BITS>`.
///
/// `in_read = n_syms`, `out_written = n_syms * BITS`.
#[derive(Debug, Clone, Copy)]
pub struct QamDecider<const BITS: usize> {
    thresholds: [f32; 15],
}

impl<const BITS: usize> QamDecider<BITS> {
    pub fn new() -> Self {
        const { check_bits(BITS); }
        Self { thresholds: build_threshold_table(BITS, axis_scale(BITS)) }
    }

    /// Map one soft axis value to `K = BITS/2` Gray-coded bit bytes (LSBs).
    #[inline(always)]
    fn decide_axis(&self, v: f32, out: &mut [u8], base: usize) {
        let k = BITS / 2;
        let m = 1 << k;
        // Count thresholds exceeded → natural index in [0, M)
        let mut nat = 0usize;
        let mut t = 0;
        while t < m - 1 {
            if v > self.thresholds[t] { nat += 1; }
            t += 1;
        }
        // Binary → Gray
        let gray = nat ^ (nat >> 1);
        // Emit K bits MSB-first
        let mut b = 0;
        while b < k {
            out[base + b] = ((gray >> (k - 1 - b)) & 1) as u8;
            b += 1;
        }
    }
}

impl<const BITS: usize> Default for QamDecider<BITS> {
    fn default() -> Self { Self::new() }
}

impl<const BITS: usize> Block for QamDecider<BITS> {
    type In  = C32;
    type Out = u8;

    #[inline(always)]
    fn process(&mut self, input: &[C32], output: &mut [u8]) -> WorkReport {
        let n_syms = input.len().min(output.len() / BITS);
        let k = BITS / 2;
        let mut i = 0;
        let nn = n_syms & !3;
        while i < nn {
            self.decide_axis(input[i].re,   output, i*BITS);
            self.decide_axis(input[i].im,   output, i*BITS+k);
            self.decide_axis(input[i+1].re, output, (i+1)*BITS);
            self.decide_axis(input[i+1].im, output, (i+1)*BITS+k);
            self.decide_axis(input[i+2].re, output, (i+2)*BITS);
            self.decide_axis(input[i+2].im, output, (i+2)*BITS+k);
            self.decide_axis(input[i+3].re, output, (i+3)*BITS);
            self.decide_axis(input[i+3].im, output, (i+3)*BITS+k);
            i += 4;
        }
        while i < n_syms {
            self.decide_axis(input[i].re, output, i*BITS);
            self.decide_axis(input[i].im, output, i*BITS+k);
            i += 1;
        }
        WorkReport { in_read: n_syms, out_written: n_syms * BITS }
    }
}

// ── type aliases ─────────────────────────────────────────────────────────────

pub type Qam16Decider  = QamDecider<4>;
pub type Qam64Decider  = QamDecider<6>;
pub type Qam256Decider = QamDecider<8>;
