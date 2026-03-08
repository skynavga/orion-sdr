// src/modulate/qam.rs
use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::Rotator;

// ── compile-time validation ──────────────────────────────────────────────────

/// Verify at compile time that BITS is one of {4, 6, 8}.
const fn check_bits(bits: usize) {
    assert!(bits == 4 || bits == 6 || bits == 8,
        "QamMapper: BITS must be 4 (QAM-16), 6 (QAM-64), or 8 (QAM-256)");
}

// ── axis amplitude table ─────────────────────────────────────────────────────

/// Normalization scale for a square QAM constellation with `bits` bits/symbol.
/// Average symbol energy = 2*(M²-1)/3 where M = 2^(bits/2).
fn axis_scale(bits: usize) -> f32 {
    let m = 1usize << (bits / 2);
    let avg_e_total = 2.0 * ((m * m - 1) as f64) / 3.0;
    (1.0 / avg_e_total.sqrt()) as f32
}

/// Build a Gray-coded amplitude table for one QAM axis.
///
/// For M = 2^(BITS/2) levels per axis the unnormalized levels are the odd
/// integers {-(M-1), -(M-3), …, -1, +1, +3, …, +(M-1)}.
///
/// Natural index g maps to level (2g+1-M).  We remap by the standard binary
/// Gray code (g → g ^ (g>>1)) so adjacent amplitude steps differ by one bit.
///
/// The table has 16 entries (enough for M up to 16, i.e. QAM-256).
/// Only the first M entries are meaningful; the rest are 0.
const fn build_axis_table(bits: usize, scale: f32) -> [f32; 16] {
    let k = bits / 2;
    let m = 1usize << k;
    let mut table = [0.0f32; 16];
    let mut g = 0usize;
    while g < m {
        let gray = g ^ (g >> 1);
        let level = (2 * g + 1) as f32 - m as f32;
        table[gray] = level * scale;
        g += 1;
    }
    table
}

// ── QamMapper ────────────────────────────────────────────────────────────────

/// Square QAM constellation mapper: `BITS` u8 bits → one C32 symbol.
///
/// - `BITS = 4` → QAM-16  (2 bits/axis, 4 levels/axis)
/// - `BITS = 6` → QAM-64  (3 bits/axis, 8 levels/axis)
/// - `BITS = 8` → QAM-256 (4 bits/axis, 16 levels/axis)
///
/// Gray-coded on each axis independently, normalized to unit average energy.
///
/// Input is consumed `BITS` bytes at a time; each byte contributes its LSB.
/// The first `BITS/2` bytes encode the I axis (MSB-first within the axis),
/// the next `BITS/2` bytes encode the Q axis.
///
/// `in_read = n_syms * BITS`, `out_written = n_syms`.
#[derive(Debug, Clone, Copy)]
pub struct QamMapper<const BITS: usize> {
    table: [f32; 16],
}

impl<const BITS: usize> QamMapper<BITS> {
    pub fn new() -> Self {
        const { check_bits(BITS); }
        Self { table: build_axis_table(BITS, axis_scale(BITS)) }
    }

    /// Encode one axis: consume `K = BITS/2` bits from `input[base..]`,
    /// pack their LSBs into a Gray index, return the normalized amplitude.
    #[inline(always)]
    fn axis_amp(&self, input: &[u8], base: usize) -> f32 {
        let k = BITS / 2;
        let mut idx = 0usize;
        let mut b = 0;
        while b < k {
            idx = (idx << 1) | ((input[base + b] & 1) as usize);
            b += 1;
        }
        self.table[idx]
    }
}

impl<const BITS: usize> Default for QamMapper<BITS> {
    fn default() -> Self { Self::new() }
}

impl<const BITS: usize> Block for QamMapper<BITS> {
    type In  = u8;
    type Out = C32;

    #[inline(always)]
    fn process(&mut self, input: &[u8], output: &mut [C32]) -> WorkReport {
        let n_syms = (input.len() / BITS).min(output.len());
        let mut i = 0;
        let nn = n_syms & !3;
        while i < nn {
            let k = BITS / 2;
            output[i]   = C32::new(self.axis_amp(input, i*BITS),       self.axis_amp(input, i*BITS+k));
            output[i+1] = C32::new(self.axis_amp(input, (i+1)*BITS),   self.axis_amp(input, (i+1)*BITS+k));
            output[i+2] = C32::new(self.axis_amp(input, (i+2)*BITS),   self.axis_amp(input, (i+2)*BITS+k));
            output[i+3] = C32::new(self.axis_amp(input, (i+3)*BITS),   self.axis_amp(input, (i+3)*BITS+k));
            i += 4;
        }
        while i < n_syms {
            let k = BITS / 2;
            output[i] = C32::new(self.axis_amp(input, i*BITS), self.axis_amp(input, i*BITS+k));
            i += 1;
        }
        WorkReport { in_read: n_syms * BITS, out_written: n_syms }
    }
}

// ── type aliases ─────────────────────────────────────────────────────────────

pub type Qam16Mapper  = QamMapper<4>;
pub type Qam64Mapper  = QamMapper<6>;
pub type Qam256Mapper = QamMapper<8>;

// ── QamMod ───────────────────────────────────────────────────────────────────

/// QAM waveform stage: C32 symbols → C32 IQ.
/// rf_hz = 0.0 → baseband passthrough with gain.
/// rf_hz != 0.0 → rotate symbols onto carrier via Rotator.
///
/// Order-independent: works for any square QAM constellation.
#[derive(Debug, Clone)]
pub struct QamMod {
    gain: f32,
    rot:  Rotator,
}

impl QamMod {
    pub fn new(fs: f32, rf_hz: f32, gain: f32) -> Self {
        Self { gain, rot: Rotator::new(rf_hz, fs) }
    }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
}

impl Block for QamMod {
    type In  = C32;
    type Out = C32;

    #[inline(always)]
    fn process(&mut self, input: &[C32], output: &mut [C32]) -> WorkReport {
        let n = input.len().min(output.len());
        let g = self.gain;
        let mut i = 0;
        let nn = n & !3;
        while i < nn {
            let s0 = input[i];   let r0 = self.rot.next();
            let s1 = input[i+1]; let r1 = self.rot.next();
            let s2 = input[i+2]; let r2 = self.rot.next();
            let s3 = input[i+3]; let r3 = self.rot.next();
            output[i]   = C32::new(g * s0.re.mul_add(r0.re, -s0.im * r0.im), g * s0.im.mul_add(r0.re, s0.re * r0.im));
            output[i+1] = C32::new(g * s1.re.mul_add(r1.re, -s1.im * r1.im), g * s1.im.mul_add(r1.re, s1.re * r1.im));
            output[i+2] = C32::new(g * s2.re.mul_add(r2.re, -s2.im * r2.im), g * s2.im.mul_add(r2.re, s2.re * r2.im));
            output[i+3] = C32::new(g * s3.re.mul_add(r3.re, -s3.im * r3.im), g * s3.im.mul_add(r3.re, s3.re * r3.im));
            i += 4;
        }
        while i < n {
            let s = input[i]; let r = self.rot.next();
            output[i] = C32::new(g * s.re.mul_add(r.re, -s.im * r.im), g * s.im.mul_add(r.re, s.re * r.im));
            i += 1;
        }
        WorkReport { in_read: n, out_written: n }
    }
}
