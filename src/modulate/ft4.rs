// src/modulate/ft4.rs
use num_complex::Complex32 as C32;
use crate::dsp::Rotator;

// FT4 protocol constants
pub const FT4_TONE_SPACING_HZ: f32 = 20.833_334; // 500/24 Hz (exact: 20000/960)
pub const FT4_BAUD: f32 = 20.833_334;
pub const FT4_SAMPLES_PER_SYM: usize = 576; // 12000 / 20.833... = 576
pub const FT4_TOTAL_SYMS: usize = 103;
pub const FT4_DATA_SYMS: usize = 87;
pub const FT4_TONES: usize = 4;
pub const FT4_FRAME_LEN: usize = FT4_TOTAL_SYMS * FT4_SAMPLES_PER_SYM; // 59_328

// FT4 Costas arrays (4 × 4-symbol arrays)
const FT4_COSTAS: [[u8; 4]; 4] = [
    [0, 1, 3, 2],
    [1, 0, 2, 3],
    [2, 3, 0, 1],
    [3, 2, 1, 0],
];
// Costas block positions in the 103-symbol sequence: [start, end)
const FT4_SYNC_POS: [(usize, usize); 4] = [(0, 4), (29, 33), (60, 64), (99, 103)];

/// FT4 data frame: 87 tone indices (0–3), no sync symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ft4Frame(pub [u8; FT4_DATA_SYMS]);

impl Ft4Frame {
    pub fn new(tones: [u8; FT4_DATA_SYMS]) -> Self { Self(tones) }
    pub fn zeros() -> Self { Self([0u8; FT4_DATA_SYMS]) }
}

/// FT4 modulator: frame-at-a-time rectangular 4-FSK.
///
/// Produces a 59 328-sample (103 × 576) complex IQ waveform at 12 kHz.
/// Phase continuity is maintained across symbol boundaries (CPFSK).
#[derive(Debug, Clone)]
pub struct Ft4Mod {
    fs: f32,
    base_hz: f32,
    rf_hz: f32,
    gain: f32,
}

impl Ft4Mod {
    /// Create a new FT4 modulator.
    ///
    /// * `fs`      — sample rate (should be 12 000 Hz)
    /// * `base_hz` — frequency of tone 0 (lowest tone)
    /// * `rf_hz`   — IF upconversion frequency (0.0 = baseband)
    /// * `gain`    — output amplitude scale
    pub fn new(fs: f32, base_hz: f32, rf_hz: f32, gain: f32) -> Self {
        Self { fs, base_hz, rf_hz, gain }
    }

    /// Build the 103-symbol tone sequence from an Ft4Frame, inserting Costas sync.
    pub fn build_symbol_sequence(frame: &Ft4Frame) -> [u8; FT4_TOTAL_SYMS] {
        let mut syms = [0u8; FT4_TOTAL_SYMS];
        // Mark sync positions
        let mut is_sync = [false; FT4_TOTAL_SYMS];
        for &(start, end) in &FT4_SYNC_POS {
            for pos in start..end {
                is_sync[pos] = true;
            }
        }
        // Fill in Costas sync symbols
        for (blk, &(start, _)) in FT4_SYNC_POS.iter().enumerate() {
            for i in 0..4 {
                syms[start + i] = FT4_COSTAS[blk][i];
            }
        }
        // Fill in data symbols
        let mut data_idx = 0usize;
        for pos in 0..FT4_TOTAL_SYMS {
            if !is_sync[pos] {
                syms[pos] = frame.0[data_idx];
                data_idx += 1;
            }
        }
        syms
    }

    /// Modulate one frame → `Vec<C32>` of length `FT4_FRAME_LEN` (59 328).
    pub fn modulate(&self, frame: &Ft4Frame) -> Vec<C32> {
        let syms = Self::build_symbol_sequence(frame);
        let total_samples = FT4_TOTAL_SYMS * FT4_SAMPLES_PER_SYM;
        let mut out = vec![C32::new(0.0, 0.0); total_samples];

        // Phase-continuous FSK: maintain running phasor across symbol boundaries
        let mut z = C32::new(1.0, 0.0);
        let mut renorm_ctr = 0u32;

        for (sym_idx, &tone) in syms.iter().enumerate() {
            let f_tone = self.base_hz + (tone as f32) * FT4_TONE_SPACING_HZ;
            let phi = core::f32::consts::TAU * f_tone / self.fs;
            let (s, c) = phi.sin_cos();
            let w = C32::new(c, s);

            let base = sym_idx * FT4_SAMPLES_PER_SYM;
            let g = self.gain;

            let mut i = 0;
            let nn = FT4_SAMPLES_PER_SYM & !3;
            while i < nn {
                let zr0 = z.re.mul_add(w.re, -z.im * w.im);
                let zi0 = z.im.mul_add(w.re,  z.re * w.im);
                z = C32::new(zr0, zi0);
                out[base + i] = C32::new(g * z.re, g * z.im);

                let zr1 = z.re.mul_add(w.re, -z.im * w.im);
                let zi1 = z.im.mul_add(w.re,  z.re * w.im);
                z = C32::new(zr1, zi1);
                out[base + i + 1] = C32::new(g * z.re, g * z.im);

                let zr2 = z.re.mul_add(w.re, -z.im * w.im);
                let zi2 = z.im.mul_add(w.re,  z.re * w.im);
                z = C32::new(zr2, zi2);
                out[base + i + 2] = C32::new(g * z.re, g * z.im);

                let zr3 = z.re.mul_add(w.re, -z.im * w.im);
                let zi3 = z.im.mul_add(w.re,  z.re * w.im);
                z = C32::new(zr3, zi3);
                out[base + i + 3] = C32::new(g * z.re, g * z.im);

                i += 4;
                renorm_ctr = renorm_ctr.wrapping_add(4);
                if (renorm_ctr & 0x3FF) < 4 {
                    let r2 = z.re * z.re + z.im * z.im;
                    let inv = r2.sqrt().recip();
                    z.re *= inv;
                    z.im *= inv;
                }
            }
            while i < FT4_SAMPLES_PER_SYM {
                let zr = z.re.mul_add(w.re, -z.im * w.im);
                let zi = z.im.mul_add(w.re,  z.re * w.im);
                z = C32::new(zr, zi);
                out[base + i] = C32::new(g * z.re, g * z.im);
                i += 1;
                renorm_ctr = renorm_ctr.wrapping_add(1);
            }
        }

        // Optional RF upconversion
        if self.rf_hz != 0.0 {
            let mut rot = Rotator::new(self.rf_hz, self.fs);
            rot.rotate_block(&out.clone(), &mut out);
        }

        out
    }
}
