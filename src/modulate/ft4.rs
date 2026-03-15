// src/modulate/ft4.rs
use num_complex::Complex32 as C32;
use crate::dsp::Rotator;

// FT4 protocol constants
pub const FT4_TONE_SPACING_HZ: f32 = 20.833_334; // 500/24 Hz (exact: 20000/960)
pub const FT4_BAUD: f32 = 20.833_334;
pub const FT4_SAMPLES_PER_SYM: usize = 576; // 12000 / 20.833... = 576
// Frame structure: R S4_1 D29 S4_2 D29 S4_3 D29 S4_4 R  (105 symbols total)
// R = ramp symbol (tone 0) at positions 0 and 104.
// 4 Costas blocks of 4 symbols + 2 ramps = 18 non-data symbols; 105 - 18 = 87 data.
pub const FT4_TOTAL_SYMS: usize = 105;
pub const FT4_DATA_SYMS: usize = 87;
pub const FT4_TONES: usize = 4;
pub const FT4_FRAME_LEN: usize = FT4_TOTAL_SYMS * FT4_SAMPLES_PER_SYM; // 60_480

// FT4 Costas arrays (4 × 4-symbol arrays). Source: ft8_lib kFT4_Costas_pattern.
const FT4_COSTAS: [[u8; 4]; 4] = [
    [0, 1, 3, 2],
    [1, 0, 2, 3],
    [2, 3, 1, 0],
    [3, 2, 0, 1],
];
// Costas block positions in the 105-symbol sequence: [start, end)
// Ramp symbols occupy positions 0 and 104 (not included here).
const FT4_SYNC_POS: [(usize, usize); 4] = [(1, 5), (34, 38), (67, 71), (100, 104)];

/// FT4 data frame: 87 tone indices (0–3), no sync symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ft4Frame(pub [u8; FT4_DATA_SYMS]);

impl Ft4Frame {
    pub fn new(tones: [u8; FT4_DATA_SYMS]) -> Self { Self(tones) }
    pub fn zeros() -> Self { Self([0u8; FT4_DATA_SYMS]) }
}

/// FT4 modulator: frame-at-a-time rectangular 4-FSK.
///
/// Produces a 60 480-sample (105 × 576) complex IQ waveform at 12 kHz.
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

    /// Build the 105-symbol tone sequence from an Ft4Frame, inserting ramps and Costas sync.
    ///
    /// Layout: R S4_1 D29 S4_2 D29 S4_3 D29 S4_4 R
    /// - Ramp symbols (tone 0) at positions 0 and 104.
    /// - Costas blocks at positions 1-4, 34-37, 67-70, 100-103.
    /// - 87 data symbols fill the remaining positions.
    pub fn build_symbol_sequence(frame: &Ft4Frame) -> [u8; FT4_TOTAL_SYMS] {
        let mut syms = [0u8; FT4_TOTAL_SYMS];

        // Ramp symbols: tone 0 at start and end (already zero from array init,
        // but set explicitly for clarity)
        syms[0] = 0;
        syms[104] = 0;

        // Mark non-data positions: ramps + Costas blocks
        let mut is_reserved = [false; FT4_TOTAL_SYMS];
        is_reserved[0] = true;
        is_reserved[104] = true;
        for &(start, end) in &FT4_SYNC_POS {
            for pos in start..end {
                is_reserved[pos] = true;
            }
        }

        // Fill in Costas sync symbols
        for (blk, &(start, _)) in FT4_SYNC_POS.iter().enumerate() {
            for i in 0..4 {
                syms[start + i] = FT4_COSTAS[blk][i];
            }
        }

        // Fill in data symbols in the remaining slots
        let mut data_idx = 0usize;
        for pos in 0..FT4_TOTAL_SYMS {
            if !is_reserved[pos] {
                syms[pos] = frame.0[data_idx];
                data_idx += 1;
            }
        }
        syms
    }

    /// Modulate one frame → `Vec<C32>` of length `FT4_FRAME_LEN` (60 480).
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
