// src/modulate/ft8.rs
use num_complex::Complex32 as C32;
use crate::dsp::Rotator;

// FT8 protocol constants
pub const FT8_TONE_SPACING_HZ: f32 = 6.25;
pub const FT8_BAUD: f32 = 6.25;
pub const FT8_SAMPLES_PER_SYM: usize = 1920; // 12000 / 6.25
pub const FT8_TOTAL_SYMS: usize = 79;
pub const FT8_DATA_SYMS: usize = 58;
pub const FT8_TONES: usize = 8;
pub const FT8_FRAME_LEN: usize = FT8_TOTAL_SYMS * FT8_SAMPLES_PER_SYM; // 151_680

// Costas synchronization arrays (3 × 7-symbol arrays)
const FT8_COSTAS: [u8; 7] = [3, 1, 4, 0, 6, 5, 2];
// Costas block positions in the 79-symbol sequence: [start, end)
const FT8_SYNC_POS: [(usize, usize); 3] = [(0, 7), (36, 43), (72, 79)];

/// FT8 data frame: 58 tone indices (0–7), no sync symbols.
/// The caller provides raw data tone indices; Costas sync is inserted by the modulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ft8Frame(pub [u8; FT8_DATA_SYMS]);

impl Ft8Frame {
    pub fn new(tones: [u8; FT8_DATA_SYMS]) -> Self { Self(tones) }
    pub fn zeros() -> Self { Self([0u8; FT8_DATA_SYMS]) }
}

/// FT8 modulator: frame-at-a-time rectangular 8-FSK.
///
/// Produces a 151 680-sample (79 × 1920) complex IQ waveform at 12 kHz.
/// Phase continuity is maintained across symbol boundaries (CPFSK).
#[derive(Debug, Clone)]
pub struct Ft8Mod {
    fs: f32,
    base_hz: f32,
    rf_hz: f32,
    gain: f32,
}

impl Ft8Mod {
    /// Create a new FT8 modulator.
    ///
    /// * `fs`      — sample rate (should be 12 000 Hz)
    /// * `base_hz` — frequency of tone 0 (lowest tone)
    /// * `rf_hz`   — IF upconversion frequency (0.0 = baseband)
    /// * `gain`    — output amplitude scale
    pub fn new(fs: f32, base_hz: f32, rf_hz: f32, gain: f32) -> Self {
        Self { fs, base_hz, rf_hz, gain }
    }

    /// Build the 79-symbol tone sequence from an Ft8Frame, inserting Costas sync.
    pub fn build_symbol_sequence(frame: &Ft8Frame) -> [u8; FT8_TOTAL_SYMS] {
        let mut syms = [0u8; FT8_TOTAL_SYMS];
        // Mark sync positions
        let mut is_sync = [false; FT8_TOTAL_SYMS];
        for &(start, end) in &FT8_SYNC_POS {
            for pos in start..end {
                is_sync[pos] = true;
            }
        }
        // Fill in Costas sync symbols
        for (blk, &(start, _)) in FT8_SYNC_POS.iter().enumerate() {
            for i in 0..7 {
                syms[start + i] = FT8_COSTAS[i];
                let _ = blk; // suppress warning
            }
        }
        // Fill in data symbols
        let mut data_idx = 0usize;
        for pos in 0..FT8_TOTAL_SYMS {
            if !is_sync[pos] {
                syms[pos] = frame.0[data_idx];
                data_idx += 1;
            }
        }
        syms
    }

    /// Modulate one frame → `Vec<C32>` of length `FT8_FRAME_LEN` (151 680).
    pub fn modulate(&self, frame: &Ft8Frame) -> Vec<C32> {
        let syms = Self::build_symbol_sequence(frame);
        let total_samples = FT8_TOTAL_SYMS * FT8_SAMPLES_PER_SYM;
        let mut out = vec![C32::new(0.0, 0.0); total_samples];

        // Phase-continuous FSK: maintain running phasor across symbol boundaries
        let mut z = C32::new(1.0, 0.0); // running phasor
        let mut renorm_ctr = 0u32;

        for (sym_idx, &tone) in syms.iter().enumerate() {
            let f_tone = self.base_hz + (tone as f32) * FT8_TONE_SPACING_HZ;
            let phi = core::f32::consts::TAU * f_tone / self.fs;
            let (s, c) = phi.sin_cos();
            let w = C32::new(c, s); // per-sample step

            let base = sym_idx * FT8_SAMPLES_PER_SYM;
            let g = self.gain;

            let mut i = 0;
            let nn = FT8_SAMPLES_PER_SYM & !3;
            while i < nn {
                // Advance phasor 4 steps, manual unroll
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
            while i < FT8_SAMPLES_PER_SYM {
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
