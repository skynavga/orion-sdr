// src/demodulate/ft8.rs
use num_complex::Complex32 as C32;
use crate::modulate::ft8::{
    Ft8Frame, FT8_DATA_SYMS, FT8_SAMPLES_PER_SYM, FT8_TONE_SPACING_HZ,
    FT8_TONES, FT8_TOTAL_SYMS, FT8_FRAME_LEN,
};

// Costas sync positions (imported via constants from modulate side)
const FT8_SYNC_POS: [(usize, usize); 3] = [(0, 7), (36, 43), (72, 79)];

/// FT8 demodulator: frame-at-a-time Goertzel/dot-product tone detector.
///
/// For each of the 79 symbol slots, computes energy at each of the 8 tone
/// frequencies using a dot-product correlator and picks the strongest tone.
/// Sync positions are stripped; the 58 data tones are returned as `Ft8Frame`.
#[derive(Debug, Clone)]
pub struct Ft8Demod {
    fs: f32,
    base_hz: f32,
}

impl Ft8Demod {
    pub fn new(fs: f32, base_hz: f32) -> Self {
        Self { fs, base_hz }
    }

    /// Demodulate a 151 680-sample IQ block → `Ft8Frame` (58 data tone indices).
    ///
    /// Returns `None` if the input slice is shorter than `FT8_FRAME_LEN`.
    pub fn demodulate(&self, iq: &[C32]) -> Option<Ft8Frame> {
        if iq.len() < FT8_FRAME_LEN {
            return None;
        }

        // Pre-compute per-sample step phasors for each tone
        let steps: [C32; FT8_TONES] = core::array::from_fn(|k| {
            let f = self.base_hz + (k as f32) * FT8_TONE_SPACING_HZ;
            let phi = -core::f32::consts::TAU * f / self.fs;
            let (s, c) = phi.sin_cos();
            C32::new(c, s)
        });

        let mut all_tones = [0u8; FT8_TOTAL_SYMS];
        for sym in 0..FT8_TOTAL_SYMS {
            let slice = &iq[sym * FT8_SAMPLES_PER_SYM..(sym + 1) * FT8_SAMPLES_PER_SYM];
            all_tones[sym] = detect_tone(slice, &steps);
        }

        // Mark sync positions
        let mut is_sync = [false; FT8_TOTAL_SYMS];
        for &(start, end) in &FT8_SYNC_POS {
            for pos in start..end {
                is_sync[pos] = true;
            }
        }

        // Extract data tones
        let mut data = [0u8; FT8_DATA_SYMS];
        let mut idx = 0;
        for pos in 0..FT8_TOTAL_SYMS {
            if !is_sync[pos] {
                data[idx] = all_tones[pos];
                idx += 1;
            }
        }

        Some(Ft8Frame::new(data))
    }
}

/// Compute energy at each tone via dot-product correlator and return argmax.
#[inline]
fn detect_tone(slice: &[C32], steps: &[C32; FT8_TONES]) -> u8 {
    let n = slice.len();
    let mut best_e = -1.0f32;
    let mut best_k = 0u8;

    for (k, &w) in steps.iter().enumerate() {
        // Correlator: acc = Σ iq[i] · w^i  where w = e^{-j2πf/fs}
        let mut acc = C32::new(0.0, 0.0);
        let mut phasor = C32::new(1.0, 0.0);

        let mut i = 0;
        let nn = n & !3;
        while i < nn {
            acc += slice[i]   * phasor;
            let p1 = C32::new(
                phasor.re.mul_add(w.re, -phasor.im * w.im),
                phasor.im.mul_add(w.re,  phasor.re * w.im),
            );
            acc += slice[i+1] * p1;
            let p2 = C32::new(
                p1.re.mul_add(w.re, -p1.im * w.im),
                p1.im.mul_add(w.re,  p1.re * w.im),
            );
            acc += slice[i+2] * p2;
            let p3 = C32::new(
                p2.re.mul_add(w.re, -p2.im * w.im),
                p2.im.mul_add(w.re,  p2.re * w.im),
            );
            acc += slice[i+3] * p3;
            // advance phasor by 4 steps
            phasor = C32::new(
                p3.re.mul_add(w.re, -p3.im * w.im),
                p3.im.mul_add(w.re,  p3.re * w.im),
            );
            i += 4;
        }
        while i < n {
            acc += slice[i] * phasor;
            phasor = C32::new(
                phasor.re.mul_add(w.re, -phasor.im * w.im),
                phasor.im.mul_add(w.re,  phasor.re * w.im),
            );
            i += 1;
        }

        let e = acc.norm_sqr();
        if e > best_e {
            best_e = e;
            best_k = k as u8;
        }
    }
    best_k
}
