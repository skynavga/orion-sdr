// src/demodulate/ft4.rs
use num_complex::Complex32 as C32;
use crate::modulate::ft4::{
    Ft4Frame, FT4_DATA_SYMS, FT4_SAMPLES_PER_SYM, FT4_TONE_SPACING_HZ,
    FT4_TONES, FT4_TOTAL_SYMS, FT4_FRAME_LEN,
};

// Costas sync positions
const FT4_SYNC_POS: [(usize, usize); 4] = [(0, 4), (29, 33), (60, 64), (99, 103)];

/// FT4 demodulator: frame-at-a-time dot-product tone detector.
///
/// For each of the 103 symbol slots, computes energy at each of the 4 tone
/// frequencies and picks the strongest. Sync positions are stripped; the 87
/// data tones are returned as `Ft4Frame`.
#[derive(Debug, Clone)]
pub struct Ft4Demod {
    fs: f32,
    base_hz: f32,
}

impl Ft4Demod {
    pub fn new(fs: f32, base_hz: f32) -> Self {
        Self { fs, base_hz }
    }

    /// Demodulate a 59 328-sample IQ block → `Ft4Frame` (87 data tone indices).
    ///
    /// Returns `None` if the input slice is shorter than `FT4_FRAME_LEN`.
    pub fn demodulate(&self, iq: &[C32]) -> Option<Ft4Frame> {
        if iq.len() < FT4_FRAME_LEN {
            return None;
        }

        // Pre-compute per-sample step phasors for each tone
        let steps: [C32; FT4_TONES] = core::array::from_fn(|k| {
            let f = self.base_hz + (k as f32) * FT4_TONE_SPACING_HZ;
            let phi = -core::f32::consts::TAU * f / self.fs;
            let (s, c) = phi.sin_cos();
            C32::new(c, s)
        });

        let mut all_tones = [0u8; FT4_TOTAL_SYMS];
        for sym in 0..FT4_TOTAL_SYMS {
            let slice = &iq[sym * FT4_SAMPLES_PER_SYM..(sym + 1) * FT4_SAMPLES_PER_SYM];
            all_tones[sym] = detect_tone(slice, &steps);
        }

        // Mark sync positions
        let mut is_sync = [false; FT4_TOTAL_SYMS];
        for &(start, end) in &FT4_SYNC_POS {
            for pos in start..end {
                is_sync[pos] = true;
            }
        }

        // Extract data tones
        let mut data = [0u8; FT4_DATA_SYMS];
        let mut idx = 0;
        for pos in 0..FT4_TOTAL_SYMS {
            if !is_sync[pos] {
                data[idx] = all_tones[pos];
                idx += 1;
            }
        }

        Some(Ft4Frame::new(data))
    }
}

/// Compute energy at each tone via dot-product correlator and return argmax.
#[inline]
fn detect_tone(slice: &[C32], steps: &[C32; FT4_TONES]) -> u8 {
    let n = slice.len();
    let mut best_e = -1.0f32;
    let mut best_k = 0u8;

    for (k, &w) in steps.iter().enumerate() {
        let mut acc = C32::new(0.0, 0.0);
        let mut phasor = C32::new(1.0, 0.0);

        let mut i = 0;
        let nn = n & !3;
        while i < nn {
            acc += slice[i] * phasor;
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
