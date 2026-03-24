#[cfg(feature = "simd")]
use core::simd::{Simd, SimdFloat};

use num_complex::Complex32 as C32;

#[derive(Debug, Clone)]
pub struct FirLowpass {
    taps: Vec<f32>,
    delay: Vec<f32>,
    idx: usize,
}

impl FirLowpass {

    /// Minimal LPF design (sinc + Hann).
    pub fn design(fs: f32, pass_hz: f32, trans_hz: f32) -> Self {
        let pass_hz = pass_hz.max(10.0);
        let trans_hz = trans_hz.max(pass_hz * 0.2);
        let ntaps = ((fs / trans_hz).ceil() as usize).max(31) | 1; // odd taps
        let mut taps = vec![0.0f32; ntaps];
        let fc = pass_hz / fs;
        let m0 = ntaps as isize / 2;
        for n in 0..ntaps {
            let m = n as isize - m0;
            let sinc = if m == 0 {
                2.0 * fc
            } else {
                let x = core::f32::consts::PI * m as f32;
                (2.0 * fc) * (2.0 * core::f32::consts::PI * fc * m as f32).sin() / x
            };
            let w = 0.5 - 0.5 * (2.0 * core::f32::consts::PI * n as f32 / (ntaps as f32 - 1.0)).cos();
            taps[n] = sinc * w;
        }
        let s: f32 = taps.iter().sum();
        for t in &mut taps { *t /= s; }
        Self { taps, delay: vec![0.0; ntaps], idx: 0 }
    }

    #[inline]
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let n = input.len().min(output.len());
        for i in 0..n {
            self.delay[self.idx] = input[i];
            output[i] = self.dot();
            self.idx = (self.idx + 1) % self.delay.len();
        }
    }

    #[inline(always)]
    fn dot(&self) -> f32 {
        let len = self.taps.len();
        let taps = &self.taps;
        let d = &self.delay;
        let mut acc = 0.0f32;

        #[cfg(feature = "simd")]
        {
            const LANES: usize = 8;
            type Vf = Simd<f32, LANES>;
            let mut i = 0;
            while i + LANES <= len {
                let mut td = [0.0f32; LANES];
                let mut tt = [0.0f32; LANES];
                for k in 0..LANES {
                    let t_idx = i + k;
                    let d_idx = (self.idx + len - 1 - t_idx) % len;
                    td[k] = d[d_idx];
                    tt[k] = taps[t_idx];
                }
                let vd = Vf::from_array(td);
                let vt = Vf::from_array(tt);
                acc += (vd * vt).reduce_sum();
                i += LANES;
            }
            for t_idx in i..len {
                let d_idx = (self.idx + len - 1 - t_idx) % len;
                acc += d[d_idx] * taps[t_idx];
            }
            return acc;
        }

        #[cfg(not(feature = "simd"))]
        {
            for t_idx in 0..len {
                let d_idx = (self.idx + len - 1 - t_idx) % len;
                acc += d[d_idx] * taps[t_idx];
            }
            return acc;
        }
    }

}

// ── Half-cosine matched filter ────────────────────────────────────────────────

/// Complex-input FIR matched filter for the PSK31 half-cosine pulse shape.
///
/// Taps are `hann[n] / sqrt(Σhann[n]²)` (unit-energy normalised), so the
/// filter's peak output equals the signal amplitude when the input is an
/// ideal noiseless pulse.
///
/// Design: feed every down-mixed complex sample into `push`; call `out()`
/// at the end of each symbol period (every `sps` samples) to get the
/// matched-filter symbol estimate.  The filter has `sps` taps so its group
/// delay is `(sps−1)/2` samples.  Since we call `out()` at sample `sps−1`
/// (the last sample of the period), the output corresponds to the aligned
/// peak of the cross-correlation — no additional latency management needed.
///
/// I and Q channels share the same real taps with a split delay line,
/// matching the pattern used by `FirDecimator`.
#[derive(Debug, Clone)]
pub struct HalfCosineMf {
    taps:     Vec<f32>,
    delay_re: Vec<f32>,
    delay_im: Vec<f32>,
    idx:      usize,
}

impl HalfCosineMf {
    /// Construct a half-cosine MF for a given number of samples per symbol.
    pub fn new(sps: usize) -> Self {
        // Half-cosine pulse: hann[n] = 0.5 − 0.5·cos(π·n / (sps−1))
        let hann: Vec<f32> = if sps <= 1 {
            vec![1.0f32; sps.max(1)]
        } else {
            let denom = (sps - 1) as f32;
            (0..sps)
                .map(|i| 0.5 - 0.5 * (core::f32::consts::PI * i as f32 / denom).cos())
                .collect()
        };
        // Normalise to unit energy.
        let energy: f32 = hann.iter().map(|&h| h * h).sum();
        let scale = if energy > 0.0 { energy.sqrt().recip() } else { 1.0 };
        let taps: Vec<f32> = hann.iter().map(|&h| h * scale).collect();
        let len = taps.len();
        Self {
            taps,
            delay_re: vec![0.0f32; len],
            delay_im: vec![0.0f32; len],
            idx: 0,
        }
    }

    /// Push one complex sample and return the current filter output.
    #[inline(always)]
    pub fn push(&mut self, s: C32) -> C32 {
        let len = self.taps.len();
        self.delay_re[self.idx] = s.re;
        self.delay_im[self.idx] = s.im;
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for t_idx in 0..len {
            let d_idx = (self.idx + len - t_idx) % len;
            let w = self.taps[t_idx];
            re += self.delay_re[d_idx] * w;
            im += self.delay_im[d_idx] * w;
        }
        self.idx = (self.idx + 1) % len;
        C32::new(re, im)
    }

    pub fn reset(&mut self) {
        self.delay_re.fill(0.0);
        self.delay_im.fill(0.0);
        self.idx = 0;
    }
}
