#[cfg(feature = "simd")]
use core::simd::{Simd, SimdFloat};

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
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
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
