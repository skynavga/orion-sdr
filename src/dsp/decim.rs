use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use super::FirLowpass;

/// FIR-based decimator for complex IQ.
#[derive(Debug, Clone)]
pub struct FirDecimator {
    fs: f32,
    m: usize,
    lp_i: FirLowpass,
    lp_q: FirLowpass,
    // scratch
    ri: Vec<f32>,
    rq: Vec<f32>,
    yi: Vec<f32>,
    yq: Vec<f32>,
}

impl FirDecimator {
    /// `cutoff_hz` & `trans_hz` are at the input rate `fs`.
    pub fn new(fs: f32, m: usize, cutoff_hz: f32, trans_hz: f32) -> Self {
        let lp_i = FirLowpass::design(fs, cutoff_hz, trans_hz);
        let lp_q = FirLowpass::design(fs, cutoff_hz, trans_hz);
        Self {
            fs, m: m.max(1),
            lp_i, lp_q,
            ri: Vec::new(), rq: Vec::new(),
            yi: Vec::new(), yq: Vec::new(),
        }
    }
}

impl Block for FirDecimator {
    type In = C32;
    type Out = C32;

    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        let n = input.len();
        if self.ri.len() < n { self.ri.resize(n, 0.0); self.rq.resize(n, 0.0); }
        if self.yi.len() < n { self.yi.resize(n, 0.0); self.yq.resize(n, 0.0); }

        // split I/Q
        for k in 0..n {
            self.ri[k] = input[k].re;
            self.rq[k] = input[k].im;
        }
        // filter at input rate
        self.lp_i.process_block(&self.ri[..n], &mut self.yi[..n]);
        self.lp_q.process_block(&self.rq[..n], &mut self.yq[..n]);

        // decimate by M
        let m = self.m;
        let n_out = (n + m - 1) / m;
        let n_write = n_out.min(output.len());
        for j in 0..n_write {
            let k = j * m;
            output[j] = C32::new(self.yi[k], self.yq[k]);
        }
        WorkReport { in_read: n, out_written: n_write }
    }
}
