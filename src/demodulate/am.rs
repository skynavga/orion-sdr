// src/demodulate/am.rs
use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::{LpCascade, DcBlocker};

#[derive(Debug, Clone, Copy)]
pub enum Envelope {
    /// env = sqrt(I^2 + Q^2) after IIR LP (highest fidelity)
    PowerSqrt,
    /// env ≈ a*|I| + b*|Q| (very fast, small amplitude error)
    AbsApprox { k1: f32, k2: f32 },
}

#[derive(Debug, Clone)]
pub struct AmEnvelopeDemod {
    lp: LpCascade,
    dc: DcBlocker,
    method: Envelope,
}

impl AmEnvelopeDemod {
    pub fn new(fs: f32, audio_bw_hz: f32) -> Self {
        // 4th-order LR at ~0.9*BW gives clean audio passband.
        let lp = LpCascade::design(fs, audio_bw_hz * 0.9);
        let dc = DcBlocker::new(fs, 2.0);
        Self { lp, dc, method: Envelope::PowerSqrt }
    }

    /// Optional: switch to abs-approx envelope (faster, small error).
    pub fn with_abs_approx(mut self, k1: f32, k2: f32) -> Self {
        self.method = Envelope::AbsApprox { k1, k2 };
        self
    }
}

impl Block for AmEnvelopeDemod {
    type In  = C32; // complex baseband AM
    type Out = f32; // audio

    #[inline(always)]
    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        let n = input.len().min(output.len());
        let mut i = 0;
        let nn = n & !3;

        match self.method {
            Envelope::PowerSqrt => {
                while i < nn {
                    // 0
                    let z0 = input[i];
                    let p0 = z0.re.mul_add(z0.re, z0.im * z0.im);
                    let y0 = self.lp.process(p0).sqrt();
                    output[i] = self.dc.process_sample(y0);

                    // 1
                    let z1 = input[i+1];
                    let p1 = z1.re.mul_add(z1.re, z1.im * z1.im);
                    let y1 = self.lp.process(p1).sqrt();
                    output[i+1] = self.dc.process_sample(y1);

                    // 2
                    let z2 = input[i+2];
                    let p2 = z2.re.mul_add(z2.re, z2.im * z2.im);
                    let y2 = self.lp.process(p2).sqrt();
                    output[i+2] = self.dc.process_sample(y2);

                    // 3
                    let z3 = input[i+3];
                    let p3 = z3.re.mul_add(z3.re, z3.im * z3.im);
                    let y3 = self.lp.process(p3).sqrt();
                    output[i+3] = self.dc.process_sample(y3);

                    i += 4;
                }
                while i < n {
                    let z = input[i];
                    let p = z.re.mul_add(z.re, z.im * z.im);
                    let y = self.lp.process(p).sqrt();
                    output[i] = self.dc.process_sample(y);
                    i += 1;
                }
            }
            Envelope::AbsApprox { k1, k2 } => {
                while i < nn {
                    // 0
                    let z0 = input[i];
                    let a0 = z0.re.abs();
                    let b0 = z0.im.abs();
                    // e0 = k1*a0 + k2*b0, fused as e0 = k1.mul_add(a0, k2 * b0)
                    let e0 = k1.mul_add(a0, k2 * b0);
                    let y0 = self.lp.process(e0);
                    output[i] = self.dc.process_sample(y0);

                    // 1
                    let z1 = input[i+1];
                    let a1 = z1.re.abs();
                    let b1 = z1.im.abs();
                    let e1 = k1.mul_add(a1, k2 * b1);
                    let y1 = self.lp.process(e1);
                    output[i+1] = self.dc.process_sample(y1);

                    // 2
                    let z2 = input[i+2];
                    let a2 = z2.re.abs();
                    let b2 = z2.im.abs();
                    let e2 = k1.mul_add(a2, k2 * b2);
                    let y2 = self.lp.process(e2);
                    output[i+2] = self.dc.process_sample(y2);

                    // 3
                    let z3 = input[i+3];
                    let a3 = z3.re.abs();
                    let b3 = z3.im.abs();
                    let e3 = k1.mul_add(a3, k2 * b3);
                    let y3 = self.lp.process(e3);
                    output[i+3] = self.dc.process_sample(y3);

                    i += 4;
                }
                while i < n {
                    let z = input[i];
                    let a = z.re.abs();
                    let b = z.im.abs();
                    let e = k1.mul_add(a, k2 * b);
                    let y = self.lp.process(e);
                    output[i] = self.dc.process_sample(y);
                    i += 1;
                }
            }
        }

        WorkReport { in_read: n, out_written: n }
    }
}
