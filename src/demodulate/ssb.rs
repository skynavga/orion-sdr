use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::{DcBlocker, FirLowpass, Rotator};

#[derive(Debug, Clone)]
pub struct SsbProductDemod {
    rot: Rotator,
    post_lp: FirLowpass,
    dc: DcBlocker,
    // scratch
    tmp: Vec<f32>,
    ytmp: Vec<f32>,
}

impl SsbProductDemod {
    pub fn new(fs: f32, bfo_hz: f32, audio_bw_hz: f32) -> Self {
        let trans_hz = (audio_bw_hz * 0.15).clamp(200.0, 600.0);
        let post_lp = FirLowpass::design(fs, audio_bw_hz, trans_hz);
        let rot = Rotator::new(bfo_hz, fs);
        let dc = DcBlocker::new(fs, 2.0);
        Self { rot, post_lp, dc, tmp: Vec::new(), ytmp: Vec::new() }
    }
}

impl Block for SsbProductDemod {
    type In = C32;
    type Out = f32;

    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        let n = input.len().min(output.len());
        if self.tmp.len() < n { self.tmp.resize(n, 0.0); }
        if self.ytmp.len() < n { self.ytmp.resize(n, 0.0); }

        // --- Hot loop: product detect (USB): y = I*cos + Q*sin ---
        // Manual unroll by 4 to reduce loop/branch overhead and allow better FMA scheduling.
        let mut i = 0;
        let nn = n & !3; // round down to multiple of 4
        while i < nn {
            // 0
            let p0 = self.rot.next();
            let z0 = input[i];
            self.tmp[i] = z0.re.mul_add(p0.re, z0.im * p0.im);

            // 1
            let p1 = self.rot.next();
            let z1 = input[i + 1];
            self.tmp[i + 1] = z1.re.mul_add(p1.re, z1.im * p1.im);

            // 2
            let p2 = self.rot.next();
            let z2 = input[i + 2];
            self.tmp[i + 2] = z2.re.mul_add(p2.re, z2.im * p2.im);

            // 3
            let p3 = self.rot.next();
            let z3 = input[i + 3];
            self.tmp[i + 3] = z3.re.mul_add(p3.re, z3.im * p3.im);

            i += 4;
        }
        // tail
        while i < n {
            let p = self.rot.next();
            let z = input[i];
            self.tmp[i] = z.re.mul_add(p.re, z.im * p.im);
            i += 1;
        }

        // --- Audio LP -> DC blocker ---
        self.post_lp.process(&self.tmp[..n], &mut self.ytmp[..n]);
        self.dc.process(&self.ytmp[..n], &mut output[..n]);

        WorkReport { in_read: n, out_written: n }
    }
}
