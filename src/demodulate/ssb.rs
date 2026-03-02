use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::{LpDcCascade, Rotator};

#[derive(Debug, Clone)]
pub struct SsbProductDemod {
    filt: LpDcCascade,
    rot: Rotator,
}

impl SsbProductDemod {
    pub fn new(fs: f32, bfo_hz: f32, audio_bw_hz: f32) -> Self {
        Self {
            filt: LpDcCascade::design(fs, audio_bw_hz * 0.9, 2.0),
            rot: Rotator::new(bfo_hz, fs),
        }
    }
}

impl Block for SsbProductDemod {
    type In = C32;
    type Out = f32;

    #[inline(always)]
    fn process(&mut self, input: &[C32], output: &mut [f32]) -> WorkReport {
        let n = input.len().min(output.len());
        let mut i = 0;
        let nn = n & !3;

        while i < nn {
            // 0
            let p0 = self.rot.next();
            let z0 = input[i];
            let y0 = z0.re.mul_add(p0.re, z0.im * p0.im);
            output[i] = self.filt.process(y0);

            // 1
            let p1 = self.rot.next();
            let z1 = input[i+1];
            let y1 = z1.re.mul_add(p1.re, z1.im * p1.im);
            output[i+1] = self.filt.process(y1);

            // 2
            let p2 = self.rot.next();
            let z2 = input[i+2];
            let y2 = z2.re.mul_add(p2.re, z2.im * p2.im);
            output[i+2] = self.filt.process(y2);

            // 3
            let p3 = self.rot.next();
            let z3 = input[i+3];
            let y3 = z3.re.mul_add(p3.re, z3.im * p3.im);
            output[i+3] = self.filt.process(y3);

            i += 4;
        }
        while i < n {
            let p = self.rot.next();
            let z = input[i];
            let y = z.re.mul_add(p.re, z.im * p.im);
            output[i] = self.filt.process(y);
            i += 1;
        }
        WorkReport { in_read: n, out_written: n }
    }
}
