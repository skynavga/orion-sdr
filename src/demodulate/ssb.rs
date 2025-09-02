use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::{DcBlocker, LpCascade, Rotator};

#[derive(Debug, Clone)]
pub struct SsbProductDemod {
    post_lp: LpCascade,
    rot: Rotator,
    dc: DcBlocker,
}

impl SsbProductDemod {
    pub fn new(fs: f32, bfo_hz: f32, audio_bw_hz: f32) -> Self {
        let post_lp = LpCascade::design(fs, audio_bw_hz * 0.9);
        Self {
            post_lp,
            rot: Rotator::new(bfo_hz, fs),
            dc: DcBlocker::new(fs, 2.0), // ~2 Hz cutoff
        }
    }
}

impl Block for SsbProductDemod {
    type In = C32;
    type Out = f32;

    #[inline]
    fn process(&mut self, input: &[C32], output: &mut [f32]) -> WorkReport {
        let n = input.len().min(output.len());
        for i in 0..n {
            let p = self.rot.next(); // (cos,sin)
            let z = input[i];
            let y = z.re.mul_add(p.re, z.im * p.im); // product detect (USB)
            let ylp = self.post_lp.process(y);
            output[i] = self.dc.process_sample(ylp);
        }
        WorkReport { in_read: n, out_written: n }
    }
}
