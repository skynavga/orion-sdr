use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::{DcBlocker, FirLowpass, Rotator};

#[derive(Debug, Clone)]
pub struct SsbProductDemod {
    rot: Rotator,
    post_lp: FirLowpass,
    dc: DcBlocker,
}

impl SsbProductDemod {
    pub fn new(fs: f32, bfo_hz: f32, audio_bw_hz: f32) -> Self {
        let trans_hz = (audio_bw_hz * 0.15).clamp(200.0, 600.0);
        let post_lp = FirLowpass::design(fs, audio_bw_hz, trans_hz);
        let rot = Rotator::new(bfo_hz, fs);
        let dc = DcBlocker::new(fs, 2.0);
        Self { rot, post_lp, dc }
    }
}

impl Block for SsbProductDemod {
    type In = C32;
    type Out = f32;

    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        let n = input.len().min(output.len());
        // Multiply by BFO phasor and take real (USB convention; sign flips choose sideband)
        let mut mix = vec![0.0f32; n];
        for i in 0..n {
            let w = self.rot.next();
            mix[i] = (input[i] * w).re;
        }
        let mut ytmp = vec![0.0f32; n];
        self.post_lp.process_block(&mix, &mut ytmp);
        self.dc.process(&ytmp, &mut output[..n]);
        WorkReport { in_read: n, out_written: n }
    }
}
