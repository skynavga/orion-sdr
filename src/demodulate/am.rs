use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::{DcBlocker, FirLowpass};

/// AM envelope detector: |IQ| -> DC block -> audio LP.
#[derive(Debug, Clone)]
pub struct AmEnvelopeDemod {
    lp: FirLowpass,
    dc: DcBlocker,
    gain: f32,
    scratch: Vec<f32>,
}

impl AmEnvelopeDemod {

    /// `audio_bw_hz` e.g. 5–8 kHz for AM broadcast
    pub fn new(sample_rate: f32, audio_bw_hz: f32) -> Self {
        Self {
            lp: FirLowpass::design(sample_rate, audio_bw_hz, audio_bw_hz * 0.5),
            dc: DcBlocker::new(sample_rate, 2.0),
            gain: 1.0,
            scratch: Vec::new(),
        }
    }

    pub fn set_gain(&mut self, g: f32) { self.gain = g; }

}

impl Block for AmEnvelopeDemod {
    type In = C32;
    type Out = f32;

    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        let n = input.len().min(output.len());
        if self.scratch.len() < n { self.scratch.resize(n, 0.0); }

        // tmp: magnitude
        let tmp = &mut self.scratch[..n];
        for i in 0..n {
            let z = input[i];
            tmp[i] = (z.re * z.re + z.im * z.im).sqrt() * self.gain;
        }

        // LP -> output
        self.lp.process(&tmp[..], &mut output[..n]);

        // DC block (separate buffers)
        self.dc.process(&output[..n], &mut tmp[..]);
        output[..n].copy_from_slice(&tmp[..]);

        WorkReport { in_read: n, out_written: n }
    }

}
