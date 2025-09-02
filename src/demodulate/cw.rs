use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
 
#[derive(Debug, Clone)]
pub struct CwEnvelopeDemod {
    alpha: f32,   // one-pole LP smoothing factor
    y: f32,       // LP state
    gain: f32,
}

impl CwEnvelopeDemod {
    pub fn new(sample_rate: f32, _tone_hz: f32, env_bw_hz: f32) -> Self {
        // One-pole LP: alpha = exp(-2π fc / fs). Larger fc → faster tracking.
        let fc = env_bw_hz.max(1.0);
        let alpha = (-std::f32::consts::TAU * fc / sample_rate).exp();
        Self { alpha, y: 0.0, gain: 1.0 }
    }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
}

impl Block for CwEnvelopeDemod {
    type In = C32;
    type Out = f32;

    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        let n = input.len().min(output.len());
        let a = self.alpha;
        for i in 0..n {
            let mag = input[i].re.hypot(input[i].im);
            self.y = a * self.y + (1.0 - a) * mag;
            output[i] = self.y * self.gain;
        }
        WorkReport { in_read: n, out_written: n }
    }
}
