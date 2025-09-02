use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::{Nco, mix_with_nco};

/// PM (direct) – instantaneous phase φ = kp * x[n], optional RF upconversion.
pub struct PmDirectPhaseMod {
    kp_rad_per_unit: f32,
    rf_nco: Nco,
    gain: f32,
}

impl PmDirectPhaseMod {
    pub fn new(sample_rate: f32, kp_rad_per_unit: f32, rf_hz: f32) -> Self {
        Self {
            kp_rad_per_unit,
            rf_nco: Nco::new(rf_hz, sample_rate),
            gain: 1.0,
        }
    }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
    pub fn set_sensitivity(&mut self, kp_rad_per_unit: f32) { self.kp_rad_per_unit = kp_rad_per_unit; }
}

impl Block for PmDirectPhaseMod {
    type In = f32;
    type Out = C32;

    fn process(&mut self, input: &[f32], output: &mut [C32]) -> WorkReport {
        let n = input.len().min(output.len());
        for i in 0..n {
            let phi = self.kp_rad_per_unit * input[i];
            let base = C32::new(phi.cos(), phi.sin()) * self.gain;
            output[i] = mix_with_nco(base, &mut self.rf_nco);
        }
        WorkReport { in_read: n, out_written: n }
    }
}
