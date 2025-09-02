use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::{Nco, mix_with_nco};

/// FM (direct) – phase accumulator with deviation scaling (Hz per unit input).
pub struct FmPhaseAccumMod {
    fs: f32,
    kf_hz_per_unit: f32, // peak deviation per |x|=1
    phase: f32,          // radians
    rf_nco: Nco,         // optional RF translation (0 Hz for baseband)
    gain: f32,           // overall output gain (post-carrier)
}

impl FmPhaseAccumMod {
    pub fn new(sample_rate: f32, deviation_hz: f32, rf_hz: f32) -> Self {
        Self {
            fs: sample_rate,
            kf_hz_per_unit: deviation_hz,
            phase: 0.0,
            rf_nco: Nco::new(rf_hz, sample_rate),
            gain: 1.0,
        }
    }
    pub fn set_deviation(&mut self, deviation_hz: f32) { self.kf_hz_per_unit = deviation_hz; }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
}

impl Block for FmPhaseAccumMod {
    type In = f32;
    type Out = C32;

    fn process(&mut self, input: &[f32], output: &mut [C32]) -> WorkReport {
        let n = input.len().min(output.len());
        let two_pi = std::f32::consts::TAU;
        for i in 0..n {
            // Δφ = 2π * kf * x / fs
            self.phase = (self.phase + two_pi * self.kf_hz_per_unit * input[i] / self.fs).rem_euclid(two_pi);
            let base = C32::new(self.phase.cos(), self.phase.sin()) * self.gain;
            output[i] = mix_with_nco(base, &mut self.rf_nco);
        }
        WorkReport { in_read: n, out_written: n }
    }
}

/// ---------------------------------------------------------------------------
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
