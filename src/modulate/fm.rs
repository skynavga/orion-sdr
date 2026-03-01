use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::{Nco, mix_with_nco};

/// FM (direct) – phase accumulator with deviation scaling (Hz per unit input).
/// Uses a phasor-recurrence oscillator: each sample multiplies the running phasor
/// by e^{jΔφ} where Δφ = 2π·kf·x/fs, eliminating rem_euclid and large-angle trig.
#[derive(Debug, Clone)]
pub struct FmPhaseAccumMod {
    fs: f32,
    kf_hz_per_unit: f32, // peak deviation per |x|=1
    z: C32,              // running phasor (cos φ + j sin φ)
    rf_nco: Nco,         // optional RF translation (0 Hz for baseband)
    gain: f32,           // overall output gain (post-carrier)
    renorm_ctr: u32,
}

impl FmPhaseAccumMod {
    pub fn new(sample_rate: f32, deviation_hz: f32, rf_hz: f32) -> Self {
        Self {
            fs: sample_rate,
            kf_hz_per_unit: deviation_hz,
            z: C32::new(1.0, 0.0),
            rf_nco: Nco::new(rf_hz, sample_rate),
            gain: 1.0,
            renorm_ctr: 0,
        }
    }
    pub fn set_deviation(&mut self, deviation_hz: f32) { self.kf_hz_per_unit = deviation_hz; }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
}

impl Block for FmPhaseAccumMod {
    type In = f32;
    type Out = C32;

    #[inline(always)]
    fn process(&mut self, input: &[f32], output: &mut [C32]) -> WorkReport {
        let n = input.len().min(output.len());
        let kf = std::f32::consts::TAU * self.kf_hz_per_unit / self.fs;
        for i in 0..n {
            // e^{jΔφ}: Δφ is small (bounded by 2π·dev_hz/fs), so sin_cos is fast
            let dphi = kf * input[i];
            let (ds, dc) = dphi.sin_cos();
            // z *= e^{jΔφ}
            let zr = self.z.re.mul_add(dc, -self.z.im * ds);
            let zi = self.z.im.mul_add(dc,  self.z.re * ds);
            self.z = C32::new(zr, zi);

            // Periodic renormalization every 1024 samples
            self.renorm_ctr = self.renorm_ctr.wrapping_add(1);
            if (self.renorm_ctr & 0x3FF) == 0 {
                let inv = (self.z.re * self.z.re + self.z.im * self.z.im).sqrt().recip();
                self.z.re *= inv;
                self.z.im *= inv;
            }

            let base = self.z * self.gain;
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
