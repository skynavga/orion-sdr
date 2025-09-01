use crate::core::{Block, WorkReport};
use num_complex::Complex32 as C32;

/// Simple RMS-tracking AGC for real audio.
#[derive(Debug, Clone)]
pub struct AgcRms {
    fs: f32,
    attack_a: f32,
    release_a: f32,
    target_rms: f32,
    min_gain: f32,
    max_gain: f32,
    env: f32,
}

impl AgcRms {
    pub fn new(fs: f32, attack_ms: f32, release_ms: f32, target_rms: f32) -> Self {
        let a = |ms: f32| (-1.0 / (fs * (ms.max(1e-3) / 1_000.0))).exp();
        Self {
            fs,
            attack_a: a(attack_ms),
            release_a: a(release_ms),
            target_rms: target_rms.max(1e-6),
            min_gain: 0.05,
            max_gain: 20.0,
            env: 0.0,
        }
    }
    #[inline] fn update_env(&mut self, x2: f32) {
        let a = if x2 > self.env { self.attack_a } else { self.release_a };
        self.env = a * self.env + (1.0 - a) * x2;
    }
}

impl Block for AgcRms {
    type In = f32;
    type Out = f32;

    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        let n = input.len().min(output.len());
        if n == 0 {
            return WorkReport { in_read: 0, out_written: 0 };
        }

        // Seed env on the first call to avoid huge initial gain
        if self.env == 0.0 {
            let x0 = input[0];
            self.env = (x0 * x0).max(1e-12);
        }

        for i in 0..n {
            let x = input[i];
            self.update_env(x * x);
            let rms = self.env.sqrt().max(1e-6);
            let mut g = self.target_rms / rms;
            g = g.clamp(self.min_gain, self.max_gain);
            output[i] = g * x;
        }
        WorkReport { in_read: n, out_written: n }
    }
}

/// RMS-tracking AGC for complex IQ streams.
/// Tracks |x| RMS (sqrt(I^2 + Q^2)) with separate attack/release and
/// applies a scalar gain equally to I and Q.
#[derive(Debug, Clone)]
pub struct AgcRmsIq {
    fs: f32,
    attack_a: f32,
    release_a: f32,
    target_rms: f32,
    min_gain: f32,
    max_gain: f32,
    env: f32,
}

impl AgcRmsIq {
    /// `attack_ms`, `release_ms` in milliseconds; `target_rms` ≈ desired RMS amplitude.
    pub fn new(fs: f32, attack_ms: f32, release_ms: f32, target_rms: f32) -> Self {
        let a = |ms: f32| (-1.0 / (fs * (ms.max(1e-3) / 1_000.0))).exp();
        Self {
            fs,
            attack_a: a(attack_ms),
            release_a: a(release_ms),
            target_rms: target_rms.max(1e-6),
            min_gain: 0.05,
            max_gain: 20.0,
            env: 0.0,
        }
    }

    #[inline]
    fn update_env(&mut self, x2: f32) {
        let a = if x2 > self.env { self.attack_a } else { self.release_a };
        self.env = a * self.env + (1.0 - a) * x2;
    }
}

impl Block for AgcRmsIq {
    type In  = C32;
    type Out = C32;

    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        let n = input.len().min(output.len());
        if n == 0 {
            return WorkReport { in_read: 0, out_written: 0 };
        }

        // Seed env on first use to prevent initial clamp blast
        if self.env == 0.0 {
            let x0 = input[0];
            self.env = (x0.re * x0.re + x0.im * x0.im).max(1e-12);
        }

        for i in 0..n {
            let x = input[i];
            let x2 = x.re * x.re + x.im * x.im;
            self.update_env(x2);
            let rms = self.env.sqrt().max(1e-6);
            let mut g = self.target_rms / rms;
            g = g.clamp(self.min_gain, self.max_gain);
            output[i] = C32::new(g * x.re, g * x.im);
        }
        WorkReport { in_read: n, out_written: n }
    }
}
