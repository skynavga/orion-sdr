use num_complex::Complex32 as C32;

/// Complex oscillation / translation without per-sample sin/cos.
#[derive(Clone, Copy, Debug)]
pub struct Rotator {
    z: C32,
    w: C32,
}

impl Rotator {
    #[inline]
    pub fn new(freq_hz: f32, fs: f32) -> Self {
        let phi = core::f32::consts::TAU * freq_hz / fs;
        Self {
            z: C32::new(1.0, 0.0),
            w: C32::new(phi.cos(), phi.sin()),
        }
    }
    #[inline]
    pub fn reset_phase(&mut self) { self.z = C32::new(1.0, 0.0); }
    #[inline]
    pub fn set_freq(&mut self, freq_hz: f32, fs: f32) {
        let phi = core::f32::consts::TAU * freq_hz / fs;
        self.w = C32::new(phi.cos(), phi.sin());
    }
    /// Advance and return next phasor.
    #[inline]
    pub fn next(&mut self) -> C32 {
        self.z *= self.w;
        self.z
    }
}
