// src/dsp/nco.rs
use core::f32::consts::TAU;

/// Lightweight NCO used for RF/IF mixing in modulators.
#[derive(Debug, Clone)]
pub struct Nco {
    fs: f32,
    freq_hz: f32,
    phase: f32,
    dphi: f32,
}

impl Nco {
    pub fn new(freq_hz: f32, fs: f32) -> Self {
        let dphi = TAU * freq_hz / fs;
        Self { fs, freq_hz, phase: 0.0, dphi }
    }

    #[inline]
    pub fn set_freq(&mut self, freq_hz: f32) {
        self.freq_hz = freq_hz;
        self.dphi = TAU * freq_hz / self.fs;
    }

    /// Return cos/sin for the current phase and advance by one sample.
    #[inline]
    pub fn next_cs(&mut self) -> (f32, f32) {
        let c = self.phase.cos();
        let s = self.phase.sin();
        self.phase = (self.phase + self.dphi) % TAU;
        (c, s)
    }
}
