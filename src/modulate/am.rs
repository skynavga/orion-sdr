use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::{Nco, mix_with_nco};

/// AM (DSB) modulator (baseband or RF-shift via NCO).
/// Output is complex IQ: s[n] = (carrier_level + m * x[n]) * e^{jφ[n]}
/// - `carrier_level` = 0.0 → DSB-SC; >0 adds a carrier (conventional AM)
/// - `modulation_index` m: recommended 0.0..1.0 to avoid overmodulation
#[derive(Debug, Clone)]
pub struct AmDsbMod {
    nco: Nco,
    modulation_index: f32,
    carrier_level: f32,
    clamp: bool,
    gain: f32,
}

impl AmDsbMod {
    /// `carrier_hz` may be 0.0 for complex baseband.
    pub fn new(sample_rate: f32, carrier_hz: f32, modulation_index: f32, carrier_level: f32) -> Self {
        Self {
            nco: Nco::new(carrier_hz, sample_rate),
            modulation_index,
            carrier_level,
            clamp: true,
            gain: 1.0,
        }
    }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
    pub fn set_modulation_index(&mut self, m: f32) { self.modulation_index = m; }
    pub fn set_carrier_level(&mut self, c: f32) { self.carrier_level = c; }
    pub fn set_limiter(&mut self, on: bool) { self.clamp = on; }
}

impl Block for AmDsbMod {
    type In  = f32;  // mono audio (−1..+1 recommended)
    type Out = C32;  // complex IQ

    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        let n = input.len().min(output.len());
        for i in 0..n {
            let m = self.carrier_level + self.modulation_index * input[i];
            let a = if self.clamp { m.clamp(-1.0, 1.0) } else { m };
            // Treat amplitude as real part, then frequency shift with NCO
            let base = C32::new(a * self.gain, 0.0);
            output[i] = mix_with_nco(base, &mut self.nco);
        }
        WorkReport { in_read: n, out_written: n }
    }
}
