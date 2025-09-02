use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::{Nco, mix_with_nco};

/// CW (keyed carrier) – envelope-shaped keyed NCO
#[derive(Debug, Clone)]
pub struct CwKeyedMod {
    nco: Nco,
    env: f32,
    alpha_rise: f32,
    alpha_fall: f32,
    gain: f32,
}

impl CwKeyedMod {
    /// `tone_hz`: CW tone frequency (baseband or RF, depending on usage)
    /// `rise_ms`/`fall_ms`: envelope time constants (to avoid key clicks)
    pub fn new(sample_rate: f32, tone_hz: f32, rise_ms: f32, fall_ms: f32) -> Self {
        let tau_r = (rise_ms.max(0.1) * 1e-3) * sample_rate;
        let tau_f = (fall_ms.max(0.1) * 1e-3) * sample_rate;
        let alpha_r = (-1.0 / tau_r).exp();
        let alpha_f = (-1.0 / tau_f).exp();
        Self {
            nco: Nco::new(tone_hz, sample_rate),
            env: 0.0,
            alpha_rise: alpha_r,
            alpha_fall: alpha_f,
            gain: 1.0,
        }
    }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
}

impl Block for CwKeyedMod {
    type In = f32;   // keying envelope 0..1 (you can derive this from audio or key events)
    type Out = C32;  // IQ

    fn process(&mut self, input: &[f32], output: &mut [C32]) -> WorkReport {
        let n = input.len().min(output.len());
        for i in 0..n {
            let tgt = input[i].clamp(0.0, 1.0);
            // asymmetric slew for rise/fall shaping
            self.env = if tgt >= self.env {
                self.alpha_rise * self.env + (1.0 - self.alpha_rise) * tgt
            } else {
                self.alpha_fall * self.env + (1.0 - self.alpha_fall) * tgt
            };
            let base = C32::new(self.env * self.gain, 0.0);
            output[i] = mix_with_nco(base, &mut self.nco);
        }
        WorkReport { in_read: n, out_written: n }
    }
}
