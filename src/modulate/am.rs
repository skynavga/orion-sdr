// src/modulate/am.rs
use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::Rotator;


#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AmDsbMod {
    fs: f32,
    gain: f32,
    carrier_level: f32,     // 1.0 => full carrier (A3E), 0.0 => DSB-SC (A3)
    modulation_index: f32,  // <= 1.0 recommended
    clamp: bool,
    rf_nco: Rotator,        // rf_hz==0.0 => baseband AM
}

impl AmDsbMod {
    pub fn new(fs: f32, rf_hz: f32, carrier_level: f32, modulation_index: f32) -> Self {
        Self {
            fs,
            gain: 1.0,
            carrier_level,
            modulation_index,
            clamp: false,
            rf_nco: Rotator::new(rf_hz, fs),
        }
    }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
    pub fn set_clamp(&mut self, on: bool) { self.clamp = on; }
}

impl Block for AmDsbMod {
    type In  = f32;  // mono audio (-1..+1 recommended)
    type Out = C32;  // complex IQ

    #[inline(always)]
    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        let n = input.len().min(output.len());
        let mi = self.modulation_index;
        let cl = self.carrier_level;
        let g  = self.gain;

        let mut i = 0;
        let nn = n & !3;

        if self.clamp {
            while i < nn {
                // 0
                let m0 = (cl + mi * input[i]).clamp(-1.0, 1.0) * g;
                let r0 = self.rf_nco.next();
                output[i] = C32::new(m0 * r0.re, m0 * r0.im);

                // 1
                let m1 = (cl + mi * input[i+1]).clamp(-1.0, 1.0) * g;
                let r1 = self.rf_nco.next();
                output[i+1] = C32::new(m1 * r1.re, m1 * r1.im);

                // 2
                let m2 = (cl + mi * input[i+2]).clamp(-1.0, 1.0) * g;
                let r2 = self.rf_nco.next();
                output[i+2] = C32::new(m2 * r2.re, m2 * r2.im);

                // 3
                let m3 = (cl + mi * input[i+3]).clamp(-1.0, 1.0) * g;
                let r3 = self.rf_nco.next();
                output[i+3] = C32::new(m3 * r3.re, m3 * r3.im);

                i += 4;
            }
            while i < n {
                let m = (cl + mi * input[i]).clamp(-1.0, 1.0) * g;
                let r = self.rf_nco.next();
                output[i] = C32::new(m * r.re, m * r.im);
                i += 1;
            }
        } else {
            // No clamp → fewer instructions (fast path)
            while i < nn {
                // 0
                let m0 = (cl + mi * input[i]) * g;
                let r0 = self.rf_nco.next();
                output[i] = C32::new(m0 * r0.re, m0 * r0.im);

                // 1
                let m1 = (cl + mi * input[i+1]) * g;
                let r1 = self.rf_nco.next();
                output[i+1] = C32::new(m1 * r1.re, m1 * r1.im);

                // 2
                let m2 = (cl + mi * input[i+2]) * g;
                let r2 = self.rf_nco.next();
                output[i+2] = C32::new(m2 * r2.re, m2 * r2.im);

                // 3
                let m3 = (cl + mi * input[i+3]) * g;
                let r3 = self.rf_nco.next();
                output[i+3] = C32::new(m3 * r3.re, m3 * r3.im);

                i += 4;
            }
            while i < n {
                let m = (cl + mi * input[i]) * g;
                let r = self.rf_nco.next();
                output[i] = C32::new(m * r.re, m * r.im);
                i += 1;
            }
        }

        WorkReport { in_read: n, out_written: n }
    }
}
