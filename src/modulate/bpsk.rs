// src/modulate/bpsk.rs
use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::Rotator;

/// BPSK constellation mapper: u8 bit (LSB) → C32 symbol.
/// bit 0 → (+1, 0),  bit 1 → (−1, 0)
#[derive(Debug, Clone, Copy, Default)]
pub struct BpskMapper;

impl BpskMapper {
    pub fn new() -> Self { Self }
}

impl Block for BpskMapper {
    type In  = u8;
    type Out = C32;

    #[inline(always)]
    fn process(&mut self, input: &[u8], output: &mut [C32]) -> WorkReport {
        let n = input.len().min(output.len());
        let mut i = 0;
        let nn = n & !3;
        while i < nn {
            output[i]   = C32::new(if (input[i]   & 1) == 0 { 1.0 } else { -1.0 }, 0.0);
            output[i+1] = C32::new(if (input[i+1] & 1) == 0 { 1.0 } else { -1.0 }, 0.0);
            output[i+2] = C32::new(if (input[i+2] & 1) == 0 { 1.0 } else { -1.0 }, 0.0);
            output[i+3] = C32::new(if (input[i+3] & 1) == 0 { 1.0 } else { -1.0 }, 0.0);
            i += 4;
        }
        while i < n {
            output[i] = C32::new(if (input[i] & 1) == 0 { 1.0 } else { -1.0 }, 0.0);
            i += 1;
        }
        WorkReport { in_read: n, out_written: n }
    }
}

/// BPSK waveform stage: C32 symbols → C32 IQ.
/// rf_hz = 0.0 → baseband passthrough with gain.
/// rf_hz != 0.0 → rotate symbols onto carrier via Rotator.
#[derive(Debug, Clone)]
pub struct BpskMod {
    gain: f32,
    rot:  Rotator,
}

impl BpskMod {
    pub fn new(fs: f32, rf_hz: f32, gain: f32) -> Self {
        Self { gain, rot: Rotator::new(rf_hz, fs) }
    }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
}

impl Block for BpskMod {
    type In  = C32;
    type Out = C32;

    #[inline(always)]
    fn process(&mut self, input: &[C32], output: &mut [C32]) -> WorkReport {
        let n = input.len().min(output.len());
        let g = self.gain;
        let mut i = 0;
        let nn = n & !3;
        while i < nn {
            let s0 = input[i];   let r0 = self.rot.next();
            let s1 = input[i+1]; let r1 = self.rot.next();
            let s2 = input[i+2]; let r2 = self.rot.next();
            let s3 = input[i+3]; let r3 = self.rot.next();
            output[i]   = C32::new(g * s0.re.mul_add(r0.re, -s0.im * r0.im), g * s0.im.mul_add(r0.re, s0.re * r0.im));
            output[i+1] = C32::new(g * s1.re.mul_add(r1.re, -s1.im * r1.im), g * s1.im.mul_add(r1.re, s1.re * r1.im));
            output[i+2] = C32::new(g * s2.re.mul_add(r2.re, -s2.im * r2.im), g * s2.im.mul_add(r2.re, s2.re * r2.im));
            output[i+3] = C32::new(g * s3.re.mul_add(r3.re, -s3.im * r3.im), g * s3.im.mul_add(r3.re, s3.re * r3.im));
            i += 4;
        }
        while i < n {
            let s = input[i]; let r = self.rot.next();
            output[i] = C32::new(g * s.re.mul_add(r.re, -s.im * r.im), g * s.im.mul_add(r.re, s.re * r.im));
            i += 1;
        }
        WorkReport { in_read: n, out_written: n }
    }
}
