// src/demodulate/bpsk.rs
use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};

/// BPSK soft-symbol estimator: C32 IQ → C32 soft symbol.
/// Re(output) is the decision metric; Im(output) is quadrature error.
/// First version: coherent passthrough with gain normalization.
#[derive(Debug, Clone, Copy)]
pub struct BpskDemod {
    gain: f32,
}

impl BpskDemod {
    pub fn new(gain: f32) -> Self { Self { gain } }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
}

impl Block for BpskDemod {
    type In  = C32;
    type Out = C32;

    #[inline(always)]
    fn process(&mut self, input: &[C32], output: &mut [C32]) -> WorkReport {
        let n = input.len().min(output.len());
        let g = self.gain;
        let mut i = 0;
        let nn = n & !3;
        while i < nn {
            output[i]   = C32::new(g * input[i].re,   g * input[i].im);
            output[i+1] = C32::new(g * input[i+1].re, g * input[i+1].im);
            output[i+2] = C32::new(g * input[i+2].re, g * input[i+2].im);
            output[i+3] = C32::new(g * input[i+3].re, g * input[i+3].im);
            i += 4;
        }
        while i < n {
            output[i] = C32::new(g * input[i].re, g * input[i].im);
            i += 1;
        }
        WorkReport { in_read: n, out_written: n }
    }
}

/// BPSK hard decision: C32 soft symbol → u8 bit (0 or 1).
/// Re(z) >= 0 → bit 0,  Re(z) < 0 → bit 1.
#[derive(Debug, Clone, Copy, Default)]
pub struct BpskDecider;

impl BpskDecider {
    pub fn new() -> Self { Self }
}

impl Block for BpskDecider {
    type In  = C32;
    type Out = u8;

    #[inline(always)]
    fn process(&mut self, input: &[C32], output: &mut [u8]) -> WorkReport {
        let n = input.len().min(output.len());
        let mut i = 0;
        let nn = n & !3;
        while i < nn {
            output[i]   = u8::from(input[i].re   < 0.0);
            output[i+1] = u8::from(input[i+1].re < 0.0);
            output[i+2] = u8::from(input[i+2].re < 0.0);
            output[i+3] = u8::from(input[i+3].re < 0.0);
            i += 4;
        }
        while i < n {
            output[i] = u8::from(input[i].re < 0.0);
            i += 1;
        }
        WorkReport { in_read: n, out_written: n }
    }
}
