// src/demodulate/qpsk.rs
use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};

/// QPSK soft-symbol estimator: C32 IQ → C32 soft symbol.
/// Re(output) is the I decision metric, Im(output) the Q decision metric.
/// First version: coherent passthrough with gain normalization.
#[derive(Debug, Clone, Copy)]
pub struct QpskDemod {
    gain: f32,
}

impl QpskDemod {
    pub fn new(gain: f32) -> Self { Self { gain } }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
}

impl Block for QpskDemod {
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

/// QPSK hard decision: C32 soft symbol → two u8 bits (b0, b1).
///
/// Matches the QpskMapper Gray coding:
///   Re(z) >= 0 → b0 = 0,  Re(z) < 0 → b0 = 1
///   Im(z) >= 0 → b1 = 0,  Im(z) < 0 → b1 = 1
///
/// Output is written as interleaved pairs: output[2k] = b0, output[2k+1] = b1.
#[derive(Debug, Clone, Copy, Default)]
pub struct QpskDecider;

impl QpskDecider {
    pub fn new() -> Self { Self }
}

impl Block for QpskDecider {
    type In  = C32;
    type Out = u8;

    #[inline(always)]
    fn process(&mut self, input: &[C32], output: &mut [u8]) -> WorkReport {
        let n_syms = input.len().min(output.len() / 2);
        let mut i = 0;
        let nn = n_syms & !3;
        while i < nn {
            output[2*i]   = u8::from(input[i].re   < 0.0);
            output[2*i+1] = u8::from(input[i].im   < 0.0);
            output[2*i+2] = u8::from(input[i+1].re < 0.0);
            output[2*i+3] = u8::from(input[i+1].im < 0.0);
            output[2*i+4] = u8::from(input[i+2].re < 0.0);
            output[2*i+5] = u8::from(input[i+2].im < 0.0);
            output[2*i+6] = u8::from(input[i+3].re < 0.0);
            output[2*i+7] = u8::from(input[i+3].im < 0.0);
            i += 4;
        }
        while i < n_syms {
            output[2*i]   = u8::from(input[i].re < 0.0);
            output[2*i+1] = u8::from(input[i].im < 0.0);
            i += 1;
        }
        WorkReport { in_read: n_syms, out_written: n_syms * 2 }
    }
}
