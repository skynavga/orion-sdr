use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::{FirLowpass};
 
/// PM demodulator via quadrature (phase difference) + post LPF
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PmQuadratureDemod {
    fs: f32,           // sample rate (kept for future use; ok if unused)
    k: f32,            // gain/sensitivity applied to phase difference
    post_lp: FirLowpass,
    prev: C32,         // previous complex sample for quadrature detector
}

impl PmQuadratureDemod {
    /// `audio_bw_hz` is the post-demod audio bandwidth (low-pass cutoff).
    /// `k` is a scaling constant (1.0 is fine; adjust per modulator).
    pub fn new(fs: f32, k: f32, audio_bw_hz: f32) -> Self {
        // Gentle transition band (25% of cutoff); tweak as needed.
        let lp = FirLowpass::design(fs, audio_bw_hz, audio_bw_hz * 0.25);
        Self {
            fs,
            k,
            post_lp: lp,
            prev: C32::new(1.0, 0.0),
        }
    }
}

impl Block for PmQuadratureDemod {
    type In  = C32;
    type Out = f32;

    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        let n = input.len().min(output.len());
        if n == 0 {
            return WorkReport { in_read: 0, out_written: 0 };
        }

        // 1) Quadrature discriminator: angle( z[n] * conj(z[n-1]) )
        // This yields Δphase; for PM this is proportional to d/dt of message.
        // If your PM modulator is symmetric (no extra integration), this
        // matches the “quadrature PM” path used in your tests.
        let mut prev = self.prev;
        let mut ytmp = vec![0.0f32; n];

        for i in 0..n {
            let z = input[i];
            let w = z * prev.conj();
            let dphi = w.im.atan2(w.re);     // [-π, π]
            ytmp[i] = self.k * dphi;
            prev = z;
        }
        self.prev = prev;

        // 2) Post-lowpass to audio bandwidth
        self.post_lp.process_block(&ytmp[..], &mut output[..n]);

        WorkReport { in_read: n, out_written: n }
    }
}
