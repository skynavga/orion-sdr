use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};
use crate::dsp::{DcBlocker, FirLowpass, Rotator};
 
#[derive(Debug, Clone)]
pub struct CwEnvelopeDemod {
    alpha: f32,   // one-pole LP smoothing factor
    y: f32,       // LP state
    gain: f32,
}

impl CwEnvelopeDemod {
    pub fn new(sample_rate: f32, _tone_hz: f32, env_bw_hz: f32) -> Self {
        // One-pole LP: alpha = exp(-2π fc / fs). Larger fc → faster tracking.
        let fc = env_bw_hz.max(1.0);
        let alpha = (-std::f32::consts::TAU * fc / sample_rate).exp();
        Self { alpha, y: 0.0, gain: 1.0 }
    }
    pub fn set_gain(&mut self, g: f32) { self.gain = g; }
}

impl crate::core::Block for CwEnvelopeDemod {
    type In = num_complex::Complex32;
    type Out = f32;

    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> crate::core::WorkReport {
        let n = input.len().min(output.len());
        let a = self.alpha;
        for i in 0..n {
            let mag = input[i].re.hypot(input[i].im);
            self.y = a * self.y + (1.0 - a) * mag;
            output[i] = self.y * self.gain;
        }
        crate::core::WorkReport { in_read: n, out_written: n }
    }
}
/// AM envelope detector: |IQ| -> DC block -> audio LP.
#[derive(Debug, Clone)]
pub struct AmEnvelopeDemod {
    lp: FirLowpass,
    dc: DcBlocker,
    gain: f32,
    scratch: Vec<f32>,
}

impl AmEnvelopeDemod {

    /// `audio_bw_hz` e.g. 5–8 kHz for AM broadcast
    pub fn new(sample_rate: f32, audio_bw_hz: f32) -> Self {
        Self {
            lp: FirLowpass::design(sample_rate, audio_bw_hz, audio_bw_hz * 0.5),
            dc: DcBlocker::new(sample_rate, 2.0),
            gain: 1.0,
            scratch: Vec::new(),
        }
    }

    pub fn set_gain(&mut self, g: f32) { self.gain = g; }

}

impl Block for AmEnvelopeDemod {
    type In = C32; type Out = f32;
    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        let n = input.len().min(output.len());
        if self.scratch.len() < n { self.scratch.resize(n, 0.0); }

        // tmp: magnitude
        let tmp = &mut self.scratch[..n];
        for i in 0..n {
            let z = input[i];
            tmp[i] = (z.re * z.re + z.im * z.im).sqrt() * self.gain;
        }

        // LP -> output
        self.lp.process_block(&tmp[..], &mut output[..n]);

        // DC block (separate buffers)
        self.dc.process(&output[..n], &mut tmp[..]);
        output[..n].copy_from_slice(&tmp[..]);

        WorkReport { in_read: n, out_written: n }
    }

}

#[derive(Debug, Clone)]
pub struct SsbProductDemod {
    rot: Rotator,
    post_lp: FirLowpass,
    dc: DcBlocker,
}

impl SsbProductDemod {
    pub fn new(fs: f32, bfo_hz: f32, audio_bw_hz: f32) -> Self {
        let trans_hz = (audio_bw_hz * 0.15).clamp(200.0, 600.0);
        let post_lp = FirLowpass::design(fs, audio_bw_hz, trans_hz);
        let rot = Rotator::new(bfo_hz, fs);
        let dc = DcBlocker::new(fs, 2.0);
        Self { rot, post_lp, dc }
    }
}

impl Block for SsbProductDemod {
    type In = C32; type Out = f32;
    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        let n = input.len().min(output.len());
        // Multiply by BFO phasor and take real (USB convention; sign flips choose sideband)
        let mut mix = vec![0.0f32; n];
        for i in 0..n {
            let w = self.rot.next();
            mix[i] = (input[i] * w).re;
        }
        let mut ytmp = vec![0.0f32; n];
        self.post_lp.process_block(&mix, &mut ytmp);
        self.dc.process(&ytmp, &mut output[..n]);
        WorkReport { in_read: n, out_written: n }
    }
}
 
 // FM Quadrature Demod
#[derive(Debug, Clone)]
 pub struct FmQuadratureDemod {
    fs: f32,
    k: f32,
    // optional translator
    xf: Option<Rotator>,
    prev: C32,
    // audio postfilter
    post_lp: FirLowpass,
 }

impl FmQuadratureDemod {

    pub fn new(fs: f32, dev_hz: f32, audio_bw_hz: f32) -> Self {
        let k = 1.0 / dev_hz.max(1.0);
        let post_lp = FirLowpass::design(fs, audio_bw_hz * 0.9, audio_bw_hz * 0.3);
         Self {
            fs,
            k,
            xf: None,
            prev: C32::new(1.0, 0.0),
            post_lp,
         }
     }

     pub fn with_translate(mut self, freq_hz: f32) -> Self {
        self.xf = Some(Rotator::new(freq_hz, self.fs)); self
    }

}
 
impl Block for FmQuadratureDemod {
    type In = C32; type Out = f32;
    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
        let n = input.len().min(output.len());
        if let Some(r) = &mut self.xf {
            // translate in-place: x := x * conj(rot)
            for i in 0..n {
                let w = r.next().conj();
                let x = input[i] * w;
                output[i] = 0.0; // will be overwritten
                self.prev = x; // just to keep prev consistent; actual demod below uses `x`
            }
        }
        // standard FM angle diff
        let mut ytmp = vec![0.0f32; n];
        for i in 0..n {
            let z = input[i];
            // angle(curr * conj(prev)) = instantaneous phase increment
            let prod = C32::new(
                z.re * self.prev.re + z.im * self.prev.im,
                z.im * self.prev.re - z.re * self.prev.im,
            );
            let dphi = prod.im.atan2(prod.re);
            ytmp[i] = dphi * self.k;
            self.prev = z;
        }

        // LP -> output
        self.post_lp.process_block(&ytmp[..], &mut output[..n]);

        WorkReport { in_read: n, out_written: n }
    }

}
 
/// PM Quadrature Demod
/// Phase-mod demodulator via quadrature (phase difference) + post LPF
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
    type In  = C32; type Out = f32;

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
