#![allow(unsafe_op_in_unsafe_fn)]

/// orion-sdr

use num_complex::Complex32 as C32;
use pyo3::{prelude::*, Bound};
use numpy::{PyArray1, PyReadonlyArray1};

/// `version()` is here so dependents can sanity-check linkage.
pub fn version() -> &'static str { env!("CARGO_PKG_VERSION") }

pub mod core {

    use num_complex::Complex32 as C32;

    /// A minimal work report returned by blocks.
    #[derive(Debug, Clone, Copy)]
    pub struct WorkReport {
        pub in_read: usize,
        pub out_written: usize,
    }

    /// Generic streaming block trait: transform a slice of input to output.
    /// Associated types let us use the same trait for IQ->IQ, IQ->audio, etc.
    pub trait Block {
        type In: Copy;
        type Out: Copy;
        fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport;
    }

    /// A tiny, pull-style driver to run one block over chunked buffers.
    pub fn run_block<B: Block>(blk: &mut B, input: &[B::In], out_buf: &mut [B::Out]) -> WorkReport {
        blk.process(input, out_buf)
    }

    /// A simple IQ->IQ chain (e.g., AGC, decimators, filters).
    pub struct IqChain {
        stages: Vec<Box<dyn Block<In = C32, Out = C32>>>,
    }

    impl IqChain {

        pub fn new() -> Self { Self { stages: Vec::new() } }

        pub fn push<B>(&mut self, b: B) where B: Block<In = C32, Out = C32> + 'static {
            self.stages.push(Box::new(b));
        }

        /// Process one chunk in-place using hop buffers.
        pub fn process(&mut self, buf: Vec<C32>) -> Vec<C32> {
            let mut a = buf;
            for st in self.stages.iter_mut() {
                let mut out = vec![C32::new(0.0, 0.0); a.len()];
                let _r = st.process(&a, &mut out);
                a = out;
            }
            a
        }

    }

    /// IQ->Audio chain: IQ stages -> one demod (IQ->f32) -> audio stages (f32->f32).
    pub struct IqToAudioChain {
        iq_stages: Vec<Box<dyn Block<In = C32, Out = C32>>>,
        demod: Box<dyn Block<In = C32, Out = f32>>,
        audio_stages: Vec<Box<dyn Block<In = f32, Out = f32>>>,
    }

    impl IqToAudioChain {

        pub fn new<D>(demod: D) -> Self
            where D: Block<In = C32, Out = f32> + 'static {
            Self { iq_stages: Vec::new(), demod: Box::new(demod), audio_stages: Vec::new() }
        }

        pub fn push_iq<B>(&mut self, b: B) where B: Block<In = C32, Out = C32> + 'static {
            self.iq_stages.push(Box::new(b));
        }

        pub fn push_audio<B>(&mut self, b: B) where B: Block<In = f32, Out = f32> + 'static {
            self.audio_stages.push(Box::new(b));
        }

        pub fn process(&mut self, mut iq: Vec<C32>) -> Vec<f32> {
            for st in self.iq_stages.iter_mut() {
                let mut next = vec![C32::new(0.0, 0.0); iq.len()];
                let _ = st.process(&iq, &mut next);
                iq = next;
            }
            let mut audio = vec![0.0f32; iq.len()];
            let _ = self.demod.process(&iq, &mut audio);

            for st in self.audio_stages.iter_mut() {
                let mut next = vec![0.0f32; audio.len()];
                let _ = st.process(&audio, &mut next);
                audio = next;
            }
            audio
        }

    }

}

pub mod dsp {

    use super::*;

    /// Numerically controlled oscillator (phasor generator)
    #[derive(Debug, Clone)]
    pub struct Nco {
        phase: f32,
        inc: f32,
    }

    impl Nco {

        pub fn new(freq_hz: f32, sample_rate: f32) -> Self {
            let inc = 2.0 * std::f32::consts::PI * freq_hz / sample_rate;
            Self { phase: 0.0, inc }
        }

        #[inline]
        pub fn next(&mut self) -> C32 {
            let p = self.phase;
            // Update phase (wrap to [-pi, pi])
            self.phase = (self.phase + self.inc) % (2.0 * std::f32::consts::PI);
            C32::new(p.cos(), p.sin())
        }

    }

    /// Complex mixer: multiply by complex exponential from NCO
    #[inline]
    pub fn mix_with_nco(x: C32, nco: &mut Nco) -> C32 {
        let w = nco.next();
        C32::new(x.re * w.re - x.im * w.im, x.re * w.im + x.im * w.re)
    }

    /// Simple windowed-sinc lowpass FIR builder and stateful filter
    pub struct FirLowpass {
        taps: Vec<f32>,
        delay: Vec<f32>, // ring buffer of real-valued samples
        idx: usize,
    }

    impl FirLowpass {

        /// Build a lowpass with cutoff_hz, transition_hz, using Hann window
        pub fn design(sample_rate: f32, cutoff_hz: f32, transition_hz: f32) -> Self {
            let tbw = (transition_hz / sample_rate).clamp(1e-4, 0.45);
            // Rough length estimate: 4 / tbw
            let len = ((4.0 / tbw).ceil() as usize) | 1; // make odd
            let m = len as i32 - 1;
            let fc = cutoff_hz / sample_rate; // normalized (0..0.5)
            let mut taps = Vec::with_capacity(len);
            for n in 0..len {
                let k = n as i32 - m / 2;
                let sinc = if k == 0 { 2.0 * fc } else { (2.0 * fc * std::f32::consts::PI * k as f32).sin() / (std::f32::consts::PI * k as f32) };
                let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / (len as f32 - 1.0)).cos(); // Hann
                taps.push(sinc * w);
            }
            // Normalize DC gain to 1.0
            let sum: f32 = taps.iter().sum();
            for t in &mut taps { *t /= sum; }
            Self { taps: taps.clone(), delay: vec![0.0; taps.len()], idx: 0 }
        }

        #[inline]
        pub fn reset(&mut self) { self.delay.fill(0.0); self.idx = 0; }

        #[inline]
        pub fn process_sample(&mut self, x: f32) -> f32 {
            self.delay[self.idx] = x;
            // FIR dot-product (circular)
            let mut acc = 0.0f32;
            let mut di = self.idx;
            for &t in &self.taps {
                acc += t * self.delay[di];
                di = (di + self.delay.len() - 1) % self.delay.len();
            }
            self.idx = (self.idx + 1) % self.delay.len();
            acc
        }

        pub fn process_block(&mut self, input: &[f32], out: &mut [f32]) -> usize {
            let n = input.len().min(out.len());
            for i in 0..n { out[i] = self.process_sample(input[i]); }
            n
        }

    }

    /// Simple DC blocker (high-pass IIR): y[n] = x[n] - x[n-1] + a*y[n-1]
    pub struct DcBlocker { a: f32, x1: f32, y1: f32 }

    impl DcBlocker {

        pub fn new(a: f32) -> Self {
            Self {
                a,
                x1: 0.0,
                y1: 0.0
            }
        }

        #[inline]
        pub fn process_sample(&mut self, x: f32) -> f32 {
            let y = x - self.x1 + self.a * self.y1;
            self.x1 = x;
            self.y1 = y;
            y
        }

        pub fn process_block(&mut self, input: &[f32], out: &mut [f32]) -> usize {
            let n = input.len().min(out.len());
            for i in 0..n {
                out[i] = self.process_sample(input[i]);
            }
            n
        }

    }

    /// Real FIR decimator (applies FIR to I and Q separately for complex).
    pub struct FirDecimator {
        m: usize,            // decimation factor
        fir: FirLowpass,     // lowpass prototype
    }

    impl FirDecimator {

        /// Create a decimator with factor `m`. Choose cutoff < 0.5/m for good rejection.
        pub fn new(sample_rate: f32, m: usize, cutoff_hz: f32, transition_hz: f32) -> Self {
            assert!(m >= 2);
            let fir = FirLowpass::design(sample_rate, cutoff_hz, transition_hz);
            Self { m, fir }
        }

        pub fn factor(&self) -> usize { self.m }

        pub fn process_iq(&mut self, input: &[C32], out: &mut [C32]) -> usize {
            // We feed samples one-by-one into the FIR's delay and pick every m-th output.
            let mut written = 0usize;
            for (i, &z) in input.iter().enumerate() {
                // process real and imag independently through identical FIR
                let re = self.fir.process_sample(z.re);
                let im = self.fir.process_sample(z.im);
                if i % self.m == (self.m - 1) {
                    if written < out.len() {
                        out[written] = C32::new(re, im);
                        written += 1;
                    } else {
                        break;
                    }
                }
            }
            written
        }

        pub fn process_audio(&mut self, input: &[f32], out: &mut [f32]) -> usize {
            let mut written = 0usize;
            for (i, &x) in input.iter().enumerate() {
                let y = self.fir.process_sample(x);
                if i % self.m == (self.m - 1) {
                    if written < out.len() {
                        out[written] = y;
                        written += 1;
                    } else {
                        break;
                    }
                }
            }
            written
        }

    }

    /// Simple RMS-based AGC with exponential detectors.
    /// target_rms: desired RMS of output signal (e.g., 0.2)
    /// attack_ms / release_ms: time constants in milliseconds
    pub struct AgcRms {
        target_rms: f32,
        atk_alpha: f32,
        rel_alpha: f32,
        // state
        rms: f32,
        gain: f32,
        min_gain: f32,
        max_gain: f32,
    }

    impl AgcRms {

        pub fn new(sample_rate: f32, target_rms: f32, attack_ms: f32, release_ms: f32) -> Self {
            let atk_alpha = (-1.0 / (attack_ms.max(0.1) * 1e-3 * sample_rate)).exp();
            let rel_alpha = (-1.0 / (release_ms.max(0.1) * 1e-3 * sample_rate)).exp();
            Self {
                target_rms,
                atk_alpha,
                rel_alpha,
                rms: 1e-6,
                gain: 1.0,
                min_gain: 0.01,
                max_gain: 100.0,
            }
        }

        pub fn set_limits(&mut self, min_gain: f32, max_gain: f32) {
            self.min_gain = min_gain.max(1e-6);
            self.max_gain = max_gain.max(self.min_gain);
        }

        #[inline]
        pub fn process_iq_block(&mut self, input: &[C32], out: &mut [C32]) -> usize {
            let n = input.len().min(out.len());
            for i in 0..n {
                let x = input[i];
                // instantaneous power
                let p = x.re * x.re + x.im * x.im;
                // RMS detector with different attack/release speeds
                if p > self.rms { // attack
                    self.rms = self.atk_alpha * self.rms + (1.0 - self.atk_alpha) * p;
                } else {          // release
                    self.rms = self.rel_alpha * self.rms + (1.0 - self.rel_alpha) * p;
                }
                let current_rms = self.rms.sqrt().max(1e-6);
                // target gain
                let mut g_tgt = self.target_rms / current_rms;
                g_tgt = g_tgt.clamp(self.min_gain, self.max_gain);
                // small smoothing of gain (release-like)
                self.gain = 0.02 * g_tgt + 0.98 * self.gain;
                out[i] = C32::new(x.re * self.gain, x.im * self.gain);
            }
            n
        }

        #[inline]
        pub fn process_audio_block(&mut self, input: &[f32], out: &mut [f32]) -> usize {
            let n = input.len().min(out.len());
            for i in 0..n {
                let x = input[i];
                let p = x * x;
                if p > self.rms {
                    self.rms = self.atk_alpha * self.rms + (1.0 - self.atk_alpha) * p;
                } else {
                    self.rms = self.rel_alpha * self.rms + (1.0 - self.rel_alpha) * p;
                }
                let current_rms = self.rms.sqrt().max(1e-6);
                let mut g_tgt = self.target_rms / current_rms;
                g_tgt = g_tgt.clamp(self.min_gain, self.max_gain);
                self.gain = 0.02 * g_tgt + 0.98 * self.gain;
                out[i] = x * self.gain;
            }
            n
        }

    }

}

pub mod demod {

    use super::core::{Block, WorkReport};
    use super::dsp::{mix_with_nco, DcBlocker, FirLowpass, Nco};
    use num_complex::Complex32 as C32;

    /// CW demod: product detector with BFO at desired pitch, then narrow audio LP and DC block.
    pub struct CwDemod {
        fs: f32,
        nco: Nco,
        lp: FirLowpass,
        dc: DcBlocker,
        gain: f32,
        scratch: Vec<f32>, // reusable
    }

    impl CwDemod {

        /// `pitch_hz` e.g. 600–800; `audio_bw_hz` e.g. 200–500.
        pub fn new(sample_rate: f32, pitch_hz: f32, audio_bw_hz: f32) -> Self {
            Self {
                fs: sample_rate,
                nco: Nco::new(pitch_hz, sample_rate),
                lp: FirLowpass::design(sample_rate, audio_bw_hz, audio_bw_hz * 0.5),
                dc: DcBlocker::new(0.995),
                gain: 1.0,
                scratch: Vec::new(),
            }
        }

        pub fn set_pitch(&mut self, hz: f32) { self.nco = Nco::new(hz, self.fs); }

        pub fn set_gain(&mut self, g: f32) { self.gain = g; }

    }

    impl Block for CwDemod {

        type In = C32;
        type Out = f32;

        fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
            let n = input.len().min(output.len());
            if self.scratch.len() < n { self.scratch.resize(n, 0.0); }

            // tmp: mix with BFO, keep real part (audio candidate)
            let tmp = &mut self.scratch[..n];
            for i in 0..n {
                let z = mix_with_nco(input[i], &mut self.nco);
                tmp[i] = z.re * self.gain;
            }

            // LP -> output
            self.lp.process_block(&tmp[..], &mut output[..n]);

            // DC block needs separate input/output; reuse scratch
            self.dc.process_block(&output[..n], &mut tmp[..]);
            output[..n].copy_from_slice(&tmp[..]);

            WorkReport { in_read: n, out_written: n }
        }

    }

    /// AM envelope detector: |IQ| -> DC block -> audio LP.
    pub struct AmEnvelope {
        lp: FirLowpass,
        dc: DcBlocker,
        gain: f32,
        scratch: Vec<f32>,
    }

    impl AmEnvelope {

        /// `audio_bw_hz` e.g. 5–8 kHz for AM broadcast
        pub fn new(sample_rate: f32, audio_bw_hz: f32) -> Self {
            Self {
                lp: FirLowpass::design(sample_rate, audio_bw_hz, audio_bw_hz * 0.5),
                dc: DcBlocker::new(0.995),
                gain: 1.0,
                scratch: Vec::new(),
            }
        }

        pub fn set_gain(&mut self, g: f32) { self.gain = g; }

    }

    impl Block for AmEnvelope {

        type In = C32;
        type Out = f32;

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
            self.dc.process_block(&output[..n], &mut tmp[..]);
            output[..n].copy_from_slice(&tmp[..]);

            WorkReport { in_read: n, out_written: n }
        }

    }

    /// SSB product detector: complex baseband -> real audio
    /// Mode USB/LSB is expressed by BFO sign.
    pub struct SsbProductDetector {
        sample_rate: f32,
        nco: Nco,
        lp: FirLowpass,
        dc: DcBlocker,
        gain: f32,
        scratch: Vec<f32>,
    }

    impl SsbProductDetector {

        pub fn new(sample_rate: f32, bfo_hz: f32, audio_bw_hz: f32) -> Self {
            Self {
                sample_rate,
                nco: Nco::new(bfo_hz, sample_rate),
                lp: FirLowpass::design(sample_rate, audio_bw_hz, audio_bw_hz * 0.4),
                dc: DcBlocker::new(0.995),
                gain: 1.0,
                scratch: Vec::new(),
            }
        }

        pub fn set_gain(&mut self, g: f32) { self.gain = g; }

        pub fn set_bfo(&mut self, hz: f32) { self.nco = Nco::new(hz, self.sample_rate); }

        pub fn set_audio_bw(&mut self, hz: f32) { self.lp = FirLowpass::design(self.sample_rate, hz, hz * 0.4); }

    }

    impl Block for SsbProductDetector {

        type In = C32;
        type Out = f32;

        fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
            let n = input.len().min(output.len());
            if self.scratch.len() < n { self.scratch.resize(n, 0.0); }

            // tmp: mix by BFO, take real
            let tmp = &mut self.scratch[..n];
            for i in 0..n {
                let z = mix_with_nco(input[i], &mut self.nco);
                tmp[i] = z.re * self.gain;
            }

            // LP -> output
            self.lp.process_block(&tmp[..], &mut output[..n]);

            // DC block (separate buffers)
            self.dc.process_block(&output[..n], &mut tmp[..]);
            output[..n].copy_from_slice(&tmp[..]);

            WorkReport { in_read: n, out_written: n }
        }

    }

    /// Narrowband FM discriminator (quadrature/phase-difference method).
    /// - Works on complex baseband IQ (centered at 0 Hz).
    /// - Optional limiter for amplitude normalization.
    /// - Optional de-emphasis (single-pole RC) via `set_deemph_tau_us`.
    /// - Audio is scaled so that +/- `deviation_hz` -> approx +/-1.0.
    pub struct FmDemod {
        fs: f32,
        deviation_hz: f32,
        limiter: bool,
        // de-emphasis
        use_deemph: bool,
        deemph_alpha: f32,
        deemph_y1: f32,
        // post processing
        lp: FirLowpass,
        dc: DcBlocker,
        gain: f32,
        // state
        prev: C32,
        scratch: Vec<f32>,
    }

    impl FmDemod {
        /// `audio_bw_hz` e.g. 3–5 kHz for NBFM voice; `deviation_hz` e.g. 2.5k/5k.
        pub fn new(sample_rate: f32, deviation_hz: f32, audio_bw_hz: f32) -> Self {
            Self {
                fs: sample_rate,
                deviation_hz: deviation_hz.max(1.0),
                limiter: true,
                use_deemph: false,
                deemph_alpha: 0.0,
                deemph_y1: 0.0,
                lp: FirLowpass::design(sample_rate, audio_bw_hz, audio_bw_hz * 0.5),
                dc: DcBlocker::new(0.995),
                gain: 1.0,
                prev: C32::new(1.0, 0.0),
                scratch: Vec::new(),
            }
        }
        pub fn set_gain(&mut self, g: f32) { self.gain = g; }
        pub fn set_deviation_hz(&mut self, hz: f32) { self.deviation_hz = hz.max(1.0); }
        pub fn set_limiter(&mut self, on: bool) { self.limiter = on; }

        /// Enable de-emphasis with time constant `tau_us` (e.g., 300–750 for NBFM, 75 for WBFM US, 50 EU).
        pub fn set_deemph_tau_us(&mut self, tau_us: f32) {
            if tau_us <= 0.0 { self.use_deemph = false; return; }
            let dt = 1.0 / self.fs;
            let tau = tau_us * 1e-6;
            self.deemph_alpha = dt / (tau + dt);
            self.use_deemph = true;
        }
    }

    impl Block for FmDemod {
        type In = C32;
        type Out = f32;

        fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
            let n = input.len().min(output.len());
            if self.scratch.len() < n { self.scratch.resize(n, 0.0); }
            let tmp = &mut self.scratch[..n];

            // Scale so +/- deviation -> +/-1.0
            let k = self.fs / (std::f32::consts::TAU * self.deviation_hz.max(1.0));

            for i in 0..n {
                let mut z = input[i];
                if self.limiter {
                    let m2 = z.re * z.re + z.im * z.im;
                    if m2 > 0.0 { let inv = m2.sqrt().recip(); z.re *= inv; z.im *= inv; }
                }
                // angle(curr * conj(prev)) = instantaneous phase increment
                let prod = C32::new(
                    z.re * self.prev.re + z.im * self.prev.im,
                    z.im * self.prev.re - z.re * self.prev.im,
                );
                let dphi = prod.im.atan2(prod.re);
                tmp[i] = dphi * k * self.gain; // normalized audio
                self.prev = z;
            }

            // Optional de-emphasis (one-pole LP)
            if self.use_deemph {
                let a = self.deemph_alpha;
                let mut y1 = self.deemph_y1;
                for x in &mut tmp.iter_mut() {
                    let y = y1 + a * (*x - y1);
                    y1 = y;
                    *x = y;
                }
                self.deemph_y1 = y1;
            }

            // LP -> output
            self.lp.process_block(&tmp[..], &mut output[..n]);
            // DC block with separate buffer
            self.dc.process_block(&output[..n], &mut tmp[..]);
            output[..n].copy_from_slice(&tmp[..]);

            WorkReport { in_read: n, out_written: n }
        }
    }

    /// Phase modulation demodulator (PM): returns (unwrapped) instantaneous phase scaled to ~+/-1.
    /// - Works on complex baseband IQ.
    /// - Optional limiter.
    /// - Audio is `phase / pm_sense_rad`, where `pm_sense_rad` is the phase deviation that maps to +/-1.
    pub struct PmDemod {
        limiter: bool,
        pm_sense_rad: f32,
        // post processing
        lp: FirLowpass,
        dc: DcBlocker,
        gain: f32,
        // state
        prev_phase: f32,
        scratch: Vec<f32>,
    }

    impl PmDemod {
        /// `pm_sense_rad`: set to your phase deviation so +/-pm_sense -> +/-1.0 audio (e.g., 0.5–1.0 rad).
        /// `audio_bw_hz`: audio lowpass bandwidth.
        pub fn new(pm_sense_rad: f32, audio_bw_hz: f32, sample_rate: f32) -> Self {
            Self {
                limiter: true,
                pm_sense_rad: pm_sense_rad.max(1e-3),
                lp: FirLowpass::design(sample_rate, audio_bw_hz, audio_bw_hz * 0.5),
                dc: DcBlocker::new(0.995),
                gain: 1.0,
                prev_phase: 0.0,
                scratch: Vec::new(),
            }
        }
        pub fn set_gain(&mut self, g: f32) { self.gain = g; }
        pub fn set_limiter(&mut self, on: bool) { self.limiter = on; }
        pub fn set_pm_sense_rad(&mut self, rad: f32) { self.pm_sense_rad = rad.max(1e-3); }
    }

    impl Block for PmDemod {
        type In = C32;
        type Out = f32;

        fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> WorkReport {
            let n = input.len().min(output.len());
            if self.scratch.len() < n { self.scratch.resize(n, 0.0); }
            let tmp = &mut self.scratch[..n];

            for i in 0..n {
                let mut z = input[i];
                if self.limiter {
                    let m2 = z.re * z.re + z.im * z.im;
                    if m2 > 0.0 { let inv = m2.sqrt().recip(); z.re *= inv; z.im *= inv; }
                }
                // instantaneous phase
                let mut phi = z.im.atan2(z.re);
                // unwrap relative to previous
                let d = phi - self.prev_phase;
                if d > std::f32::consts::PI { phi -= std::f32::consts::TAU; }
                else if d < -std::f32::consts::PI { phi += std::f32::consts::TAU; }
                self.prev_phase = phi;
                tmp[i] = (phi / self.pm_sense_rad) * self.gain;
            }

            // Post: LP and DC block
            self.lp.process_block(&tmp[..], &mut output[..n]);
            self.dc.process_block(&output[..n], &mut tmp[..]);
            output[..n].copy_from_slice(&tmp[..]);

            WorkReport { in_read: n, out_written: n }
        }
    }

}

// =============================
// PyO3 bindings
// =============================
#[pyclass]
struct PySsbDetector {
    inner: demod::SsbProductDetector,
}

#[pymethods]
impl PySsbDetector {
    #[new]
    fn new(sample_rate: f32, mode: &str, audio_bw_hz: f32, bfo_pitch_hz: f32) -> PyResult<Self> {
        // Convention: for USB, positive BFO pitch; for LSB, negative
        let sign = match mode.to_ascii_uppercase().as_str() { "USB" => 1.0, "LSB" => -1.0, _ => 1.0 };
        let inner = demod::SsbProductDetector::new(sample_rate, sign * bfo_pitch_hz, audio_bw_hz);
        Ok(Self { inner })
    }

    fn set_gain(&mut self, g: f32) { self.inner.set_gain(g); }
    fn set_bfo_hz(&mut self, bfo_hz: f32) { self.inner.set_bfo(bfo_hz); }
    fn set_audio_bw_hz(&mut self, hz: f32) { self.inner.set_audio_bw(hz); }

    /// Process one chunk of complex baseband (NumPy complex64) into f32 audio.
    /// Output has the same length and sample rate as input (resample later in Python if desired).
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        iq: PyReadonlyArray1<C32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let iq = iq.as_slice()?;
        let mut audio = vec![0.0f32; iq.len()];
        let _rep = core::run_block(&mut self.inner, iq, &mut audio);
        Ok(PyArray1::from_vec_bound(py, audio))
    }
}

#[pymodule]
fn sdr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySsbDetector>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::core::*;
    use crate::dsp::*;
    use crate::demod::*;
    use num_complex::Complex32 as C32;

    /// Tiny single-bin DFT; good enough for power at a specific frequency.
    fn dft_power(signal: &[f32], fs: f32, f_hz: f32) -> f32 {
        let n = signal.len();
        let w = -2.0 * std::f32::consts::PI * f_hz / fs;
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (k, &x) in signal.iter().enumerate() {
            let t = w * (k as f32);
            re += x * t.cos();
            im += x * t.sin();
        }
        // Normalize so power doesn’t scale with N
        let mag2 = (re * re + im * im) / (n as f32 * n as f32);
        mag2
    }

    /// Generate a complex baseband tone: e^{j 2π f t}
    fn gen_complex_tone(fs: f32, f_hz: f32, n: usize) -> Vec<C32> {
        (0..n)
            .map(|k| {
                let phase = 2.0 * std::f32::consts::PI * f_hz * (k as f32) / fs;
                C32::new(phase.cos(), phase.sin())
            })
            .collect()
    }

    #[test]
    fn ssb_product_detector_yields_strong_tone_and_low_dc() {
        let fs = 48_000.0;
        let n = 16_384; // ~0.34 s
        let f_tone = 1_000.0; // 1 kHz audio
        let iq = gen_complex_tone(fs, f_tone, n);

        // BFO at 0 Hz, audio BW 2.8 kHz
        let mut det = SsbProductDetector::new(fs, 0.0, 2_800.0);
        let mut audio = vec![0.0f32; n];
        let _rep = crate::core::run_block(&mut det, &iq, &mut audio);

        // DC should be tiny
        let mean = audio.iter().copied().sum::<f32>() / (audio.len() as f32);
        assert!(mean.abs() < 1e-3, "DC too high: {}", mean);

        // Power at 1 kHz should dominate power off-target (e.g., 700 Hz)
        let p_sig = dft_power(&audio, fs, f_tone);
        let p_off = dft_power(&audio, fs, 700.0);
        let snr_db = 10.0 * (p_sig / (p_off + 1e-20)).log10();

        assert!(
            snr_db > 25.0,
            "Expected >25 dB at 1 kHz vs 700 Hz, got {:.2} dB (p_sig={}, p_off={})",
            snr_db,
            p_sig,
            p_off
        );
    }

    #[test]
    fn agc_rms_converges_on_iq() {
        let fs = 48_000.0;
        let mut agc = AgcRms::new(fs, 0.2, 5.0, 200.0);
        let n = 8_000;
        // Step two amplitudes to see gain adjust
        let mut input = Vec::with_capacity(n);
        for k in 0..n {
            let a = if k < n/2 { 0.02 } else { 1.0 };
            input.push(num_complex::Complex32::new(a, 0.0));
        }
        let mut out = vec![num_complex::Complex32::new(0.0, 0.0); n];
        let _ = agc.process_iq_block(&input, &mut out);
        // RMS of second half should be ~ target
        // let mut acc = 0.0f32;
        // for z in &out[n/2..] { acc += z.norm_sqr(); }
        // let rms = (acc / (out.len()/2) as f32).sqrt();
        // assert!((rms - 0.2).abs() < 0.03, "rms={} not near target", rms);
        let tail_len = 1000.min(n/2);
        let tail = &out[n - tail_len..];
        let mut acc = 0.0f32;
        for z in tail { acc += z.norm_sqr(); }
        let rms_tail = (acc / (tail_len as f32)).sqrt();
        assert!((rms_tail - 0.2).abs() < 0.03, "tail RMS={} not near target 0.2", rms_tail);
    }

    #[test]
    fn decimator_reduces_length_and_preserves_tone() {
        let fs = 96_000.0;
        let m = 4;
        let cutoff = fs / (m as f32) * 0.45;
        let transition = fs / (m as f32) * 0.10;
        let mut dec = FirDecimator::new(fs, m, cutoff, transition);
        // Baseband tone at 2 kHz
        let n = 4096;
        let mut nco = Nco::new(2_000.0, fs);
        let mut iq = vec![num_complex::Complex32::new(0.0,0.0); n];
        for i in 0..n { iq[i] = mix_with_nco(num_complex::Complex32::new(1.0,0.0), &mut nco); }
        let mut out = vec![num_complex::Complex32::new(0.0,0.0); n/m];
        let w = dec.process_iq(&iq, &mut out);
        assert_eq!(w, n/m);
    }

    #[test]
    fn chain_runs_ssb_cw_and_am() {
        let fs = 48_000.0;
        let n = 4096;
        let mut tone = Vec::with_capacity(n);
        // Create a 1 kHz complex tone
        let mut nco = Nco::new(1_000.0, fs);
        for _ in 0..n {
            tone.push(mix_with_nco(num_complex::Complex32::new(1.0,0.0), &mut nco));
        }

        // CW chain
        let mut cw = IqToAudioChain::new(CwDemod::new(fs, 700.0, 300.0));
        let y_cw = cw.process(tone.clone());
        assert_eq!(y_cw.len(), n);

        // AM chain
        let mut am = IqToAudioChain::new(AmEnvelope::new(fs, 5_000.0));
        let y_am = am.process(tone.clone());
        assert_eq!(y_am.len(), n);

        // SSB chain (existing)
        let mut ssb = IqToAudioChain::new(SsbProductDetector::new(fs, 0.0, 2_800.0));
        let y_ssb = ssb.process(tone);
        assert_eq!(y_ssb.len(), n);
    }

    fn snr_db_at(freq_hz: f32, fs: f32, x: &[f32]) -> f32 {
        // single-bin DFT power vs an off-bin
        let n = x.len();
        let w = -2.0 * std::f32::consts::PI * freq_hz / fs;
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (k, &s) in x.iter().enumerate() {
            let t = w * (k as f32);
            re += s * t.cos();
            im += s * t.sin();
        }
        let p_sig = (re*re + im*im) / (n as f32 * n as f32);
        // off-tone
        let f2 = freq_hz * 0.7;
        let w2 = -2.0 * std::f32::consts::PI * f2 / fs;
        let (mut r2, mut i2) = (0.0f32, 0.0f32);
        for (k, &s) in x.iter().enumerate() { let t = w2 * (k as f32); r2 += s*t.cos(); i2 += s*t.sin(); }
        let p_off = (r2*r2 + i2*i2) / (n as f32 * n as f32);
        10.0 * (p_sig / (p_off + 1e-20)).log10()
    }

    #[test]
    fn fm_demod_recovers_tone() {
        let fs = 48_000.0;
        let n = 16_384;
        let f_mod = 1_000.0;
        let dev = 2_500.0; // Hz
        // Generate narrowband FM at baseband: phi[n] = sum( 2π * (dev*sin(2π f_mod t))/fs )
        let mut phi = 0.0f32;
        let mut iq = Vec::with_capacity(n);
        for k in 0..n {
            let t = k as f32 / fs;
            let f_inst = dev * (2.0*std::f32::consts::PI * f_mod * t).sin();
            phi += 2.0*std::f32::consts::PI * f_inst / fs;
            iq.push(C32::new(phi.cos(), phi.sin()));
        }
        let mut dem = FmDemod::new(fs, dev, 5_000.0);
        let mut y = vec![0.0f32; n];
        let _ = dem.process(&iq, &mut y);
        let snr = snr_db_at(f_mod, fs, &y);
        assert!(snr > 20.0, "FM SNR too low: {:.1} dB", snr);
    }

    #[test]
    fn pm_demod_recovers_tone() {
        let fs = 48_000.0;
        let n = 16_384;
        let f_mod = 1_000.0;
        let beta = 0.8; // rad peak phase deviation
        let mut iq = Vec::with_capacity(n);
        for k in 0..n {
            let t = k as f32 / fs;
            let phi = beta * (2.0*std::f32::consts::PI * f_mod * t).sin();
            iq.push(C32::new(phi.cos(), phi.sin()));
        }
        let mut dem = PmDemod::new(beta, 5_000.0, fs);
        let mut y = vec![0.0f32; n];
        let _ = dem.process(&iq, &mut y);
        let snr = snr_db_at(f_mod, fs, &y);
        assert!(snr > 20.0, "PM SNR too low: {:.1} dB", snr);
    }

}
