#![allow(unsafe_op_in_unsafe_fn)]

/// orion-sdr

use num_complex::Complex32 as C32;
use pyo3::{prelude::*, Bound};
use numpy::{PyArray1, PyReadonlyArray1};

/// `version()` is here so dependents can sanity-check linkage.
pub fn version() -> &'static str { env!("CARGO_PKG_VERSION") }

pub mod core {

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
        pub fn new(a: f32) -> Self { Self { a, x1: 0.0, y1: 0.0 } }
        #[inline]
        pub fn process_sample(&mut self, x: f32) -> f32 { let y = x - self.x1 + self.a * self.y1; self.x1 = x; self.y1 = y; y }
        pub fn process_block(&mut self, input: &[f32], out: &mut [f32]) -> usize { let n = input.len().min(out.len()); for i in 0..n { out[i] = self.process_sample(input[i]); } n }
    }
}

pub mod demod {
    use super::core::{Block, WorkReport};
    use super::dsp::{mix_with_nco, DcBlocker, FirLowpass, Nco};
    use super::*;

    /// SSB product detector: complex baseband -> real audio
    /// Mode USB/LSB is expressed by BFO sign.
    pub struct SsbProductDetector {
        sample_rate: f32,
        nco: Nco,
        lp: FirLowpass,
        dc: DcBlocker,
        gain: f32,
    }

    impl SsbProductDetector {
        pub fn new(sample_rate: f32, bfo_hz: f32, audio_bw_hz: f32) -> Self {
            let nco = Nco::new(bfo_hz, sample_rate);
            let lp = FirLowpass::design(sample_rate, audio_bw_hz, audio_bw_hz * 0.4);
            let dc = DcBlocker::new(0.995);
            Self { sample_rate, nco, lp, dc, gain: 1.0 }
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
            // 1) Mix by BFO (complex multiply) to shift the selected sideband
            // 2) Take real part as audio candidate
            // 3) Lowpass to audio BW
            // 4) DC block (remove residual carrier/DC)
            let mut tmp = vec![0.0f32; n];
            for i in 0..n {
                let z = mix_with_nco(input[i], &mut self.nco);
                tmp[i] = z.re * self.gain;
            }
            self.lp.process_block(&tmp[..n], &mut output[..n]);
            // self.dc.process_block(&output[..n], &mut output[..n]);
            let mut tmp = vec![0.0f32; n];
            for i in 0..n {
                let z = mix_with_nco(input[i], &mut self.nco);
                tmp[i] = z.re * self.gain;
            }
            self.lp.process_block(&tmp[..n], &mut output[..n]);

            // DC block needs separate input/output slices:
            let mut tmp2 = vec![0.0f32; n];
            self.dc.process_block(&output[..n], &mut tmp2[..n]);
            output[..n].copy_from_slice(&tmp2[..n]);

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

        // Replace `audio.into_pyarray(py)` with:
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
    use crate::demod::SsbProductDetector;

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
    fn gen_complex_tone(fs: f32, f_hz: f32, n: usize) -> Vec<num_complex::Complex32> {
        (0..n)
            .map(|k| {
                let phase = 2.0 * std::f32::consts::PI * f_hz * (k as f32) / fs;
                num_complex::Complex32::new(phase.cos(), phase.sin())
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
}
