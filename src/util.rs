use num_complex::Complex32 as C32;
use crate::core::{Block, WorkReport};

/// Root-mean-square of a real slice.
#[inline]
pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() { return 0.0; }
    let s: f32 = x.iter().map(|v| v*v).sum();
    (s / (x.len() as f32)).sqrt()
}

/// Hann window (periodic) of length n.
pub fn hann(n: usize) -> Vec<f32> {
    (0..n).map(|k| 0.5 - 0.5 * (core::f32::consts::TAU * k as f32 / n as f32).cos()).collect()
}

/// Generate a real tone (sine) with amplitude `amp`.
pub fn tone(fs: f32, f_hz: f32, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|k| amp * (core::f32::consts::TAU * f_hz * (k as f32) / fs).sin())
        .collect()
}

/// Generate a complex baseband tone: e^{j 2π f t}
pub fn gen_complex_tone(fs: f32, f_hz: f32, n: usize) -> Vec<C32> {
    (0..n)
        .map(|k| {
            let ph = core::f32::consts::TAU * f_hz * (k as f32) / fs;
            C32::new(ph.cos(), ph.sin())
        })
        .collect()
}

/// Single-bin SNR around `f_hz` using a Hann window and simple DFT.
pub fn snr_db_at(fs: f32, f_hz: f32, x: &[f32]) -> f32 {
    let n = x.len().max(1);
    let w = hann(n);
    let two_pi = core::f32::consts::TAU;
    let mut re = 0.0f32;
    let mut im = 0.0f32;
    for (k, (&xi, &wi)) in x.iter().zip(w.iter()).enumerate() {
        let ph = two_pi * f_hz * (k as f32) / fs;
        re += wi * xi * ph.cos();
        im += wi * xi * ph.sin();
    }
    let sig = (re*re + im*im).sqrt() / (w.iter().sum::<f32>() + 1e-12);
    // noise estimate: total power minus signal power (coarse but stable for tests)
    let p_total: f32 = x.iter().map(|v| v*v).sum::<f32>() / (n as f32);
    let p_sig = sig * sig;
    let p_noise = (p_total - p_sig).max(1e-12);
    10.0 * (p_sig / p_noise).log10()
}

/// Measure throughput of a closure that processes `n` samples; returns (Msps, seconds).
pub fn measure<F: FnMut() -> usize>(mut f: F, n: usize) -> (f32, f32) {
    let t0 = std::time::Instant::now();
    let _ = f();
    let dt = t0.elapsed().as_secs_f32();
    let msps = (n as f32) / dt / 1e6;
    (msps, dt)
}

/// Run a `Block` into a preallocated output buffer.
/// Matches test usage: `run_block(&mut blk, &input, &mut output)`.
#[inline]
pub fn run_block<B: Block>(blk: &mut B, input: &[B::In], output: &mut [B::Out]) -> WorkReport {
    blk.process(input, output)
}

/// Convenience: run a `Block` and collect output into a Vec of the same length as input.
#[inline]
pub fn run_block_vec<B: Block>(blk: &mut B, input: &[B::In]) -> (Vec<B::Out>, WorkReport)
where
    B::Out: Default + Copy,
{
    let mut out = vec![B::Out::default(); input.len()];
    let wr = blk.process(input, &mut out);
    (out, wr)
}
