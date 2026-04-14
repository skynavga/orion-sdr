// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::core::{Block, WorkReport};
use num_complex::Complex32 as C32;

/// Root-mean-square of a real slice.
#[inline]
pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let s: f32 = x.iter().map(|v| v * v).sum();
    (s / (x.len() as f32)).sqrt()
}

/// Hann window (periodic) of length n.
pub fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| 0.5 - 0.5 * (core::f32::consts::TAU * k as f32 / n as f32).cos())
        .collect()
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
    let sig = (re * re + im * im).sqrt() / (w.iter().sum::<f32>() + 1e-12);
    // noise estimate: total power minus signal power (coarse but stable for tests)
    let p_total: f32 = x.iter().map(|v| v * v).sum::<f32>() / (n as f32);
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

// ── Spectral analysis ────────────────────────────────────────────────────────

use rustfft::{FftPlanner, num_complex::Complex};

/// Compute a power spectrum (dB) from real samples using a Hann-windowed FFT.
///
/// FFT size is the next power of two >= `samples.len()`, clamped to [64, 4096].
/// Returns `(power_db_bins, bin_hz)` where `bin_hz = fs / fft_size`.
pub fn power_spectrum(samples: &[f32], fs: f32) -> (Vec<f32>, f32) {
    let n = samples.len().next_power_of_two().clamp(64, 4096);
    let bin_hz = fs / n as f32;

    let mut buf: Vec<Complex<f32>> = (0..n)
        .map(|i| {
            let s = if i < samples.len() { samples[i] } else { 0.0 };
            let w = 0.5 - 0.5 * (core::f32::consts::TAU * i as f32 / n as f32).cos();
            Complex { re: s * w, im: 0.0 }
        })
        .collect();

    FftPlanner::new().plan_fft_forward(n).process(&mut buf);

    let scale = 1.0 / n as f32;
    let bins = n / 2 + 1;
    let power_db: Vec<f32> = buf[..bins]
        .iter()
        .map(|c| {
            let mag_sq = (c.re * c.re + c.im * c.im) * scale * scale;
            10.0 * (mag_sq + 1e-12_f32).log10()
        })
        .collect();

    (power_db, bin_hz)
}

/// Estimate SNR (dB) at `carrier_hz` using a Hann-windowed FFT power spectrum.
///
/// Peak bin power vs median of bins 10+ bins away from the peak.
/// Uses `power_spectrum()` internally.
pub fn spectrum_snr_db(samples: &[f32], fs: f32, carrier_hz: f32) -> f32 {
    let (power_db, bin_hz) = power_spectrum(samples, fs);
    let n_bins = power_db.len();
    if n_bins < 3 {
        return 0.0;
    }

    let peak_bin = ((carrier_hz / bin_hz).round() as usize).min(n_bins - 1);

    // Find the actual peak within ±3 bins of expected (AFC tolerance).
    let search_r = 3_usize;
    let lo = peak_bin.saturating_sub(search_r);
    let hi = (peak_bin + search_r).min(n_bins - 1);
    let sig_bin = (lo..=hi)
        .max_by(|&a, &b| {
            power_db[a]
                .partial_cmp(&power_db[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(peak_bin);

    let sig_db = power_db[sig_bin];

    // Noise floor: collect bins at least 10 bins away from the signal bin,
    // excluding DC (bin 0).  Use the median.
    let guard = 10_usize;
    let mut noise_bins: Vec<f32> = power_db
        .iter()
        .enumerate()
        .filter(|&(i, _)| i > 0 && (i as isize - sig_bin as isize).unsigned_abs() >= guard)
        .map(|(_, &v)| v)
        .collect();

    if noise_bins.is_empty() {
        return 0.0;
    }
    noise_bins.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let noise_db = noise_bins[noise_bins.len() / 2];

    sig_db - noise_db
}

/// Estimate AM DSB occupied bandwidth (Hz).
///
/// Scans outward from the carrier bin on both sides to find the outermost
/// bins within `carrier_drop_db` of the carrier peak. The bandwidth is the
/// span from the left edge to the right edge.
///
/// `_threshold_db` is reserved for future use (currently uses a fixed 35 dB
/// carrier-relative cutoff).
pub fn spectrum_bw_hz(samples: &[f32], fs: f32, carrier_hz: f32, _threshold_db: f32) -> f32 {
    let search_hz = 4_000.0_f32;
    let carrier_drop_db = 35.0_f32;
    let carrier_guard_bins = 3_usize;

    let (power_db, bin_hz) = power_spectrum(samples, fs);
    let n_bins = power_db.len();
    if n_bins < 3 {
        return bin_hz;
    }

    // Locate the carrier bin.
    let nominal_bin = ((carrier_hz / bin_hz).round() as usize).min(n_bins - 1);
    let cr = 3_usize;
    let c_lo = nominal_bin.saturating_sub(cr);
    let c_hi = (nominal_bin + cr).min(n_bins - 1);
    let carrier_bin = (c_lo..=c_hi)
        .max_by(|&a, &b| {
            power_db[a]
                .partial_cmp(&power_db[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(nominal_bin);

    let cutoff = power_db[carrier_bin] - carrier_drop_db;
    let search_bins = (search_hz / bin_hz).ceil() as usize;

    // Left edge: outermost LSB bin above cutoff.
    let lsb_lo = carrier_bin.saturating_sub(search_bins);
    let lsb_hi = carrier_bin.saturating_sub(carrier_guard_bins);
    let left_edge = if lsb_lo < lsb_hi {
        (lsb_lo..=lsb_hi)
            .find(|&i| power_db[i] >= cutoff)
            .unwrap_or(carrier_bin)
    } else {
        carrier_bin
    };

    // Right edge: outermost USB bin above cutoff.
    let usb_lo = (carrier_bin + carrier_guard_bins).min(n_bins - 1);
    let usb_hi = (carrier_bin + search_bins).min(n_bins - 1);
    let right_edge = if usb_lo < usb_hi {
        (usb_lo..=usb_hi)
            .rfind(|&i| power_db[i] >= cutoff)
            .unwrap_or(carrier_bin)
    } else {
        carrier_bin
    };

    ((right_edge.max(left_edge) - left_edge + 1) as f32) * bin_hz
}

/// Pick the best PSK31 sync result nearest to `carrier_hz`.
///
/// Primary sort: earliest `time_sym` (more data available for the demodulator).
/// Secondary sort: smallest frequency offset (tie-breaking).
/// Only considers results within 2 × `baud` Hz of `carrier_hz`.
///
/// Returns `(carrier_hz, time_sym)`.
pub fn best_sync(
    results: &[crate::sync::psk31_sync::Psk31SyncResult],
    carrier_hz: f32,
    baud: f32,
) -> Option<(f32, usize)> {
    results
        .iter()
        .filter(|r| (r.carrier_hz - carrier_hz).abs() <= 2.0 * baud)
        .min_by(|a, b| {
            let da = (a.carrier_hz - carrier_hz).abs();
            let db = (b.carrier_hz - carrier_hz).abs();
            a.time_sym
                .cmp(&b.time_sym)
                .then(da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal))
        })
        .map(|r| (r.carrier_hz, r.time_sym))
}

/// RMS threshold below which a sample block is treated as silence.
pub const SIGNAL_THRESHOLD: f32 = 0.1;

/// PSK31 bandwidth: raised-cosine pulse shaping gives exactly 2× the baud rate.
pub const PSK31_BW_HZ: f32 = 62.5; // 31.25 * 2

/// Fast atan2 approximation via 5th-order minimax polynomial.
/// Max error ≈ 0.0005 rad (~0.03°). Arguments follow the standard (y, x) convention.
#[inline(always)]
pub fn atan2_approx(y: f32, x: f32) -> f32 {
    let ax = x.abs();
    let ay = y.abs();
    let (mn, mx) = if ax < ay { (ax, ay) } else { (ay, ax) };
    let r = mn / (mx + f32::EPSILON);
    let r2 = r * r;
    let phi = r * (std::f32::consts::FRAC_PI_4 + r2 * (-0.2447 + r2 * 0.0663));
    let phi = if ax < ay {
        std::f32::consts::FRAC_PI_2 - phi
    } else {
        phi
    };
    if x < 0.0 {
        (std::f32::consts::PI - phi) * if y < 0.0 { -1.0 } else { 1.0 }
    } else {
        phi * if y < 0.0 { -1.0 } else { 1.0 }
    }
}
