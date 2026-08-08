// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::core::{Block, WorkReport};
use num_complex::Complex32 as C32;

#[derive(Debug, Clone)]
pub struct FirLowpass {
    taps: Vec<f32>,
    delay: Vec<f32>,
    idx: usize,
}

impl FirLowpass {
    /// Minimal LPF design (sinc + Hann).
    pub fn design(fs: f32, pass_hz: f32, trans_hz: f32) -> Self {
        let pass_hz = pass_hz.max(10.0);
        let trans_hz = trans_hz.max(pass_hz * 0.2);
        let ntaps = ((fs / trans_hz).ceil() as usize).max(31) | 1; // odd taps
        let mut taps = vec![0.0f32; ntaps];
        let fc = pass_hz / fs;
        let m0 = ntaps as isize / 2;
        for (n, tap) in taps.iter_mut().enumerate() {
            let m = n as isize - m0;
            let sinc = if m == 0 {
                2.0 * fc
            } else {
                let x = core::f32::consts::PI * m as f32;
                (2.0 * fc) * (2.0 * core::f32::consts::PI * fc * m as f32).sin() / x
            };
            let w =
                0.5 - 0.5 * (2.0 * core::f32::consts::PI * n as f32 / (ntaps as f32 - 1.0)).cos();
            *tap = sinc * w;
        }
        let s: f32 = taps.iter().sum();
        for t in &mut taps {
            *t /= s;
        }
        Self {
            taps,
            delay: vec![0.0; ntaps],
            idx: 0,
        }
    }

    #[inline]
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let n = input.len().min(output.len());
        for i in 0..n {
            self.delay[self.idx] = input[i];
            output[i] = self.dot();
            self.idx = (self.idx + 1) % self.delay.len();
        }
    }

    #[inline(always)]
    fn dot(&self) -> f32 {
        let len = self.taps.len();
        let d = &self.delay;
        let mut acc = 0.0f32;
        for (t_idx, &tap) in self.taps.iter().enumerate() {
            let d_idx = (self.idx + len - 1 - t_idx) % len;
            acc += d[d_idx] * tap;
        }
        acc
    }
}

// ── Kaiser-windowed low-pass tap design ──────────────────────────────────────

/// Kaiser window shape parameter `β` for a target stop-band attenuation `a_db`
/// (Kaiser's empirical formula). Below 21 dB the window degenerates to
/// rectangular (`β = 0`).
fn kaiser_beta(a_db: f32) -> f32 {
    if a_db > 50.0 {
        0.1102 * (a_db - 8.7)
    } else if a_db >= 21.0 {
        0.5842 * (a_db - 21.0).powf(0.4) + 0.07886 * (a_db - 21.0)
    } else {
        0.0
    }
}

/// Modified Bessel function of the first kind, order 0: `Σ ((x/2)^k / k!)²`.
/// The series converges in a handful of terms for the `β ≲ 10` used here.
fn bessel_i0(x: f32) -> f32 {
    let half = 0.5 * x;
    let mut term = 1.0f32;
    let mut sum = 1.0f32;
    for k in 1..=40u32 {
        term *= half / k as f32;
        let t = term * term;
        sum += t;
        if t < 1e-12 * sum {
            break;
        }
    }
    sum
}

/// Designs `num_taps` linear-phase low-pass FIR coefficients by the
/// Kaiser-windowed-sinc method, normalized to unit DC gain.
///
/// `cutoff_norm` is the −6 dB cutoff as a fraction of the sample rate
/// (`0 < cutoff_norm < 0.5`); applied to a **complex** baseband stream the
/// filter passes `[−cutoff_norm·fs, +cutoff_norm·fs]`. `stopband_db` is the
/// Kaiser design target `A` (≥ 21 dB; below that the window is rectangular).
///
/// `num_taps` is forced odd and at least 3, so the design is Type I: symmetric
/// taps, exactly linear phase, and an **integer** group delay of
/// `(num_taps − 1)/2` samples. Length and attenuation together fix the
/// transition width — see [`kaiser_transition_norm`] / [`kaiser_num_taps`].
pub fn kaiser_lowpass_taps(num_taps: usize, cutoff_norm: f32, stopband_db: f32) -> Vec<f32> {
    let m = num_taps.max(3) | 1;
    let mid = (m / 2) as f32;
    let fc = cutoff_norm.clamp(1e-4, 0.499_9);
    let beta = kaiser_beta(stopband_db);
    let i0_beta = bessel_i0(beta);

    let mut taps = vec![0.0f32; m];
    for (n, tap) in taps.iter_mut().enumerate() {
        let d = n as f32 - mid;
        // 2·fc·sinc(2·fc·d) — the ideal brick-wall impulse response.
        let ideal = if d == 0.0 {
            2.0 * fc
        } else {
            (core::f32::consts::TAU * fc * d).sin() / (core::f32::consts::PI * d)
        };
        let r = d / mid;
        let w = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0_beta;
        *tap = ideal * w;
    }
    // Normalize to unit DC gain so the filter neither adds nor removes level.
    let s: f32 = taps.iter().sum();
    if s.abs() > f32::EPSILON {
        for t in &mut taps {
            *t /= s;
        }
    }
    taps
}

/// Approximate transition width (fraction of `fs`) a Kaiser design of
/// `num_taps` achieves at `stopband_db`: `Δf/fs ≈ (A − 8) / (14.36 · M)`. The
/// −6 dB cutoff sits at the centre of that transition, so the pass band ends at
/// `cutoff_norm − Δf/(2·fs)` and the stop band begins at `cutoff_norm + Δf/(2·fs)`.
pub fn kaiser_transition_norm(num_taps: usize, stopband_db: f32) -> f32 {
    let m = (num_taps.max(3) | 1) as f32;
    (stopband_db.max(21.0) - 8.0) / (14.36 * m)
}

/// The inverse of [`kaiser_transition_norm`]: the (odd) tap count needed to hit
/// `transition_norm` (fraction of `fs`) at `stopband_db`.
pub fn kaiser_num_taps(transition_norm: f32, stopband_db: f32) -> usize {
    let m = ((stopband_db.max(21.0) - 8.0) / (14.36 * transition_norm.max(1e-4))).ceil();
    (m.max(3.0) as usize) | 1
}

// ── Complex-sample low-pass FIR ──────────────────────────────────────────────

/// Linear-phase low-pass FIR over **complex** (IQ) samples with real taps — the
/// `C32` counterpart of [`FirLowpass`], in the same spirit as
/// [`AgcRmsIq`](crate::dsp::AgcRmsIq) beside `AgcRms`. Real symmetric taps mean
/// I and Q are filtered identically, so a complex baseband spectrum is shaped
/// symmetrically about DC with no differential I/Q phase.
///
/// Two ways to run it:
///
/// - **Streaming** — [`push`](Self::push) per sample, or the [`Block`] impl over
///   a slice. Output lags input by the group delay `(num_taps − 1)/2`.
/// - **Aligned block** — [`filter_aligned`](Self::filter_aligned) filters a whole
///   buffer in place, **same length and time-aligned**: the group delay is
///   compensated internally, so sample `i` out corresponds to sample `i` in.
///   This is what a transmitter post-pass wants — it leaves stream length and
///   symbol boundaries untouched, so a strided receiver's cursor is unaffected.
#[derive(Debug, Clone)]
pub struct FirLowpassIq {
    taps: Vec<f32>,
    delay: Vec<C32>,
    idx: usize,
}

impl FirLowpassIq {
    /// Designs a Kaiser-windowed low-pass (see [`kaiser_lowpass_taps`] for the
    /// parameter meanings).
    pub fn design(num_taps: usize, cutoff_norm: f32, stopband_db: f32) -> Self {
        Self::from_taps(kaiser_lowpass_taps(num_taps, cutoff_norm, stopband_db))
    }

    /// Wraps a caller-supplied tap vector (e.g. an externally designed spectral
    /// mask). Group delay is reported as `(len − 1)/2`, which is the true group
    /// delay only for symmetric (linear-phase) taps.
    pub fn from_taps(taps: Vec<f32>) -> Self {
        let mut taps = taps;
        if taps.is_empty() {
            taps.push(1.0);
        }
        let len = taps.len();
        Self {
            taps,
            delay: vec![C32::default(); len],
            idx: 0,
        }
    }

    pub fn taps(&self) -> &[f32] {
        &self.taps
    }

    pub fn num_taps(&self) -> usize {
        self.taps.len()
    }

    /// Group delay in samples, `(num_taps − 1)/2` (exact for the symmetric
    /// designs produced by [`design`](Self::design)).
    pub fn group_delay(&self) -> usize {
        (self.taps.len() - 1) / 2
    }

    /// Clears the delay line (zero initial state).
    pub fn reset(&mut self) {
        self.delay.fill(C32::default());
        self.idx = 0;
    }

    /// Pushes one sample and returns the filter output for it (delayed by
    /// [`group_delay`](Self::group_delay)).
    #[inline(always)]
    pub fn push(&mut self, s: C32) -> C32 {
        let len = self.taps.len();
        self.delay[self.idx] = s;
        let (mut re, mut im) = (0.0f32, 0.0f32);
        // Split the circular-buffer walk at the wrap point so the inner loops
        // need no modulo: taps[0] pairs with the newest sample (at `idx`).
        for (j, &t) in self.taps[..=self.idx].iter().enumerate() {
            let d = self.delay[self.idx - j];
            re = d.re.mul_add(t, re);
            im = d.im.mul_add(t, im);
        }
        for (k, &t) in self.taps[self.idx + 1..].iter().enumerate() {
            let d = self.delay[len - 1 - k];
            re = d.re.mul_add(t, re);
            im = d.im.mul_add(t, im);
        }
        self.idx = if self.idx + 1 == len { 0 } else { self.idx + 1 };
        C32::new(re, im)
    }

    /// Filters `io` in place, **same length and time-aligned**: the filter's
    /// group delay is compensated, so `io[i]` afterwards is the filtered value
    /// of the original `io[i]`, not of `io[i − group_delay]`.
    ///
    /// The delay line is reset first and the buffer is treated as zero outside
    /// its bounds, so the leading and trailing `group_delay` samples carry the
    /// filter's edge transient. For a transmitted burst that is the natural
    /// behaviour (the signal really does start and stop there).
    ///
    /// Because the length is unchanged, this composes with the fixed-`sps`
    /// stride a receiver uses: symbol boundaries do not move.
    pub fn filter_aligned(&mut self, io: &mut [C32]) {
        let d = self.group_delay();
        let n = io.len();
        self.reset();
        // Prime the delay line with the first `d` samples (their outputs belong
        // to negative time and are discarded), so the first emitted output is
        // the response to input `d` — i.e. aligned with input 0.
        for i in 0..d {
            let x = io.get(i).copied().unwrap_or_default();
            self.push(x);
        }
        for i in 0..n {
            // Read ahead by `d`, write behind at `i`: safe in place.
            let x = if i + d < n { io[i + d] } else { C32::default() };
            io[i] = self.push(x);
        }
    }
}

impl Block for FirLowpassIq {
    type In = C32;
    type Out = C32;

    /// Streaming (causal) filtering: output lags input by
    /// [`group_delay`](Self::group_delay). Use
    /// [`filter_aligned`](Self::filter_aligned) when the output must stay
    /// time-aligned with the input.
    fn process(&mut self, input: &[C32], output: &mut [C32]) -> WorkReport {
        let n = input.len().min(output.len());
        for i in 0..n {
            output[i] = self.push(input[i]);
        }
        WorkReport {
            in_read: n,
            out_written: n,
        }
    }
}

// ── Half-cosine matched filter ────────────────────────────────────────────────

/// Complex-input FIR matched filter for the PSK31 half-cosine pulse shape.
///
/// Taps are `hann[n] / sqrt(Σhann[n]²)` (unit-energy normalised), so the
/// filter's peak output equals the signal amplitude when the input is an
/// ideal noiseless pulse.
///
/// Design: feed every down-mixed complex sample into `push`; call `out()`
/// at the end of each symbol period (every `sps` samples) to get the
/// matched-filter symbol estimate.  The filter has `sps` taps so its group
/// delay is `(sps−1)/2` samples.  Since we call `out()` at sample `sps−1`
/// (the last sample of the period), the output corresponds to the aligned
/// peak of the cross-correlation — no additional latency management needed.
///
/// I and Q channels share the same real taps with a split delay line,
/// matching the pattern used by `FirDecimator`.
#[derive(Debug, Clone)]
pub struct HalfCosineMf {
    taps: Vec<f32>,
    delay_re: Vec<f32>,
    delay_im: Vec<f32>,
    idx: usize,
}

impl HalfCosineMf {
    /// Construct a half-cosine MF for a given number of samples per symbol.
    pub fn new(sps: usize) -> Self {
        // Half-cosine pulse: hann[n] = 0.5 − 0.5·cos(π·n / (sps−1))
        let hann: Vec<f32> = if sps <= 1 {
            vec![1.0f32; sps.max(1)]
        } else {
            let denom = (sps - 1) as f32;
            (0..sps)
                .map(|i| 0.5 - 0.5 * (core::f32::consts::PI * i as f32 / denom).cos())
                .collect()
        };
        // Normalise to unit energy.
        let energy: f32 = hann.iter().map(|&h| h * h).sum();
        let scale = if energy > 0.0 {
            energy.sqrt().recip()
        } else {
            1.0
        };
        let taps: Vec<f32> = hann.iter().map(|&h| h * scale).collect();
        let len = taps.len();
        Self {
            taps,
            delay_re: vec![0.0f32; len],
            delay_im: vec![0.0f32; len],
            idx: 0,
        }
    }

    /// Push one complex sample and return the current filter output.
    #[inline(always)]
    pub fn push(&mut self, s: C32) -> C32 {
        let len = self.taps.len();
        self.delay_re[self.idx] = s.re;
        self.delay_im[self.idx] = s.im;
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for t_idx in 0..len {
            let d_idx = (self.idx + len - t_idx) % len;
            let w = self.taps[t_idx];
            re += self.delay_re[d_idx] * w;
            im += self.delay_im[d_idx] * w;
        }
        self.idx = (self.idx + 1) % len;
        C32::new(re, im)
    }

    pub fn reset(&mut self) {
        self.delay_re.fill(0.0);
        self.delay_im.fill(0.0);
        self.idx = 0;
    }
}
