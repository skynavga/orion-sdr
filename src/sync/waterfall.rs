// src/sync/waterfall.rs
//
// Symbol-rate magnitude spectrogram ("waterfall") for FT8/FT4 sync.
//
// The waterfall is a 2D grid of per-tone log-magnitude values, one row per
// symbol slot and one column per frequency bin.  It supports oversampled
// time and frequency grids via `time_osr` and `freq_osr` parameters, which
// allow sub-symbol and sub-bin offset searching in the Costas scorer.
//
// Algorithm mirrors ft8_lib make_waterfall / compute_wf:
//   For each (time_sub, freq_sub) pair the IQ stream is segmented into
//   symbol-length windows starting at sample offset `time_sub * (samples_per_sym
//   / time_osr)`.  Each window is dot-product correlated against each tone phasor
//   (Goertzel), producing energy values.  Energies are stored as `f32` log-power.

use num_complex::Complex32 as C32;

/// Magnitude waterfall over a single (time_sub, freq_sub) slice of the IQ buffer.
///
/// Fields:
/// - `mag[sym][tone]` — log-power at symbol `sym`, tone bin `tone`
/// - `num_syms` — number of symbol rows
/// - `num_tones` — number of tone columns (bins)
pub struct Waterfall {
    /// Flattened `num_syms × num_tones` log-power magnitudes.
    pub mag: Vec<f32>,
    pub num_syms: usize,
    pub num_tones: usize,
}

impl Waterfall {
    /// Allocate a zeroed waterfall grid.
    pub fn new(num_syms: usize, num_tones: usize) -> Self {
        Self {
            mag: vec![0.0f32; num_syms * num_tones],
            num_syms,
            num_tones,
        }
    }

    /// Access `mag[sym][tone]`.
    #[inline]
    pub fn get(&self, sym: usize, tone: usize) -> f32 {
        self.mag[sym * self.num_tones + tone]
    }

    /// Set `mag[sym][tone]`.
    #[inline]
    pub fn set(&mut self, sym: usize, tone: usize, val: f32) {
        self.mag[sym * self.num_tones + tone] = val;
    }
}

/// Compute a waterfall for one (time_sub, freq_sub) offset.
///
/// Parameters:
/// - `iq`              — IQ input buffer (arbitrary length)
/// - `fs`              — sample rate (Hz)
/// - `base_hz`         — frequency of bin 0
/// - `tone_spacing_hz` — spacing between adjacent bins
/// - `samples_per_sym` — samples in one symbol (integer; no oversampling here)
/// - `num_syms`        — number of symbol rows to fill
/// - `num_tones`       — number of frequency bins (columns)
/// - `time_offset`     — additional sample offset into `iq` (for time_sub)
///
/// Returns a `Waterfall` with `num_syms × num_tones` log-power entries.
/// Symbol `s` uses IQ samples `[time_offset + s*samples_per_sym .. +samples_per_sym)`.
/// Missing samples are treated as zero energy (−120 dB).
pub fn compute_waterfall(
    iq: &[C32],
    fs: f32,
    base_hz: f32,
    tone_spacing_hz: f32,
    samples_per_sym: usize,
    num_syms: usize,
    num_tones: usize,
    time_offset: usize,
) -> Waterfall {
    // Pre-compute per-sample step phasors for each bin.
    let steps: Vec<C32> = (0..num_tones)
        .map(|k| {
            let f = base_hz + (k as f32) * tone_spacing_hz;
            let phi = -core::f32::consts::TAU * f / fs;
            let (s, c) = phi.sin_cos();
            C32::new(c, s)
        })
        .collect();

    let mut wf = Waterfall::new(num_syms, num_tones);

    for sym in 0..num_syms {
        let start = time_offset + sym * samples_per_sym;
        let end = start + samples_per_sym;

        if start >= iq.len() {
            // Beyond end of buffer — leave as 0.0 (−∞ in log space; safe for max-log)
            continue;
        }

        let slice = if end <= iq.len() {
            &iq[start..end]
        } else {
            // Partial symbol at the end — use what we have
            &iq[start..]
        };

        for (k, &w) in steps.iter().enumerate() {
            let e = goertzel_energy(slice, w);
            // Store as log-power (natural log of energy; add tiny floor to avoid -inf)
            wf.set(sym, k, (e + 1e-12).ln());
        }
    }

    wf
}

/// Compute the squared magnitude of a dot-product correlator (Goertzel-style).
///
/// Returns |Σ iq[i] · w^i|² where w = e^{-j2πf/fs}.
/// Uses the same 4-sample unrolled inner loop as the Phase 1 demodulator.
#[inline]
fn goertzel_energy(slice: &[C32], w: C32) -> f32 {
    let n = slice.len();
    let mut acc = C32::new(0.0, 0.0);
    let mut phasor = C32::new(1.0, 0.0);

    let mut i = 0;
    let nn = n & !3;
    while i < nn {
        acc += slice[i] * phasor;
        let p1 = mul(phasor, w);
        acc += slice[i + 1] * p1;
        let p2 = mul(p1, w);
        acc += slice[i + 2] * p2;
        let p3 = mul(p2, w);
        acc += slice[i + 3] * p3;
        phasor = mul(p3, w);
        i += 4;
    }
    while i < n {
        acc += slice[i] * phasor;
        phasor = mul(phasor, w);
        i += 1;
    }

    acc.norm_sqr()
}

/// Complex multiply with fused multiply-add.
#[inline(always)]
fn mul(a: C32, b: C32) -> C32 {
    C32::new(
        a.re.mul_add(b.re, -a.im * b.im),
        a.im.mul_add(b.re,  a.re * b.im),
    )
}
