// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/sync/ofdm_sync.rs
//
// Packet sync and fractional/integer CFO plus timing acquisition for OFDM,
// via a Schmidl & Cox-style repeated-segment preamble (generic, not tied to
// any standard's specific preamble design), optionally followed by a
// dedicated training symbol for wide-range integer-CFO recovery.
//
// Fractional stage (Release E): a preamble of `num_repeats` identical
// length-`repeat_len` complex segments is transmitted before the OFDM data
// symbols. At a candidate start `d`, adjacent repeated segments are
// correlated:
//
//   P(d) = Σ_{i=0}^{repeat_len-1} conj(r[d+i]) · r[d+i+repeat_len]
//   R(d) = Σ_{i=0}^{repeat_len-1} |r[d+i+repeat_len]|²
//
// summed over all `num_repeats - 1` adjacent segment pairs. The normalized
// timing metric `M(d) = |P(d)|² / R(d)²` plateaus near the true preamble
// start; its peak gives coarse timing. The correlation phase at the peak
// gives the fractional CFO: `cfo_hz = angle(P) / (2π · repeat_len / fs)`,
// unambiguous only within ±½ the subcarrier spacing (±`fs / (2·repeat_len)`)
// — larger offsets alias.
//
// Integer stage (Release F): a dedicated training symbol — one full
// `n_fft`+CP OFDM symbol with a known value on every subcarrier bin —
// follows the S&C preamble. After the fractional CFO/timing found above is
// corrected, the training symbol is FFT'd and correlated against its known
// frequency-domain pattern across candidate integer bin shifts; the shift
// maximizing correlation is the integer CFO
// (`integer_cfo_bins · fs / n_fft`). The same training symbol is reused by
// Release G's channel estimator.

use crate::core::Block;
use crate::dsp::Rotator;
use crate::modulate::OfdmConfig;
use crate::multicarrier::{CyclicPrefixRemove, FftBlock};
use num_complex::Complex32 as C32;

/// Repeated-segment preamble parameters: `num_repeats` identical segments of
/// `repeat_len` samples each, optionally followed by a dedicated training
/// symbol for integer-CFO recovery (and, in a later release, channel
/// estimation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OfdmPreamble {
    pub num_repeats: usize,
    pub repeat_len: usize,
    /// Present once a caller opts into wide-range integer-CFO recovery
    /// (Release F). `None` preserves Release E's fractional-only behavior.
    pub training_symbol: Option<TrainingSymbol>,
}

impl OfdmPreamble {
    pub fn new(num_repeats: usize, repeat_len: usize) -> Self {
        Self {
            num_repeats,
            repeat_len,
            training_symbol: None,
        }
    }

    /// Opts into the integer-CFO training symbol, sized to `n_fft` +
    /// `cp_len` from the caller's `CarrierPlan`.
    pub fn with_training_symbol(mut self, n_fft: usize, cp_len: usize) -> Self {
        self.training_symbol = Some(TrainingSymbol { n_fft, cp_len });
        self
    }

    /// Total preamble length in samples, including the training symbol if
    /// present.
    pub fn total_len(&self) -> usize {
        self.num_repeats * self.repeat_len + self.training_symbol.map_or(0, |t| t.total_len())
    }
}

/// Dedicated training symbol used for integer-CFO recovery: one full
/// `n_fft`-point OFDM symbol (plus cyclic prefix) with a known value on
/// every subcarrier bin, maximizing discriminating structure for the
/// integer-bin-shift search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingSymbol {
    pub n_fft: usize,
    pub cp_len: usize,
}

impl TrainingSymbol {
    pub fn total_len(&self) -> usize {
        self.n_fft + self.cp_len
    }
}

/// One packet-sync candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OfdmSyncResult {
    /// Sample offset of the preamble's start.
    pub start_sample: usize,
    /// Fractional CFO estimate (Hz), unambiguous within ±½ the subcarrier
    /// spacing (±`fs / (2 · repeat_len)`); larger offsets alias.
    pub cfo_hz: f32,
    /// Integer CFO estimate, in whole subcarrier-spacing units. `0` unless
    /// `preamble.training_symbol` is present and the integer search ran.
    /// Total CFO is `cfo_hz + integer_cfo_bins as f32 * subcarrier_spacing`.
    pub integer_cfo_bins: i32,
    /// Normalized timing-metric score in `[0, 1]`; higher is a better match.
    pub score: f32,
}

/// Generates a repeated-segment preamble: a deterministic, reproducible
/// pseudo-random unit-average-energy base sequence of `repeat_len` samples,
/// tiled `num_repeats` times, followed by the training symbol (time-domain,
/// CP included) if `preamble.training_symbol` is present.
///
/// The repeat base sequence and the training symbol's frequency-domain
/// pattern are both generated from fixed seeds (not derived from `cfg`), so
/// the same `OfdmPreamble` always produces the same preamble on both the TX
/// and RX side without requiring shared external state. `cfg` is accepted
/// for signature symmetry with the rest of the OFDM API and to allow a
/// future release to derive the sequences from the carrier plan; it is
/// currently unused.
pub fn generate_ofdm_preamble(preamble: &OfdmPreamble, _cfg: &OfdmConfig) -> Vec<C32> {
    let base = pseudo_random_unit_sequence(preamble.repeat_len, 0x4F46_444D_5052_4531);
    let mut out = Vec::with_capacity(preamble.total_len());
    for _ in 0..preamble.num_repeats {
        out.extend_from_slice(&base);
    }
    if let Some(training) = preamble.training_symbol {
        out.extend_from_slice(&generate_training_symbol_time_domain(training));
    }
    out
}

/// The training symbol's known frequency-domain pattern: one unit-magnitude
/// pseudo-random value per FFT bin (natural rustfft bin order), from a fixed
/// seed distinct from the S&C repeat base sequence's.
///
/// `pub(crate)` so `demodulate::ofdm::OfdmEqualizer` can reuse the exact same
/// known pattern for `TrainingSymbolHold` channel estimation without
/// duplicating (and risking a mismatched) generator.
pub(crate) fn training_symbol_freq_pattern(n_fft: usize) -> Vec<C32> {
    pseudo_random_unit_sequence(n_fft, 0x4F46_444D_5452_4E31)
}

/// IFFTs the training symbol's known frequency-domain pattern to a
/// time-domain symbol and prepends its cyclic prefix, matching
/// `OfdmMod`'s TX chain (`IfftBlock` then `CyclicPrefixInsert`) so the
/// training symbol round-trips through the same channel as data symbols.
fn generate_training_symbol_time_domain(training: TrainingSymbol) -> Vec<C32> {
    use crate::multicarrier::{CyclicPrefixInsert, IfftBlock};

    let freq = training_symbol_freq_pattern(training.n_fft);
    let mut ifft = IfftBlock::new(training.n_fft);
    let mut time = vec![C32::default(); training.n_fft];
    ifft.process(&freq, &mut time);

    let mut cp_insert = CyclicPrefixInsert::new(training.n_fft, training.cp_len);
    let mut out = vec![C32::default(); training.total_len()];
    cp_insert.process(&time, &mut out);
    out
}

/// Deterministic pseudo-random complex sequence, unit average energy.
fn pseudo_random_unit_sequence(len: usize, seed: u64) -> Vec<C32> {
    let mut state = seed;
    let mut next_f32 = || -> f32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f32) / (u64::MAX as f32) - 0.5
    };

    let scale = std::f32::consts::FRAC_1_SQRT_2;
    (0..len)
        .map(|_| {
            let re = if next_f32() >= 0.0 { scale } else { -scale };
            let im = if next_f32() >= 0.0 { scale } else { -scale };
            C32::new(re, im)
        })
        .collect()
}

/// Searches `iq[search_start..search_end)` for a repeated-segment preamble
/// match, returning candidates sorted by descending score.
///
/// `search_end` is clamped so every candidate start has room for the full
/// preamble (`2 * repeat_len` samples for the correlation window, extended
/// across all `num_repeats` segments). Returns an empty `Vec` if the search
/// range is too short to hold a full preamble.
pub fn ofdm_sync(
    iq: &[C32],
    fs: f32,
    preamble: &OfdmPreamble,
    search_start: usize,
    search_end: usize,
) -> Vec<OfdmSyncResult> {
    let repeat_len = preamble.repeat_len;
    let num_repeats = preamble.num_repeats;
    if repeat_len == 0 || num_repeats < 2 || fs <= 0.0 {
        return Vec::new();
    }

    let preamble_len = preamble.total_len();
    let end = search_end.min(iq.len().saturating_sub(preamble_len));
    if search_start >= end {
        return Vec::new();
    }

    // The correlation-phase timing metric alone (`score`) forms a plateau,
    // not a sharp spike: a purely periodic preamble correlates against
    // itself at any offset that keeps the window fully inside the repeated
    // structure, not only at the true start. `R` — the correlated window's
    // own energy, summed over all `num_repeats - 1` segment pairs — breaks
    // the tie: it is maximized only where every correlated sample is real
    // preamble signal, which (for a preamble bounded by non-periodic
    // content on both sides) happens at exactly one offset, the true start.
    // Candidates are ranked by `score * (r / r_peak)`, so a result must be
    // both phase-coherent (S&C's actual acquisition criterion) and
    // maximally in-window to rank first.
    let mut all = Vec::with_capacity(end - search_start);
    let mut r_peak = 0.0f32;
    for d in search_start..end {
        let mut p = C32::default();
        let mut r = 0.0f32;

        for seg in 0..num_repeats - 1 {
            let a0 = d + seg * repeat_len;
            let b0 = a0 + repeat_len;
            let (seg_p, seg_r) = correlate_segment(iq, a0, b0, repeat_len);
            p += seg_p;
            r += seg_r;
        }

        if r <= 0.0 {
            continue;
        }
        r_peak = r_peak.max(r);

        let score = (p.norm_sqr() / (r * r)).clamp(0.0, 1.0);
        let cfo_hz = p.im.atan2(p.re) / (core::f32::consts::TAU * repeat_len as f32 / fs);

        all.push((
            r,
            OfdmSyncResult {
                start_sample: d,
                cfo_hz,
                integer_cfo_bins: 0,
                score,
            },
        ));
    }

    if all.is_empty() || r_peak <= 0.0 {
        return Vec::new();
    }

    let mut results: Vec<OfdmSyncResult> = all
        .into_iter()
        .map(|(r, mut result)| {
            result.score *= r / r_peak;
            result
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Integer-CFO search runs only on a small number of the top timing
    // candidates (bounding cost) using the dedicated training symbol
    // immediately following the S&C repeats, if the caller opted in.
    if let Some(training) = preamble.training_symbol {
        let top_n = results.len().min(5);
        for result in &mut results[..top_n] {
            let training_start = result.start_sample + repeat_len * num_repeats;
            result.integer_cfo_bins =
                estimate_integer_cfo_bins(iq, fs, training, training_start, result.cfo_hz);
        }
    }

    results
}

/// Estimates the integer CFO (whole subcarrier-spacing units) from the
/// dedicated training symbol at `training_start`: corrects the already-known
/// fractional CFO, strips the cyclic prefix, FFTs the result, and searches
/// candidate circular bin shifts for the one maximizing correlation against
/// the training symbol's known frequency-domain pattern.
///
/// Returns `0` if `iq` doesn't have room for the full training symbol at
/// `training_start`.
fn estimate_integer_cfo_bins(
    iq: &[C32],
    fs: f32,
    training: TrainingSymbol,
    training_start: usize,
    fractional_cfo_hz: f32,
) -> i32 {
    let total_len = training.total_len();
    if training_start + total_len > iq.len() {
        return 0;
    }

    let raw = &iq[training_start..training_start + total_len];
    let mut corrected = vec![C32::default(); total_len];
    let mut rot = Rotator::new(-fractional_cfo_hz, fs);
    rot.rotate_block(raw, &mut corrected);

    let n_fft = training.n_fft;
    let mut cp_remove = CyclicPrefixRemove::new(n_fft, training.cp_len);
    let mut time = vec![C32::default(); n_fft];
    if cp_remove.process(&corrected, &mut time).out_written != n_fft {
        return 0;
    }

    let mut fft = FftBlock::new(n_fft);
    let mut freq = vec![C32::default(); n_fft];
    if fft.process(&time, &mut freq).out_written != n_fft {
        return 0;
    }

    let known = training_symbol_freq_pattern(n_fft);

    // Search circular bin shifts within the signed carrier-index range
    // (natural rustfft bin order: shift k means the received spectrum is
    // rotated by k bins relative to the known pattern).
    let max_shift = (n_fft / 2) as i32;
    let mut best_shift = 0i32;
    let mut best_corr = -1.0f32;
    for shift in -max_shift..=max_shift {
        let mut corr = C32::default();
        for (bin, &k) in known.iter().enumerate() {
            let src_bin = (bin as i32 + shift).rem_euclid(n_fft as i32) as usize;
            corr += k.conj() * freq[src_bin];
        }
        let mag = corr.norm_sqr();
        if mag > best_corr {
            best_corr = mag;
            best_shift = shift;
        }
    }

    best_shift
}

/// Correlate two adjacent length-`len` segments starting at `a0`/`b0`:
/// `P = Σ conj(iq[a0+i]) · iq[b0+i]`, `R = Σ |iq[b0+i]|²`.
#[inline]
fn correlate_segment(iq: &[C32], a0: usize, b0: usize, len: usize) -> (C32, f32) {
    let mut p = C32::default();
    let mut r = 0.0f32;
    let mut i = 0;
    let nn = len & !3;
    while i < nn {
        p += iq[a0 + i].conj() * iq[b0 + i];
        r += iq[b0 + i].norm_sqr();
        p += iq[a0 + i + 1].conj() * iq[b0 + i + 1];
        r += iq[b0 + i + 1].norm_sqr();
        p += iq[a0 + i + 2].conj() * iq[b0 + i + 2];
        r += iq[b0 + i + 2].norm_sqr();
        p += iq[a0 + i + 3].conj() * iq[b0 + i + 3];
        r += iq[b0 + i + 3].norm_sqr();
        i += 4;
    }
    while i < len {
        p += iq[a0 + i].conj() * iq[b0 + i];
        r += iq[b0 + i].norm_sqr();
        i += 1;
    }
    (p, r)
}
