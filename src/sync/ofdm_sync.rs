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
use crate::multicarrier::SymbolFft;
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
/// and RX side without requiring shared external state.
///
/// **`cfg.gain` is applied to the result**, exactly as [`OfdmMod`] applies it
/// to every data symbol. This is not cosmetic: the preamble and the payload
/// must share one amplitude scale or the frame is undecodable.
///
/// - The Schmidl & Cox timing metric normalizes against received energy, so a
///   preamble that is quiet relative to the payload collapses the score. At
///   `gain = 121` with an unscaled preamble the best score falls from 1.00 to
///   0.095 — below the streaming receiver's 0.5 acceptance threshold, so no
///   candidate is ever accepted and nothing decodes.
/// - `EqualizerMethod::TrainingSymbolHold` estimates the channel from the
///   training symbol. If the training symbol is unscaled while the payload is
///   not, the estimate omits the gain, the equalizer never divides it out, and
///   the demapper's LLRs are miscalibrated by that factor.
///
/// `cfg`'s other fields remain unused here. In particular **`rf_hz` is still
/// not applied** — a caller using a nonzero `rf_hz` gets a baseband preamble
/// ahead of an upconverted body. Applying it correctly requires phase
/// continuity with the symbols that follow, which the per-block modulator
/// construction does not currently provide; modulate at `rf_hz = 0.0` and
/// upconvert the whole burst with one continuous
/// [`Rotator`](crate::dsp::Rotator) instead.
///
/// [`OfdmMod`]: crate::modulate::OfdmMod
pub fn generate_ofdm_preamble(preamble: &OfdmPreamble, cfg: &OfdmConfig) -> Vec<C32> {
    let n_fft = cfg.carrier_plan.n_fft();
    let occupied_half = cfg.carrier_plan.occupied_half_carriers();
    let n_data = cfg.carrier_plan.data_carriers().len();

    let base = band_limited_repeat_base(preamble.repeat_len, n_fft, occupied_half, n_data)
        .unwrap_or_else(|| {
            // No usable band-limited construction (see the helper). Fall back to
            // the wideband sequence rather than emitting nothing.
            pseudo_random_unit_sequence(preamble.repeat_len, 0x4F46_444D_5052_4531)
        });
    let mut out = Vec::with_capacity(preamble.total_len());
    for _ in 0..preamble.num_repeats {
        out.extend_from_slice(&base);
    }
    if let Some(training) = preamble.training_symbol {
        out.extend_from_slice(&generate_training_symbol_time_domain(
            training,
            occupied_half,
        ));
    }
    let g = cfg.gain;
    if g != 1.0 {
        for s in &mut out {
            s.re *= g;
            s.im *= g;
        }
    }
    out
}

/// Amplitude of the S&C repeats relative to a data symbol.
///
/// `ofdm_sync` ranks candidates by `score * (r / r_peak)` — the correlated
/// window's energy against the loudest window anywhere in the search range —
/// so a preamble at exactly data power is no longer the energy peak and its
/// score is scaled down by whatever the payload happens to reach. Measured on
/// a clean frame, parity with the data drops the score from 1.00 to 0.54,
/// grazing the receiver's 0.5 acceptance threshold.
///
/// A boost restores it: 1.5x already returns a perfect 1.00, and 2x is taken
/// for margin. Transmitting the preamble hot is ordinary practice — 802.11
/// boosts its short training field for the same reason — and 6 dB costs
/// almost nothing against the ~70 dB of out-of-band excess band-limiting
/// removes.
const SC_PREAMBLE_BOOST: f32 = 2.0;

/// One period of a band-limited Schmidl & Cox base segment, or `None` when the
/// geometry does not admit one.
///
/// Built in the **frequency domain**: loading only bins that are multiples of
/// `k = n_fft / repeat_len` makes the inverse transform repeat with period
/// `repeat_len` by construction, so the repetition S&C correlates on is exact
/// rather than approximate. Restricting those bins to the plan's occupied span
/// is what band-limits it.
///
/// Returns `None` unless `repeat_len` divides `n_fft` and at least one occupied
/// bin falls on a multiple of `k` — a sparse or tiny plan can leave nothing to
/// load.
///
/// Amplitude is matched to a data symbol's: an OFDM symbol loading `m` bins at
/// unit magnitude lands at RMS `sqrt(m) / n_fft`, so the segment is scaled to
/// the value `n_data` loaded bins would give. Equal preamble and payload power
/// is the usual arrangement, and it keeps the S&C metric well conditioned.
fn band_limited_repeat_base(
    repeat_len: usize,
    n_fft: usize,
    occupied_half: usize,
    n_data: usize,
) -> Option<Vec<C32>> {
    if repeat_len == 0 || n_fft == 0 || !n_fft.is_multiple_of(repeat_len) || occupied_half == 0 {
        return None;
    }
    let k = n_fft / repeat_len;

    // Signed carrier indices inside the occupied span that land on a multiple
    // of `k`. DC is skipped, as the carrier plans do.
    let loaded: Vec<usize> = (1..=occupied_half as i32)
        .flat_map(|i| [i, -i])
        .filter(|i| (i.unsigned_abs() as usize).is_multiple_of(k))
        .map(|i| {
            if i >= 0 {
                i as usize
            } else {
                n_fft - i.unsigned_abs() as usize
            }
        })
        .collect();
    if loaded.is_empty() {
        return None;
    }

    let values = pseudo_random_unit_sequence(loaded.len(), 0x4F46_444D_5052_4531);
    let mut freq = vec![C32::default(); n_fft];
    for (&bin, &v) in loaded.iter().zip(values.iter()) {
        freq[bin] = v;
    }

    let mut ifft = crate::multicarrier::IfftBlock::new(n_fft);
    let mut time = vec![C32::default(); n_fft];
    ifft.process(&freq, &mut time);
    time.truncate(repeat_len);

    // Scale to a data symbol's RMS.
    let rms = (time.iter().map(|c| c.norm_sqr()).sum::<f32>() / time.len() as f32).sqrt();
    if rms > 0.0 {
        let target = SC_PREAMBLE_BOOST * (n_data as f32).sqrt() / n_fft as f32;
        let scale = target / rms;
        for c in &mut time {
            c.re *= scale;
            c.im *= scale;
        }
    }
    Some(time)
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
fn generate_training_symbol_time_domain(
    training: TrainingSymbol,
    occupied_half: usize,
) -> Vec<C32> {
    use crate::multicarrier::{CyclicPrefixInsert, IfftBlock};

    // Transmit the known pattern only inside the occupied span. The pattern
    // itself is unchanged — the receiver still divides by the full-band
    // reference — so the estimate on an occupied bin is exactly `H`, with no
    // scale to divide back out.
    //
    // Band-limiting also amplitude-matches it for free: the symbol's RMS is
    // `sqrt(loaded bins) / n_fft`, so loading the occupied span instead of
    // every bin brings it to the same level as a data symbol. Its old
    // full-band excess was that difference, not a gain.
    //
    // Bins outside the span are never extracted as data, so the estimate there
    // going to zero is harmless — `EQUALIZER_FLOOR` guards the division.
    let mut freq = training_symbol_freq_pattern(training.n_fft);
    if occupied_half > 0 {
        for (bin, v) in freq.iter_mut().enumerate() {
            let idx = if bin <= training.n_fft / 2 {
                bin
            } else {
                training.n_fft - bin
            };
            if idx == 0 || idx > occupied_half {
                *v = C32::default();
            }
        }
    }
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

    // Rank by `score * (r / r_peak)`, but **report the raw score**.
    //
    // The energy ratio is a tie-break: it picks the offset that is maximally
    // in-window among a plateau of equally phase-coherent ones. It is not a
    // measure of whether a preamble is present, and folding it into the score
    // made acceptance depend on whatever else is loud in the search range —
    // a preamble at ordinary signal level scores 0.54 rather than 1.00 merely
    // because the payload matches it for energy, and any louder transient
    // (a corrupted burst, an adjacent signal) suppresses a perfectly good
    // candidate below the threshold entirely.
    //
    // Ordering by the product keeps the tie-break; thresholding on the raw
    // score keeps acceptance a question about phase coherence, which is what
    // Schmidl & Cox actually measures.
    let mut ranked: Vec<(f32, OfdmSyncResult)> = all
        .into_iter()
        .map(|(r, result)| (result.score * (r / r_peak), result))
        .collect();

    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut results: Vec<OfdmSyncResult> = ranked.into_iter().map(|(_, r)| r).collect();

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

/// Picks the **earliest** accepted candidate from [`ofdm_sync`]'s output — the
/// one a receiver draining a buffer front-to-back should decode next.
///
/// [`ofdm_sync`] ranks by *quality*, best first, which is what a caller
/// answering "is there a preamble here, and where is it best aligned?" wants.
/// A streaming receiver asks a different question: "which frame comes next?"
/// Taking the best-ranked candidate answers the wrong one, and the consequence
/// is silent data loss rather than a visible error — the receiver locks onto a
/// frame further down the buffer and drains every frame before it without
/// reporting anything.
///
/// Measured before this existed, on eight back-to-back frames at an excellent
/// link (EVM ~-42 dB, BER 0): sequence numbers `[6, 7]` came back out of
/// `0..=7`, with **zero** errors reported. Every preamble scored 1.000, so the
/// quality ranking was decided by noise among equals. At zero noise the sort is
/// stable and all eight arrive, which is why a clean-signal test cannot see it.
///
/// Selection is by earliest **cluster**, not earliest offset. The timing metric
/// forms a plateau (see [`ofdm_sync`]), so one preamble occurrence yields a run
/// of accepted offsets; `cluster_len` — pass the preamble's `total_len()` —
/// groups them, and the best-ranked offset *within the earliest cluster* wins.
/// Taking the earliest offset outright would systematically pick the leading
/// edge of the plateau and give away timing accuracy for nothing.
pub fn earliest_accepted(
    results: Vec<OfdmSyncResult>,
    score_threshold: f32,
    cluster_len: usize,
) -> Option<OfdmSyncResult> {
    let accepted: Vec<OfdmSyncResult> = results
        .into_iter()
        .filter(|r| r.score >= score_threshold)
        .collect();
    let earliest = accepted.iter().map(|r| r.start_sample).min()?;
    // `accepted` preserves `ofdm_sync`'s quality ranking, so the first entry
    // falling inside the earliest cluster is that occurrence's best offset.
    accepted
        .into_iter()
        .find(|r| r.start_sample - earliest < cluster_len.max(1))
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
    // Integer-CFO estimation uses the standard CP-boundary window (no back-off):
    // it correlates the training symbol against a known frequency-domain pattern
    // to detect a whole-subcarrier shift, independent of the data window.
    let mut symbol_fft = SymbolFft::new(n_fft, training.cp_len);
    let freq = match symbol_fft.demod_symbol(&corrected) {
        Some(f) => f,
        None => return 0,
    };

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
