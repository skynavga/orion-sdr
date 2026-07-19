// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/sync/ofdm_sync.rs
//
// Packet sync and fractional CFO/timing acquisition for OFDM, via a
// Schmidl & Cox-style repeated-segment preamble (generic, not tied to any
// standard's specific preamble design).
//
// Algorithm: a preamble of `num_repeats` identical length-`repeat_len`
// complex segments is transmitted before the OFDM data symbols. At a
// candidate start `d`, adjacent repeated segments are correlated:
//
//   P(d) = Σ_{i=0}^{repeat_len-1} conj(r[d+i]) · r[d+i+repeat_len]
//   R(d) = Σ_{i=0}^{repeat_len-1} |r[d+i+repeat_len]|²
//
// summed over all `num_repeats - 1` adjacent segment pairs. The normalized
// timing metric `M(d) = |P(d)|² / R(d)²` plateaus near the true preamble
// start; its peak gives coarse timing. The correlation phase at the peak
// gives the fractional CFO: `cfo_hz = angle(P) / (2π · repeat_len / fs)`,
// unambiguous only within ±½ the subcarrier spacing (±`fs / (2·repeat_len)`)
// — larger offsets alias. Integer-CFO recovery (Release F) extends this.

use crate::modulate::OfdmConfig;
use num_complex::Complex32 as C32;

/// Repeated-segment preamble parameters: `num_repeats` identical segments of
/// `repeat_len` samples each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OfdmPreamble {
    pub num_repeats: usize,
    pub repeat_len: usize,
}

impl OfdmPreamble {
    pub fn new(num_repeats: usize, repeat_len: usize) -> Self {
        Self {
            num_repeats,
            repeat_len,
        }
    }

    /// Total preamble length in samples.
    pub fn total_len(&self) -> usize {
        self.num_repeats * self.repeat_len
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
    /// Normalized timing-metric score in `[0, 1]`; higher is a better match.
    pub score: f32,
}

/// Generates a repeated-segment preamble: a deterministic, reproducible
/// pseudo-random unit-average-energy base sequence of `repeat_len` samples,
/// tiled `num_repeats` times.
///
/// The base sequence is generated from a fixed seed (not derived from
/// `cfg`), so the same `OfdmPreamble` always produces the same preamble on
/// both the TX and RX side without requiring shared external state. `cfg`
/// is accepted for signature symmetry with the rest of the OFDM API and to
/// allow a future release to derive the sequence from the carrier plan; it
/// is currently unused.
pub fn generate_ofdm_preamble(preamble: &OfdmPreamble, _cfg: &OfdmConfig) -> Vec<C32> {
    let base = pseudo_random_unit_sequence(preamble.repeat_len, 0x4F46_444D_5052_4531);
    let mut out = Vec::with_capacity(preamble.total_len());
    for _ in 0..preamble.num_repeats {
        out.extend_from_slice(&base);
    }
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
    results
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
