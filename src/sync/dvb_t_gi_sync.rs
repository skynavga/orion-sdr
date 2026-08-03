// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/sync/dvb_t_gi_sync.rs
//
// Guard-interval (cyclic-prefix) acquisition for a preamble-less OFDM frame —
// the DVB-T way. A conformant DVB-T signal carries NO Schmidl & Cox preamble;
// the receiver finds the symbol boundary and fractional carrier-frequency
// offset directly from the cyclic prefix (the van de Beek / Sandell / Börjesson
// ML estimator, "ML estimation of time and frequency offset in OFDM systems",
// IEEE Trans. Signal Processing, 1997).
//
// Each OFDM symbol is `cp_len + n_fft` samples, where the first `cp_len`
// samples (the guard interval) are a copy of the symbol's last `cp_len`
// samples. So for a correct symbol-start offset `d`, the two windows
// `r[d .. d+cp_len]` and `r[d+n_fft .. d+n_fft+cp_len]` are identical up to
// noise and a phase ramp `exp(j·2π·ε·n_fft)` from a residual CFO `ε` (in
// subcarrier-spacing units). Define the correlation and energy terms
//
//     γ(d) = Σ_{k=0}^{cp_len-1} r[d+k] · conj(r[d+n_fft+k])
//     Φ(d) = ½ Σ_{k=0}^{cp_len-1} ( |r[d+k]|² + |r[d+n_fft+k]|² ).
//
// The ML timing estimate maximizes the log-likelihood metric
//
//     Λ(d) = |γ(d)| − ρ · Φ(d),
//
// where ρ = SNR/(SNR+1) = |correlation coefficient| weights the energy term by
// the signal reliability (ρ→1 at high SNR fully subtracts Φ; ρ→0 at low SNR
// leaves a pure correlation peak-pick). Ranking by |γ| alone — dropping ρ·Φ —
// is only the correct ML rule in the ρ→0 limit and spuriously rewards
// high-energy offsets; the full metric is used here. The fractional CFO comes
// from the angle at the winning offset:
//
//     ε̂ = −∠γ / (2π)        (cycles per n_fft samples)
//     cfo_hz = ε̂ · fs / n_fft = −∠γ · fs / (2π · n_fft).
//
// The estimate is unambiguous only within ±½ a subcarrier spacing (±fs/2n_fft);
// integer-CFO and frame/super-frame lock come from the scattered pilots and TPS
// downstream. This provides the timing + fractional-CFO acquisition an external
// IQ capture needs before any symbol can be demapped.
//
// DIVERGENCE FROM THE PUBLISHED ESTIMATOR. Van de Beek's Λ is derived over a
// SINGLE cyclic prefix. Here γ and Φ may optionally be accumulated coherently
// over up to `max_symbols` consecutive symbols at the same candidate offset
// (like gr-dtv's moving-average correlator), which sharpens the lock for a
// known-length batch with a stable channel. This is coherent, so a large
// residual CFO rotates successive symbols' γ and partially cancels the sum —
// keep `max_symbols` small (a few symbols) when a meaningful CFO may be present,
// or set it to 1 for the strict single-symbol estimator. Fractional-CFO
// estimation itself is unaffected (it reads the accumulated angle, which the
// per-symbol phase ramp shares).

use num_complex::Complex32 as C32;

/// Tuning for [`dvb_t_gi_sync_with`]: the ML energy-term weight and the coherent
/// accumulation bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GiSyncConfig {
    /// The van de Beek energy-term weight `ρ = SNR/(SNR+1) ∈ [0, 1]` in the
    /// timing metric `|γ| − ρ·Φ`. The default (`0.95`) suits the moderate-to-high
    /// SNR of a decodable DVB-T signal; lower it toward 0 for very low SNR (where
    /// the pure correlation magnitude is the better statistic).
    pub rho: f32,
    /// Maximum number of consecutive symbols whose CP correlation is accumulated
    /// coherently at each candidate offset. `1` is the strict single-symbol van
    /// de Beek estimator; a small value (the default, `4`) sharpens the lock for
    /// a batch while staying robust to residual CFO. Larger values risk coherent
    /// cancellation under CFO (see the module header).
    pub max_symbols: usize,
}

impl Default for GiSyncConfig {
    fn default() -> Self {
        Self {
            rho: 0.95,
            max_symbols: 4,
        }
    }
}

/// One guard-interval acquisition result: a candidate symbol-start offset with
/// its correlation strength and fractional-CFO estimate. Mirrors
/// [`crate::sync::OfdmSyncResult`]'s role for the preamble path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GiSyncResult {
    /// Sample offset of the first full OFDM symbol's start (its cyclic prefix).
    pub start_sample: usize,
    /// Fractional CFO estimate (Hz), unambiguous within ±`fs / (2·n_fft)`.
    pub cfo_hz: f32,
    /// Normalized correlation score in `[0, 1]`: `|γ(d)| / Φ(d)` at the winning
    /// offset, higher is a stronger cyclic-prefix match (1 for identical
    /// noiseless windows). Reported for quality/thresholding; the *selection*
    /// uses the ML metric `|γ| − ρ·Φ`, not this ratio.
    pub score: f32,
}

/// Searches `iq` for the best guard-interval-aligned symbol start in the offset
/// range `0..search_len`, using the cyclic prefix of a `(n_fft, cp_len)` OFDM
/// symbol and the [default](GiSyncConfig::default) tuning. Returns `None` if
/// `iq` is too short to hold a full symbol plus the search span.
///
/// `search_len` should span at least one full symbol period (`n_fft + cp_len`)
/// so the true peak is included; callers typically pass one symbol period to
/// lock onto the symbol grid. See [`dvb_t_gi_sync_with`] to tune the ML energy
/// weight and the coherent-accumulation bound.
pub fn dvb_t_gi_sync(
    iq: &[C32],
    n_fft: usize,
    cp_len: usize,
    fs: f32,
    search_len: usize,
) -> Option<GiSyncResult> {
    dvb_t_gi_sync_with(iq, n_fft, cp_len, fs, search_len, &GiSyncConfig::default())
}

/// Like [`dvb_t_gi_sync`], with an explicit [`GiSyncConfig`] (`ρ` weight and the
/// coherent-accumulation bound `max_symbols`). Selects the offset maximizing the
/// van de Beek ML timing metric `|γ(d)| − ρ·Φ(d)`.
pub fn dvb_t_gi_sync_with(
    iq: &[C32],
    n_fft: usize,
    cp_len: usize,
    fs: f32,
    search_len: usize,
    cfg: &GiSyncConfig,
) -> Option<GiSyncResult> {
    if cp_len == 0 || n_fft == 0 {
        return None;
    }
    // Need, for the largest tested offset d = search_len-1, the window
    // r[d + n_fft .. d + n_fft + cp_len].
    let need = search_len.saturating_sub(1) + n_fft + cp_len;
    if iq.len() < need || search_len == 0 {
        return None;
    }

    let period = n_fft + cp_len;
    let max_syms = cfg.max_symbols.max(1);
    let mut best_d = 0usize;
    let mut best_metric = f32::NEG_INFINITY;
    let mut best_gamma = C32::default();
    let mut best_phi = 0.0f32;
    for d in 0..search_len {
        // Accumulate γ and Φ over up to `max_syms` consecutive symbols whose
        // CP+tail windows fit within the buffer (coherent for γ).
        let mut gamma = C32::default();
        let mut phi = 0.0f32;
        let mut base = d;
        let mut used = 0usize;
        while used < max_syms && base + n_fft + cp_len <= iq.len() {
            for k in 0..cp_len {
                let a = iq[base + k];
                let b = iq[base + n_fft + k];
                gamma += a * b.conj();
                phi += a.norm_sqr() + b.norm_sqr();
            }
            base += period;
            used += 1;
        }
        phi *= 0.5; // Φ = ½ Σ(|a|² + |b|²)

        // van de Beek ML timing metric: |γ| − ρ·Φ.
        let metric = gamma.norm() - cfg.rho * phi;
        if metric > best_metric {
            best_metric = metric;
            best_d = d;
            best_gamma = gamma;
            best_phi = phi;
        }
    }

    // Reported score: |γ| / Φ at the winning offset, in [0, 1] (1 for identical
    // noiseless windows). Selection used the ML metric above, not this ratio.
    let score = if best_phi > 0.0 {
        (best_gamma.norm() / best_phi).min(1.0)
    } else {
        0.0
    };
    // Fractional CFO: negate the accumulated angle so a positive applied CFO
    // reads as a positive estimate (ε̂ = −∠γ / 2π).
    let cfo_hz = -best_gamma.im.atan2(best_gamma.re) * fs / (core::f32::consts::TAU * n_fft as f32);
    Some(GiSyncResult {
        start_sample: best_d,
        cfo_hz,
        score,
    })
}

/// Refines a coarse offset to the best cyclic-prefix ML metric in a small window
/// `±radius` around `coarse`, using the [default](GiSyncConfig::default) tuning.
/// A convenience for a receiver that already has an approximate symbol boundary
/// (e.g. from a prior frame) and wants a cheap local re-lock rather than a full-
/// period search.
pub fn dvb_t_gi_refine(
    iq: &[C32],
    n_fft: usize,
    cp_len: usize,
    fs: f32,
    coarse: usize,
    radius: usize,
) -> Option<GiSyncResult> {
    dvb_t_gi_refine_with(
        iq,
        n_fft,
        cp_len,
        fs,
        coarse,
        radius,
        &GiSyncConfig::default(),
    )
}

/// Like [`dvb_t_gi_refine`], with an explicit [`GiSyncConfig`].
pub fn dvb_t_gi_refine_with(
    iq: &[C32],
    n_fft: usize,
    cp_len: usize,
    fs: f32,
    coarse: usize,
    radius: usize,
    cfg: &GiSyncConfig,
) -> Option<GiSyncResult> {
    let start = coarse.saturating_sub(radius);
    let span = 2 * radius + 1;
    let sub = iq.get(start..)?;
    let mut r = dvb_t_gi_sync_with(sub, n_fft, cp_len, fs, span.min(sub.len()), cfg)?;
    r.start_sample += start;
    Some(r)
}

// ── Integer-CFO estimation (continual-pilot spectral correlation) ────────────
//
// The guard-interval estimator above resolves the CFO only within ±½ a
// subcarrier spacing; a real front end can be off by whole subcarriers, which
// slides the entire spectrum by that integer `k`. DVB-T's 45 continual pilots sit
// at FIXED carrier positions on every symbol (§4.5.4) and are boosted (16/9
// power), so they anchor the integer offset: after fractional correction and
// symbol alignment, FFT one symbol and, for each trial shift `k`, sum the energy
// landing at the continual-pilot bins shifted by `k`. The `k` that maximizes that
// pilot-position energy is the integer CFO — the boosted pilots dominate any
// coincidental data energy at the same 45 positions. This is the DVB-T-native
// counterpart to the OFDM preamble path's training-symbol integer-CFO recovery
// (`sync::ofdm_sync`), which a preamble-less frame cannot use.

use crate::waveform::dvb_t::continual_pilot_bins;

/// One integer-CFO estimate: the offset in whole subcarrier spacings and a
/// confidence ratio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntegerCfoResult {
    /// Integer CFO in subcarrier spacings (the spectrum is shifted by this many
    /// bins). `cfo_hz = bins · fs / n_fft`; correct by rotating the time domain by
    /// `−cfo_hz`.
    pub bins: i32,
    /// Confidence: the winning shift's continual-pilot energy divided by the mean
    /// over all trial shifts (`1.0` = no discrimination; higher is a clearer
    /// lock). Useful for thresholding whether an integer offset is present.
    pub confidence: f32,
}

/// Estimates the integer carrier-frequency offset of a DVB-T symbol from its
/// **frequency-domain** bins `freq` (one symbol's FFT output, `n_fft` long, in
/// rustfft bin order — i.e. already CP-removed, FFT'd, and ideally fractional-CFO
/// corrected). Searches trial shifts `k ∈ [−max_bins, max_bins]` and returns the
/// one maximizing the energy at the 45 continual-pilot bins shifted by `k`.
///
/// Returns `None` if `freq.len() < n_fft` or `max_bins == 0`. `max_bins` bounds
/// the search; the continual pilots span the active band, so shifts larger than
/// the guard-band margin slide pilots out of the band and lose discrimination —
/// a few tens of subcarriers is a generous front-end range.
pub fn dvb_t_integer_cfo(freq: &[C32], n_fft: usize, max_bins: i32) -> Option<IntegerCfoResult> {
    if freq.len() < n_fft || n_fft == 0 || max_bins <= 0 {
        return None;
    }
    let pilot_bins = continual_pilot_bins();

    // Pilot-position energy for a trial shift `k` (bins wrap mod n_fft).
    let energy_at = |k: i32| -> f32 {
        pilot_bins
            .iter()
            .map(|&b| {
                let idx = (b as i32 + k).rem_euclid(n_fft as i32) as usize;
                freq[idx].norm_sqr()
            })
            .sum()
    };

    let mut best_k = 0i32;
    let mut best_energy = f32::NEG_INFINITY;
    let mut sum_energy = 0.0f32;
    let mut count = 0u32;
    for k in -max_bins..=max_bins {
        let e = energy_at(k);
        sum_energy += e;
        count += 1;
        if e > best_energy {
            best_energy = e;
            best_k = k;
        }
    }

    let mean = sum_energy / count as f32;
    let confidence = if mean > 0.0 { best_energy / mean } else { 0.0 };
    Some(IntegerCfoResult {
        bins: best_k,
        confidence,
    })
}
