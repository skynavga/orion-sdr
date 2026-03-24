// src/sync/psk31_sync.rs
//
// PSK31 carrier-detection sync.
//
// Reuses `compute_waterfall` from `waterfall.rs` to build a symbol-rate
// spectrogram.  PSK31 has no Costas synchronisation pattern — carrier
// detection uses per-bin energy persistence with a local-peak criterion.
//
// Works for both BPSK31 and QPSK31 (same carrier characteristics at 31.25 baud).

use num_complex::Complex32 as C32;
use crate::modulate::psk31::{psk31_sps, PSK31_BAUD};
use crate::sync::waterfall::compute_waterfall;
use crate::demodulate::psk31::Bpsk31Demod;
use crate::core::Block;

/// Result of a successful PSK31 sync operation.
pub struct Psk31SyncResult {
    /// Symbol offset in the input buffer where the carrier was detected.
    pub time_sym: usize,
    /// Waterfall frequency bin (carrier_hz ≈ base_hz + bin * PSK31_BAUD).
    pub freq_bin: usize,
    /// Estimated carrier frequency in Hz.
    pub carrier_hz: f32,
    /// Mean log-energy over the detected carrier run (higher = stronger signal).
    pub score: f32,
    /// Soft bits from `Bpsk31Demod` starting at `time_sym`.
    /// Positive = bit 1 (no phase change), negative = bit 0 (phase flip).
    pub soft_bits: Vec<f32>,
}

/// Scan an IQ buffer for PSK31 carriers and extract soft bits per candidate.
///
/// # Parameters
/// - `iq`               — IQ input at sample rate `fs`
/// - `fs`               — sample rate (Hz)
/// - `base_hz`          — lowest carrier frequency to search (Hz)
/// - `max_hz`           — highest carrier frequency to search (Hz)
/// - `min_carrier_syms` — minimum run length to qualify as a carrier (symbols; default 8)
/// - `peak_margin_db`   — minimum excess above noise floor in dB (default 6.0)
/// - `n_bits`           — number of soft bits to extract per candidate
/// - `max_cand`         — maximum number of candidates to return
///
/// # Returns
/// Up to `max_cand` results sorted by `score` descending.
pub fn psk31_sync(
    iq: &[C32],
    fs: f32,
    base_hz: f32,
    max_hz: f32,
    min_carrier_syms: usize,
    peak_margin_db: f32,
    n_bits: usize,
    max_cand: usize,
) -> Vec<Psk31SyncResult> {
    let sps = psk31_sps(fs);
    if sps == 0 || iq.is_empty() {
        return Vec::new();
    }

    let num_syms = iq.len() / sps;
    if num_syms == 0 {
        return Vec::new();
    }

    let freq_range = (max_hz - base_hz).max(0.0);
    let num_bins = (freq_range / PSK31_BAUD).ceil() as usize + 1;
    if num_bins == 0 {
        return Vec::new();
    }

    // Build the waterfall (one bin per PSK31_BAUD Hz).
    let wf = compute_waterfall(iq, fs, base_hz, PSK31_BAUD, sps, num_syms, num_bins, 0);

    // Convert dB threshold: natural-log units.
    // The waterfall stores ln(energy), so peak_margin_db dB corresponds to
    // peak_margin_db * ln(10) / 10 nepers.
    let ln_margin = peak_margin_db * std::f32::consts::LN_2 / 3.0;

    // Minimum run duration in symbols.
    let min_run = min_carrier_syms.max(1);

    let mut candidates: Vec<Psk31SyncResult> = Vec::new();

    for bin in 0..num_bins {
        // Compute the per-bin median log-energy across all symbol rows.
        let mut energies: Vec<f32> = (0..num_syms).map(|s| wf.get(s, bin)).collect();
        let median = median_f32(&mut energies);
        let threshold = median + ln_margin;

        // Scan through symbols looking for persistent energy peaks.
        let mut run_start: Option<usize> = None;
        let mut run_energy_sum = 0.0f32;
        let mut run_len = 0usize;

        for sym in 0..num_syms {
            let e = wf.get(sym, bin);

            // Spectral sharpness: local maximum in frequency.
            // Use ≥ to handle the case where a carrier falls between two bins
            // (both adjacent bins share energy equally).
            let e_left  = if bin > 0            { wf.get(sym, bin - 1) } else { f32::NEG_INFINITY };
            let e_right = if bin + 1 < num_bins { wf.get(sym, bin + 1) } else { f32::NEG_INFINITY };

            let is_peak = e > threshold && e >= e_left && e >= e_right;

            if is_peak {
                if run_start.is_none() {
                    run_start = Some(sym);
                    run_energy_sum = 0.0;
                    run_len = 0;
                }
                run_energy_sum += e;
                run_len += 1;
            } else {
                if let Some(start) = run_start.take() {
                    if run_len >= min_run {
                        record_candidate(
                            &mut candidates,
                            start,
                            bin,
                            base_hz,
                            run_energy_sum / run_len as f32,
                            iq,
                            fs,
                            n_bits,
                        );
                    }
                    run_len = 0;
                }
            }
        }

        // Flush a run that reaches end of buffer.
        if let Some(start) = run_start.take() {
            if run_len >= min_run {
                record_candidate(
                    &mut candidates,
                    start,
                    bin,
                    base_hz,
                    run_energy_sum / run_len as f32,
                    iq,
                    fs,
                    n_bits,
                );
            }
        }
    }

    // Sort by score descending, keep top N.
    candidates.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(max_cand);
    candidates
}

/// Build a `Psk31SyncResult` for a detected carrier run.
fn record_candidate(
    out: &mut Vec<Psk31SyncResult>,
    time_sym: usize,
    freq_bin: usize,
    base_hz: f32,
    score: f32,
    iq: &[C32],
    fs: f32,
    n_bits: usize,
) {
    let sps = psk31_sps(fs);
    let carrier_hz = base_hz + freq_bin as f32 * PSK31_BAUD;
    let start_sample = time_sym * sps;
    if start_sample >= iq.len() { return; }

    let slice = &iq[start_sample..];
    let mut demod = Bpsk31Demod::new(fs, carrier_hz, 1.0);
    let mut soft = vec![0.0f32; n_bits + 2];
    let wr = demod.process(slice, &mut soft);
    soft.truncate(wr.out_written.min(n_bits));

    out.push(Psk31SyncResult {
        time_sym,
        freq_bin,
        carrier_hz,
        score,
        soft_bits: soft,
    });
}

/// Compute the median of an `f32` slice (mutates the slice to sort it).
fn median_f32(v: &mut [f32]) -> f32 {
    if v.is_empty() { return 0.0; }
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[mid - 1] + v[mid]) * 0.5
    } else {
        v[mid]
    }
}
