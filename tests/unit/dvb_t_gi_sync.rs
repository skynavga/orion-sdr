// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// Guard-interval (cyclic-prefix) acquisition tests: a preamble-less OFDM signal
// is acquired from its cyclic prefix alone (timing + fractional CFO), the DVB-T
// way — no Schmidl & Cox preamble.

use num_complex::Complex32 as C32;
use orion_sdr::core::Block;
use orion_sdr::dsp::Rotator;
use orion_sdr::multicarrier::{CyclicPrefixInsert, IfftBlock};
use orion_sdr::sync::{GiSyncConfig, dvb_t_gi_refine, dvb_t_gi_sync, dvb_t_gi_sync_with};

/// Builds `n_syms` back-to-back OFDM symbols (random QPSK-ish data on every bin),
/// each `n_fft`-IFFT + `cp_len` cyclic prefix, with `lead` zero samples in front.
fn synth_ofdm(n_fft: usize, cp_len: usize, n_syms: usize, lead: usize) -> Vec<C32> {
    let mut rng = 0x9E37_79B9u32;
    let mut next = || {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let re = if (rng >> 16) & 1 == 0 { 1.0 } else { -1.0 };
        let im = if (rng >> 17) & 1 == 0 { 1.0 } else { -1.0 };
        C32::new(re, im)
    };
    let mut out = vec![C32::default(); lead];
    let mut ifft = IfftBlock::new(n_fft);
    let mut cp = CyclicPrefixInsert::new(n_fft, cp_len);
    for _ in 0..n_syms {
        let freq: Vec<C32> = (0..n_fft).map(|_| next()).collect();
        let mut time = vec![C32::default(); n_fft];
        ifft.process(&freq, &mut time);
        let mut with_cp = vec![C32::default(); n_fft + cp_len];
        cp.process(&time, &mut with_cp);
        out.extend_from_slice(&with_cp);
    }
    out
}

const N_FFT: usize = 256; // small stand-in FFT (fast); the estimator is size-agnostic
const CP_LEN: usize = 32;
const FS: f32 = 1_000_000.0;

#[test]
fn gi_sync_locks_symbol_boundary_noiseless() {
    let lead = 40;
    let iq = synth_ofdm(N_FFT, CP_LEN, 4, lead);
    let period = N_FFT + CP_LEN;
    let r = dvb_t_gi_sync(&iq, N_FFT, CP_LEN, FS, period).expect("acquire");
    // The lock is on the symbol grid: start ≡ lead (mod period). The estimator
    // may lock to any symbol boundary within the search window; assert the phase.
    assert_eq!(
        r.start_sample % period,
        lead % period,
        "found start {} not on the symbol grid (lead={lead})",
        r.start_sample
    );
    assert!(
        r.score > 0.9,
        "noiseless CP correlation should be near 1: {}",
        r.score
    );
    assert!(r.cfo_hz.abs() < 1.0, "no CFO applied: {}", r.cfo_hz);
}

#[test]
fn gi_sync_estimates_fractional_cfo() {
    let lead = 0;
    let mut iq = synth_ofdm(N_FFT, CP_LEN, 4, lead);
    // Apply a fractional CFO well within ±fs/(2·n_fft) = ±1953 Hz.
    let subcarrier = FS / N_FFT as f32;
    let cfo = 0.2 * subcarrier; // ~781 Hz
    let mut rot = Rotator::new(cfo, FS);
    let mut with_cfo = vec![C32::default(); iq.len()];
    rot.rotate_block(&iq, &mut with_cfo);
    iq = with_cfo;

    let period = N_FFT + CP_LEN;
    let r = dvb_t_gi_sync(&iq, N_FFT, CP_LEN, FS, period).expect("acquire");
    assert!(r.score > 0.85, "score under CFO: {}", r.score);
    // Recovered CFO within ~5% of the subcarrier spacing.
    assert!(
        (r.cfo_hz - cfo).abs() < 0.05 * subcarrier,
        "CFO estimate {} vs applied {}",
        r.cfo_hz,
        cfo
    );
}

#[test]
fn gi_sync_survives_awgn() {
    let lead = 17;
    let iq = synth_ofdm(N_FFT, CP_LEN, 8, lead);
    // Scale the AWGN to a realistic ~6 dB acquisition SNR relative to the actual
    // signal power (the IFFT of unit-magnitude bins spreads energy, so per-sample
    // power is small — the noise must be scaled from it, not a fixed amplitude).
    let sig_power: f32 = iq.iter().map(|c| c.norm_sqr()).sum::<f32>()
        / iq.iter().filter(|c| c.norm_sqr() > 0.0).count() as f32;
    let snr_db = 6.0f32;
    let noise_power = sig_power / 10f32.powf(snr_db / 10.0);
    let sigma = (noise_power / 2.0).sqrt(); // per real/imag component

    let mut rng = 0xABCD_1234u64;
    let mut gauss = || {
        // Sum-of-uniforms ≈ Gaussian (variance 12·(1/12) = 1 for 12 terms).
        let mut s = 0.0f32;
        for _ in 0..12 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            s += (rng as f32 / u64::MAX as f32) - 0.5;
        }
        s * sigma
    };
    let noisy: Vec<C32> = iq.iter().map(|&c| c + C32::new(gauss(), gauss())).collect();
    let period = N_FFT + CP_LEN;
    // The ML metric (|γ| − ρ·Φ) with coherent accumulation locks the boundary at
    // this SNR.
    let r = dvb_t_gi_sync(&noisy, N_FFT, CP_LEN, FS, period).expect("acquire");
    assert_eq!(
        r.start_sample % period,
        lead % period,
        "boundary under AWGN"
    );
}

#[test]
fn gi_refine_locks_near_coarse() {
    let lead = 100;
    let iq = synth_ofdm(N_FFT, CP_LEN, 4, lead);
    // A coarse estimate a few samples off the true boundary.
    let r = dvb_t_gi_refine(&iq, N_FFT, CP_LEN, FS, lead + 3, 8).expect("refine");
    assert_eq!(r.start_sample, lead, "refine snaps to the true boundary");
    assert!(r.score > 0.9);
}

#[test]
fn gi_sync_single_symbol_van_de_beek() {
    // max_symbols = 1 is the strict single-symbol estimator (no accumulation); it
    // still locks a clean signal.
    let lead = 23;
    let iq = synth_ofdm(N_FFT, CP_LEN, 3, lead);
    let cfg = GiSyncConfig {
        rho: 0.9,
        max_symbols: 1,
        ..GiSyncConfig::default()
    };
    let period = N_FFT + CP_LEN;
    let r = dvb_t_gi_sync_with(&iq, N_FFT, CP_LEN, FS, period, &cfg).expect("acquire");
    assert_eq!(r.start_sample % period, lead % period);
    assert!(r.score > 0.9, "clean single-symbol score: {}", r.score);
}

#[test]
fn gi_sync_config_default_is_documented() {
    // The default tuning: ρ = 0.95 (high-SNR energy weight), 4-symbol coherent
    // accumulation, and a 0.5 score-ratio unwrapping guard (see
    // `gi_sync_unwraps_a_peak_that_landed_below_the_period`).
    let d = GiSyncConfig::default();
    assert_eq!(d.rho, 0.95);
    assert_eq!(d.max_symbols, 4);
    assert_eq!(d.origin_score_ratio, 0.5);
}

#[test]
fn gi_sync_does_not_unwrap_a_genuine_lead_in() {
    // The guard against over-eager unwrapping. A lead-in that ends just below a
    // period boundary makes the peak *look* wrapped — it sits within cp_len/2 of
    // the boundary, exactly like a taper-biased peak does. What separates them is
    // the boundary's own single-symbol correlation: here offset 0 is silence and
    // does not correlate, so the frame does not start there and the true lead
    // must survive.
    //
    // The single-symbol check is what makes this work. An accumulated score would
    // pass: with `max_symbols = 4`, offset 0's second, third and fourth symbols
    // are only 5 samples off alignment and correlate strongly, masking that the
    // first one is silence.
    let period = N_FFT + CP_LEN;
    for lead in [200usize, period - 5, period - CP_LEN / 4] {
        let iq = synth_ofdm(N_FFT, CP_LEN, 6, lead);
        let r = dvb_t_gi_sync(&iq, N_FFT, CP_LEN, FS, period).expect("acquire");
        assert_eq!(
            r.start_sample, lead,
            "lead={lead}: reported {} should track the true start, not collapse to 0",
            r.start_sample
        );
    }
}

#[test]
fn gi_sync_unwrapping_is_opt_out() {
    // `origin_score_ratio = 0.0` restores the plain argmax, so a caller that
    // wants the raw van de Beek estimate can have it.
    let period = N_FFT + CP_LEN;
    let iq = synth_ofdm(N_FFT, CP_LEN, 4, 40);
    let off = GiSyncConfig {
        origin_score_ratio: 0.0,
        ..GiSyncConfig::default()
    };
    let a = dvb_t_gi_sync_with(&iq, N_FFT, CP_LEN, FS, period, &off).expect("acquire");
    let b = dvb_t_gi_sync(&iq, N_FFT, CP_LEN, FS, period).expect("acquire");
    // No wrap in this buffer, so both agree — the guard only ever fires on a peak
    // sitting just below a boundary.
    assert_eq!(a.start_sample, 40);
    assert_eq!(b.start_sample, 40);
}

#[test]
fn gi_sync_too_short_returns_none() {
    let iq = vec![C32::new(1.0, 0.0); N_FFT]; // shorter than a full symbol
    assert!(dvb_t_gi_sync(&iq, N_FFT, CP_LEN, FS, N_FFT + CP_LEN).is_none());
}
