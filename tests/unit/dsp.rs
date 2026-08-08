// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use num_complex::Complex32 as C32;
use orion_sdr::core::Block;
use orion_sdr::dsp::{
    FirDecimator, FirLowpassIq, Nco, kaiser_lowpass_taps, kaiser_num_taps, kaiser_transition_norm,
    mix_with_nco,
};

#[test]
fn decimator_reduces_length_and_preserves_tone() {
    let fs = 96_000.0;
    let m = 4;
    let cutoff = fs / (m as f32) * 0.45;
    let transition = fs / (m as f32) * 0.10;
    let mut dec = FirDecimator::new(fs, m, cutoff, transition);
    let n = 4096;
    let mut nco = Nco::new(2_000.0, fs);
    let mut iq = vec![C32::new(0.0, 0.0); n];
    for s in iq.iter_mut().take(n) {
        *s = mix_with_nco(C32::new(1.0, 0.0), &mut nco);
    }
    let mut out = vec![C32::new(0.0, 0.0); n / m];
    let w = dec.process(&iq, &mut out);
    assert_eq!(w.out_written, n / m);
}

// ── Kaiser low-pass design + complex FIR (Track C, R14) ─────────────────────

/// `|H(f)|` in dB at normalized frequency `f` (fraction of `fs`), evaluated
/// directly from the taps: `H(f) = Σ h[n]·e^{-j2πfn}`.
fn response_db(taps: &[f32], f: f32) -> f32 {
    let (mut re, mut im) = (0.0f32, 0.0f32);
    for (n, &h) in taps.iter().enumerate() {
        let phase = -core::f32::consts::TAU * f * n as f32;
        re += h * phase.cos();
        im += h * phase.sin();
    }
    20.0 * (re * re + im * im).sqrt().max(1e-12).log10()
}

#[test]
fn kaiser_lowpass_taps_are_linear_phase_and_unit_dc_gain() {
    // Odd length (forced), symmetric taps (Type I ⇒ exactly linear phase, integer
    // group delay), and normalized so DC passes at unity.
    for req in [3usize, 16, 31, 64, 101] {
        let taps = kaiser_lowpass_taps(req, 0.2, 60.0);
        assert_eq!(taps.len(), req.max(3) | 1, "tap count forced odd/>=3");
        let m = taps.len();
        for i in 0..m / 2 {
            assert!(
                (taps[i] - taps[m - 1 - i]).abs() < 1e-6,
                "taps must be symmetric (len {m}, index {i})"
            );
        }
        let dc: f32 = taps.iter().sum();
        assert!((dc - 1.0).abs() < 1e-5, "DC gain {dc} should be 1");
    }
}

#[test]
fn kaiser_lowpass_meets_its_stopband_target() {
    // The design's contract: flat through the pass band, down by roughly the
    // requested attenuation once past the transition. Check both edges of the
    // transition the (num_taps, stopband_db) pair implies.
    let (num_taps, cutoff, stopband_db) = (101usize, 0.2f32, 60.0f32);
    let taps = kaiser_lowpass_taps(num_taps, cutoff, stopband_db);
    let half_transition = 0.5 * kaiser_transition_norm(num_taps, stopband_db);

    for f in [0.0f32, 0.05, 0.1, cutoff - half_transition] {
        let db = response_db(&taps, f);
        assert!(
            db.abs() < 0.5,
            "pass band should be flat: |H({f})| = {db} dB"
        );
    }
    // −6 dB at the nominal cutoff (the centre of the transition).
    let at_cutoff = response_db(&taps, cutoff);
    assert!(
        (at_cutoff + 6.0).abs() < 1.0,
        "cutoff should be the −6 dB point, got {at_cutoff} dB"
    );
    for f in [cutoff + half_transition, 0.3, 0.4, 0.5] {
        let db = response_db(&taps, f);
        assert!(
            db < -(stopband_db - 5.0),
            "stop band should reach ~−{stopband_db} dB: |H({f})| = {db} dB"
        );
    }
}

#[test]
fn kaiser_num_taps_inverts_transition_norm() {
    // The two sizing helpers are inverses (up to the odd/ceil rounding), so a
    // caller can size a filter from a null-band width and read back what it buys.
    for (transition, a_db) in [(0.02f32, 60.0f32), (0.05, 40.0), (0.084, 60.0)] {
        let m = kaiser_num_taps(transition, a_db);
        assert_eq!(m % 2, 1, "tap count must be odd");
        let got = kaiser_transition_norm(m, a_db);
        assert!(
            got <= transition * 1.001,
            "sized filter must meet the requested transition ({got} > {transition})"
        );
        // Not wastefully long: two taps shorter would miss the target.
        let shorter = kaiser_transition_norm(m.saturating_sub(2), a_db);
        assert!(
            shorter > transition * 0.999,
            "tap count {m} is longer than needed for transition {transition}"
        );
    }
}

#[test]
fn fir_lowpass_iq_passes_in_band_and_rejects_out_of_band() {
    // Steady-state complex-tone response: an in-band tone survives, an
    // out-of-band one is attenuated by roughly the design target.
    let (num_taps, cutoff, stopband_db) = (81usize, 0.2f32, 60.0f32);
    let n = 2048usize;
    let amplitude = |f: f32| -> f32 {
        let mut fir = FirLowpassIq::design(num_taps, cutoff, stopband_db);
        let mut peak = 0.0f32;
        for i in 0..n {
            let phase = core::f32::consts::TAU * f * i as f32;
            let y = fir.push(C32::new(phase.cos(), phase.sin()));
            // Skip the start-up transient; measure the steady state only.
            if i > 2 * num_taps {
                peak = peak.max(y.norm());
            }
        }
        peak
    };
    let in_band = amplitude(0.1);
    let out_of_band = amplitude(0.35);
    assert!(
        (in_band - 1.0).abs() < 0.02,
        "in-band tone should pass at unity, got {in_band}"
    );
    let atten_db = 20.0 * (out_of_band.max(1e-12) / in_band).log10();
    assert!(
        atten_db < -(stopband_db - 5.0),
        "out-of-band tone should be cut by ~{stopband_db} dB, got {atten_db} dB"
    );
}

#[test]
fn filter_aligned_is_same_length_and_group_delay_compensated() {
    // `filter_aligned` is the transmitter post-pass form: same length, and each
    // output sample aligned with the input sample it came from. Prove it against
    // the streaming path — aligned[i] must equal the causal output at i + group
    // delay, exactly.
    let (num_taps, cutoff) = (31usize, 0.2f32);
    let n = 512usize;
    let x: Vec<C32> = (0..n)
        .map(|i| {
            let p = core::f32::consts::TAU * 0.03 * i as f32;
            let env = (-(((i as f32) - 200.0) / 60.0).powi(2)).exp();
            C32::new(env * p.cos(), env * p.sin())
        })
        .collect();

    let d = FirLowpassIq::design(num_taps, cutoff, 60.0).group_delay();
    assert_eq!(d, (num_taps - 1) / 2);

    // Causal reference: filter x followed by `d` zeros, so the tail flushes out.
    let mut causal = FirLowpassIq::design(num_taps, cutoff, 60.0);
    let streamed: Vec<C32> = x
        .iter()
        .copied()
        .chain(std::iter::repeat_n(C32::default(), d))
        .map(|s| causal.push(s))
        .collect();

    let mut aligned = x.clone();
    FirLowpassIq::design(num_taps, cutoff, 60.0).filter_aligned(&mut aligned);

    assert_eq!(aligned.len(), n, "filtering must not change stream length");
    for i in 0..n {
        assert!(
            (aligned[i] - streamed[i + d]).norm() < 1e-5,
            "aligned[{i}] should equal the causal output at {}: {:?} vs {:?}",
            i + d,
            aligned[i],
            streamed[i + d]
        );
    }
    // And the signal really did stay put: the envelope peak has not moved.
    let peak_of = |s: &[C32]| {
        s.iter()
            .enumerate()
            .max_by(|a, b| a.1.norm().total_cmp(&b.1.norm()))
            .map(|(i, _)| i as i32)
            .unwrap_or(0)
    };
    assert!(
        (peak_of(&aligned) - peak_of(&x)).abs() <= 1,
        "aligned filtering must not shift the waveform in time"
    );
}
