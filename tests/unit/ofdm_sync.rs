// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use num_complex::Complex32 as C32;
use orion_sdr::dsp::Rotator;
use orion_sdr::modulate::{ConstellationOrder, OfdmConfig};
use orion_sdr::multicarrier::CarrierPlan;
use orion_sdr::sync::{OfdmPreamble, generate_ofdm_preamble, ofdm_sync};

const FS: f32 = 48_000.0;
const N_FFT: usize = 64;
const CP_LEN: usize = 8;

fn config() -> OfdmConfig {
    let plan = CarrierPlan::new(N_FFT, CP_LEN).with_data_carriers(1..32);
    OfdmConfig::new(plan, FS, 0.0, 1.0, ConstellationOrder::Qpsk)
}

fn preamble() -> OfdmPreamble {
    OfdmPreamble::new(4, 32)
}

fn preamble_with_training() -> OfdmPreamble {
    preamble().with_training_symbol(N_FFT, CP_LEN)
}

fn subcarrier_spacing_hz() -> f32 {
    FS / N_FFT as f32
}

fn apply_cfo(iq: &[C32], cfo_hz: f32, fs: f32) -> Vec<C32> {
    let mut rot = Rotator::new(cfo_hz, fs);
    let mut out = vec![C32::default(); iq.len()];
    rot.rotate_block(iq, &mut out);
    out
}

fn build_test_buffer(time_offset: usize, tail_len: usize, cfo_hz: f32) -> Vec<C32> {
    let cfg = config();
    let pre = preamble();
    let clean_preamble = generate_ofdm_preamble(&pre, &cfg);

    let mut buf = vec![C32::default(); time_offset];
    buf.extend_from_slice(&clean_preamble);
    buf.extend(vec![C32::default(); tail_len]);

    if cfo_hz != 0.0 {
        apply_cfo(&buf, cfo_hz, FS)
    } else {
        buf
    }
}

#[test]
fn ofdm_sync_finds_known_offset() {
    let time_offset = 100;
    let buf = build_test_buffer(time_offset, 64, 0.0);
    let pre = preamble();

    let results = ofdm_sync(&buf, FS, &pre, 0, buf.len());
    assert!(!results.is_empty(), "expected at least one sync candidate");

    let best = results[0];
    assert_eq!(
        best.start_sample, time_offset,
        "expected peak at the known preamble start"
    );
    assert!(
        best.score > 0.9,
        "expected a strong timing-metric score, got {}",
        best.score
    );
}

#[test]
fn ofdm_sync_cfo_estimate_accuracy() {
    let pre = preamble();
    // Capture range is ±fs / (2 * repeat_len); pick a CFO safely within it.
    let capture_hz = FS / (2.0 * pre.repeat_len as f32);
    let applied_cfo = capture_hz * 0.4;

    let time_offset = 50;
    let buf = build_test_buffer(time_offset, 64, applied_cfo);

    let results = ofdm_sync(&buf, FS, &pre, 0, buf.len());
    assert!(!results.is_empty());
    let best = results[0];

    assert_eq!(best.start_sample, time_offset);
    let tol_hz = capture_hz * 0.1;
    assert!(
        (best.cfo_hz - applied_cfo).abs() < tol_hz,
        "CFO estimate {} Hz too far from applied {} Hz (tol {} Hz)",
        best.cfo_hz,
        applied_cfo,
        tol_hz
    );
}

#[test]
fn ofdm_sync_cfo_beyond_half_spacing_aliases() {
    let pre = preamble();
    let capture_hz = FS / (2.0 * pre.repeat_len as f32);
    // Well beyond the documented ±capture_hz bound, and deliberately NOT an
    // integer multiple of the aliasing period (2·capture_hz), so the aliased
    // estimate lands at a specific, predictable value rather than on the
    // ±capture_hz boundary. applied = 2.3·capture_hz aliases to
    // 2.3·capture_hz − 2·capture_hz = 0.3·capture_hz.
    let applied_cfo = capture_hz * 2.3;
    let period = 2.0 * capture_hz;
    let expected_alias = applied_cfo - period; // = 0.3 * capture_hz

    let time_offset = 50;
    let buf = build_test_buffer(time_offset, 64, applied_cfo);

    let results = ofdm_sync(&buf, FS, &pre, 0, buf.len());
    assert!(!results.is_empty());
    let best = results[0];

    // The estimator recovers the phase modulo 2π, i.e. the CFO modulo the
    // aliasing period. Assert the estimate matches the *predicted* aliased
    // value (0.3·capture_hz), not merely that it stays inside the range — a
    // structural bound the atan2-based formula satisfies for any input. This
    // is what actually demonstrates aliasing and gives Release F's
    // integer-CFO extension a concrete baseline. (A large residual CFO can
    // still perturb the S&C timing-metric plateau, so exact-sample timing
    // lock is not asserted here — that is covered by
    // `ofdm_sync_cfo_estimate_accuracy` within the capture range.)
    let tol_hz = capture_hz * 0.1;
    assert!(
        (best.cfo_hz - expected_alias).abs() < tol_hz,
        "aliased CFO {} Hz should fold to {} Hz (applied {} Hz mod {} Hz), tol {} Hz",
        best.cfo_hz,
        expected_alias,
        applied_cfo,
        period,
        tol_hz
    );
}

#[test]
fn ofdm_sync_no_false_positive_on_noise() {
    // "No false positive" is a statistical claim; sweep several independent
    // noise realizations rather than trusting one lucky/unlucky seed. Every
    // realization must produce a candidate (the search always scores every
    // offset, so an empty result here would itself be a regression — the old
    // `if let Some(..)` form would have passed vacuously on that) and none may
    // reach a convincing score.
    let pre = preamble();
    let seeds: [u64; 5] = [
        0xC0FF_EE00_1234_5678,
        0x1357_9BDF_2468_ACE0,
        0xDEAD_BEEF_CAFE_F00D,
        0x0BAD_F00D_1122_3344,
        0xA5A5_5A5A_9999_6666,
    ];

    for &seed in &seeds {
        let mut rng = seed;
        let mut next_f32 = || -> f32 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f32) / (u64::MAX as f32) - 0.5
        };
        let noise: Vec<C32> = (0..2000)
            .map(|_| C32::new(next_f32(), next_f32()))
            .collect();

        let results = ofdm_sync(&noise, FS, &pre, 0, noise.len());
        let best = results
            .first()
            .unwrap_or_else(|| panic!("seed {:#x}: expected a candidate on noise", seed));
        assert!(
            best.score < 0.5,
            "seed {:#x}: unexpected high-confidence sync on pure noise: score {}",
            seed,
            best.score
        );
    }
}

#[test]
fn ofdm_sync_integer_cfo_recovers_multi_spacing_offset() {
    let cfg = config();
    let pre = preamble_with_training();
    let spacing = subcarrier_spacing_hz();

    // A CFO several whole subcarrier spacings wide, well beyond the
    // fractional-only ±½-spacing capture range from Release E.
    //
    // Note: the fractional S&C estimator's own ambiguity period is
    // `fs / repeat_len`, not `fs / n_fft` — for this preamble
    // (`repeat_len = n_fft / 2`) that's *two* subcarrier spacings, so
    // `cfo_hz` alone can legitimately land anywhere in `(-spacing, +spacing]`
    // rather than only `(-spacing/2, +spacing/2]`. The integer search
    // resolves whatever residual the fractional stage actually reports, so
    // the invariant this test checks is the reconstructed total
    // (`cfo_hz + integer_cfo_bins * spacing`), not a specific hardcoded
    // `integer_cfo_bins` value tied to a naive spacing-only decomposition.
    let applied_integer_bins = 3i32;
    let applied_cfo = applied_integer_bins as f32 * spacing + 0.3 * spacing;

    let time_offset = 50;
    let clean_preamble = generate_ofdm_preamble(&pre, &cfg);
    let mut buf = vec![C32::default(); time_offset];
    buf.extend_from_slice(&clean_preamble);
    buf.extend(vec![C32::default(); 64]);

    let mut rot = Rotator::new(applied_cfo, FS);
    let mut with_cfo = vec![C32::default(); buf.len()];
    rot.rotate_block(&buf, &mut with_cfo);

    let results = ofdm_sync(&with_cfo, FS, &pre, 0, with_cfo.len());
    assert!(!results.is_empty());
    let best = results[0];

    assert_eq!(best.start_sample, time_offset);
    assert_ne!(
        best.integer_cfo_bins, 0,
        "expected the integer stage to recover a nonzero multi-spacing shift"
    );
    let total_cfo = best.cfo_hz + best.integer_cfo_bins as f32 * spacing;
    let tol_hz = spacing * 0.1;
    assert!(
        (total_cfo - applied_cfo).abs() < tol_hz,
        "reconstructed total CFO {} Hz too far from applied {} Hz (tol {} Hz)",
        total_cfo,
        applied_cfo,
        tol_hz
    );
}

#[test]
fn ofdm_sync_total_cfo_matches_applied_offset() {
    let cfg = config();
    let pre = preamble_with_training();
    let spacing = subcarrier_spacing_hz();

    // A combined fractional + integer offset: 3 whole spacings plus 30% of
    // one more, well beyond Release E's fractional-only ±½-spacing capture.
    let applied_integer_bins = 3i32;
    let applied_fraction = 0.3 * spacing;
    let applied_cfo = applied_integer_bins as f32 * spacing + applied_fraction;

    let time_offset = 50;
    let clean_preamble = generate_ofdm_preamble(&pre, &cfg);
    let mut buf = vec![C32::default(); time_offset];
    buf.extend_from_slice(&clean_preamble);
    buf.extend(vec![C32::default(); 64]);

    let mut rot = Rotator::new(applied_cfo, FS);
    let mut with_cfo = vec![C32::default(); buf.len()];
    rot.rotate_block(&buf, &mut with_cfo);

    let results = ofdm_sync(&with_cfo, FS, &pre, 0, with_cfo.len());
    assert!(!results.is_empty());
    let best = results[0];

    assert_eq!(best.start_sample, time_offset);
    let total_cfo = best.cfo_hz + best.integer_cfo_bins as f32 * spacing;
    let tol_hz = spacing * 0.1;
    assert!(
        (total_cfo - applied_cfo).abs() < tol_hz,
        "combined CFO estimate {} Hz too far from applied {} Hz (tol {} Hz)",
        total_cfo,
        applied_cfo,
        tol_hz
    );
}
