// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use num_complex::Complex32 as C32;
use orion_sdr::core::Block;
use orion_sdr::dsp::Rotator;
use orion_sdr::modulate::{ConstellationOrder, OfdmConfig};
use orion_sdr::multicarrier::{CarrierPlan, FftBlock};
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

// ── The training symbol occupies exactly the bins the plan does ─────────────
//
// `generate_training_symbol_time_domain` used to take the occupied band's
// half-width, which can only describe a symmetric span and so nulled DC
// unconditionally — while `CarrierPlan::with_contiguous_data(_, true)` was
// handing DC out as a data carrier. The receiver divided the never-transmitted
// bin by a nonzero reference and equalized the payload with the result.
//
// These assert the bin set structurally rather than inferring it from a
// decode: a link-level test only makes a tx/rx disagreement *probable*, and
// this one cost exactly one carrier of 33.

/// The bins the preamble's training symbol actually transmits — strip the CP,
/// forward-FFT, and keep everything above the numerical floor. What the
/// receiver sees, not what the generator meant to emit.
fn training_symbol_loaded_bins(pre: &OfdmPreamble, cfg: &OfdmConfig) -> Vec<usize> {
    let training = pre
        .training_symbol
        .expect("this preamble carries a training symbol");
    let iq = generate_ofdm_preamble(pre, cfg);
    let start = pre.num_repeats * pre.repeat_len + training.cp_len;
    let mut fft = FftBlock::new(training.n_fft);
    let mut freq = vec![C32::default(); training.n_fft];
    fft.process(&iq[start..start + training.n_fft], &mut freq);
    // The known pattern is unit-magnitude and `FftBlock(IfftBlock(x)) == x`, so
    // a loaded bin returns at exactly `cfg.gain` and an unloaded one at float
    // noise some six orders of magnitude down. Half the loaded level is nowhere
    // near either.
    freq.iter()
        .enumerate()
        .filter(|(_, v)| v.norm() > 0.5 * cfg.gain)
        .map(|(bin, _)| bin)
        .collect()
}

fn plan_config(plan: CarrierPlan) -> OfdmConfig {
    OfdmConfig::new(plan, FS, 0.0, 1.0, ConstellationOrder::Qpsk)
}

#[test]
fn training_symbol_loads_exactly_the_plans_occupied_bins() {
    let pre = preamble_with_training();
    let pilots = [(-6i32, C32::new(1.0, 0.0)), (9, C32::new(0.0, 1.0))];

    for (name, plan) in [
        (
            "contiguous, DC nulled",
            CarrierPlan::new(N_FFT, CP_LEN).with_contiguous_data(4, false),
        ),
        (
            "contiguous, DC occupied",
            CarrierPlan::new(N_FFT, CP_LEN).with_contiguous_data(4, true),
        ),
        (
            "contiguous with pilots, DC occupied",
            CarrierPlan::new(N_FFT, CP_LEN)
                .with_pilot_carriers(pilots)
                .with_contiguous_data(4, true),
        ),
        (
            // The band edge cannot express this one at all: a one-sided plan
            // has the same occupied half-width as its symmetric counterpart.
            "one-sided (positive frequencies only)",
            CarrierPlan::new(N_FFT, CP_LEN).with_data_carriers(1..24),
        ),
    ] {
        let cfg = plan_config(plan.clone());
        assert_eq!(
            training_symbol_loaded_bins(&pre, &cfg),
            plan.occupied_bins(),
            "{name}: the training symbol must load the plan's occupied bins and nothing else"
        );
    }
}

#[test]
fn occupying_dc_puts_dc_in_the_training_symbol() {
    // The defect in one line. `with_contiguous_data(_, true)` promises DC is a
    // data carrier; the training symbol has to keep that promise or the
    // equalizer's estimate there is noise.
    let pre = preamble_with_training();
    for include_dc in [false, true] {
        let plan = CarrierPlan::new(N_FFT, CP_LEN).with_contiguous_data(4, include_dc);
        assert_eq!(
            plan.data_carriers().contains(&0),
            include_dc,
            "the plan itself must agree about DC"
        );
        assert_eq!(
            training_symbol_loaded_bins(&pre, &plan_config(plan)).contains(&0),
            include_dc,
            "training symbol DC (include_dc = {include_dc})"
        );
    }
}

#[test]
fn a_dc_nulled_plan_still_loads_the_symmetric_span() {
    // This is a fix in a path every OFDM user is on, so the DC-off case must
    // not move at all. For a contiguous plan the occupied bin set is exactly
    // the symmetric span `1..=occupied_half` and its mirror — which is what the
    // band-edge construction produced — so deriving it from the plan is a
    // no-op here and a fix only where the two disagreed.
    let guard = 4;
    let plan = CarrierPlan::new(N_FFT, CP_LEN).with_contiguous_data(guard, false);
    let half = plan.occupied_half_carriers();
    let expected: Vec<usize> = (1..=half).chain(N_FFT - half..N_FFT).collect();
    assert_eq!(plan.occupied_bins(), expected);
    assert_eq!(
        training_symbol_loaded_bins(&preamble_with_training(), &plan_config(plan)),
        expected
    );
}

#[test]
fn the_repeats_never_load_dc_even_when_the_plan_does() {
    // Unlike the training symbol, the S&C repeats owe the plan no bin-for-bin
    // agreement: they are correlated for timing and CFO and never used to
    // estimate a channel. A loaded bin 0 is a constant offset across the
    // segment, identically self-similar at every lag, so it broadens the timing
    // plateau while adding nothing to localize on.
    let cfg = plan_config(CarrierPlan::new(N_FFT, CP_LEN).with_contiguous_data(4, true));
    let pre = preamble(); // repeats only, no training symbol
    let iq = generate_ofdm_preamble(&pre, &cfg);
    let mean: C32 = iq.iter().sum::<C32>() / iq.len() as f32;
    assert!(
        mean.norm() < 1.0e-5,
        "the repeats carry a DC component of {:.3e}",
        mean.norm()
    );
}
