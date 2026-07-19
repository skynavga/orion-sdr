// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use num_complex::Complex32 as C32;
use orion_sdr::dsp::Rotator;
use orion_sdr::modulate::{ConstellationOrder, OfdmConfig};
use orion_sdr::multicarrier::CarrierPlan;
use orion_sdr::sync::{OfdmPreamble, generate_ofdm_preamble, ofdm_sync};

const FS: f32 = 48_000.0;

fn config() -> OfdmConfig {
    let plan = CarrierPlan::new(64, 8).with_data_carriers(1..32);
    OfdmConfig::new(plan, FS, 0.0, 1.0, ConstellationOrder::Qpsk)
}

fn preamble() -> OfdmPreamble {
    OfdmPreamble::new(4, 32)
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
    // Well beyond the documented ±capture_hz bound.
    let applied_cfo = capture_hz * 3.0;

    let time_offset = 50;
    let buf = build_test_buffer(time_offset, 64, applied_cfo);

    let results = ofdm_sync(&buf, FS, &pre, 0, buf.len());
    assert!(!results.is_empty());
    let best = results[0];

    // A large residual CFO distorts the S&C timing-metric plateau (not just
    // the correlation phase), so exact-sample timing lock is not guaranteed
    // here the way it is within the documented capture range — that's
    // covered by `ofdm_sync_cfo_estimate_accuracy`. What this test locks in
    // is the ±capture_hz aliasing bound itself: the reported CFO must never
    // exceed the documented range, and must diverge from the true
    // out-of-range applied CFO.
    assert!(
        best.cfo_hz.abs() <= capture_hz + 1e-3,
        "aliased CFO estimate {} Hz should stay within the ±{} Hz capture range",
        best.cfo_hz,
        capture_hz
    );
    assert!(
        (best.cfo_hz - applied_cfo).abs() > capture_hz,
        "expected the reported CFO to diverge from the out-of-range applied CFO due to aliasing"
    );
}

#[test]
fn ofdm_sync_no_false_positive_on_noise() {
    let mut rng: u64 = 0xC0FF_EE00_1234_5678;
    let mut next_f32 = || -> f32 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as f32) / (u64::MAX as f32) - 0.5
    };
    let noise: Vec<C32> = (0..2000)
        .map(|_| C32::new(next_f32(), next_f32()))
        .collect();

    let pre = preamble();
    let results = ofdm_sync(&noise, FS, &pre, 0, noise.len());

    // Noise can produce weak spurious peaks, but none should reach a
    // convincing normalized-metric score.
    if let Some(best) = results.first() {
        assert!(
            best.score < 0.5,
            "unexpected high-confidence sync on pure noise: score {}",
            best.score
        );
    }
}
