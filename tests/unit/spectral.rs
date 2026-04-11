// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use orion_sdr::util::{
    power_spectrum, spectrum_snr_db, spectrum_bw_hz,
    tone, SIGNAL_THRESHOLD,
};

const FS: f32 = 48_000.0;

// ── power_spectrum ───────────────────────────────────────────────────────────

#[test]
fn power_spectrum_returns_correct_bin_count() {
    let samples = tone(FS, 1000.0, 4096, 1.0);
    let (bins, bin_hz) = power_spectrum(&samples, FS);
    assert_eq!(bins.len(), 2049);
    assert!((bin_hz - FS / 4096.0).abs() < 0.01);
}

#[test]
fn power_spectrum_peak_at_tone_frequency() {
    let freq = 5000.0;
    let samples = tone(FS, freq, 4096, 1.0);
    let (bins, bin_hz) = power_spectrum(&samples, FS);
    let expected_bin = (freq / bin_hz).round() as usize;
    let peak_bin = bins.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert!((peak_bin as isize - expected_bin as isize).unsigned_abs() <= 1,
        "peak at bin {} but expected ~{}", peak_bin, expected_bin);
}

#[test]
fn power_spectrum_short_input_pads() {
    let samples = tone(FS, 1000.0, 32, 1.0);
    let (bins, _) = power_spectrum(&samples, FS);
    assert_eq!(bins.len(), 33); // 64/2 + 1
}

// ── spectrum_snr_db ──────────────────────────────────────────────────────────

#[test]
fn snr_high_for_clean_tone() {
    let freq = 3000.0;
    let samples = tone(FS, freq, 4096, 1.0);
    let snr = spectrum_snr_db(&samples, FS, freq);
    assert!(snr > 30.0, "expected high SNR for clean tone, got {snr}");
}

#[test]
fn snr_low_for_noise() {
    let mut rng: u64 = 0x1234_5678_DEAD_BEEF;
    let samples: Vec<f32> = (0..4096).map(|_| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f32 * (1.0 / (1u64 << 53) as f32) * 2.0 - 1.0
    }).collect();
    let snr = spectrum_snr_db(&samples, FS, 3000.0);
    assert!(snr < 15.0, "expected low SNR for noise, got {snr}");
}

// ── spectrum_bw_hz ───────────────────────────────────────────────────────────

#[test]
fn bw_narrow_for_pure_tone() {
    let freq = 5000.0;
    let samples = tone(FS, freq, 4096, 1.0);
    let bw = spectrum_bw_hz(&samples, FS, freq, 7.0);
    assert!(bw < 200.0, "expected narrow BW for pure tone, got {bw}");
}

#[test]
fn bw_minimum_for_short_input() {
    let samples = tone(FS, 1000.0, 64, 1.0);
    let bw = spectrum_bw_hz(&samples, FS, 1000.0, 7.0);
    assert!(bw > 0.0);
}

// ── Constants ────────────────────────────────────────────────────────────────

#[test]
fn signal_threshold_reasonable() {
    const _: () = assert!(SIGNAL_THRESHOLD > 0.0);
    const _: () = assert!(SIGNAL_THRESHOLD < 1.0);
}
