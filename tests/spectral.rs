use orion_sdr::util::{
    power_spectrum, spectrum_snr_db, spectrum_bw_hz, best_sync,
    tone, SIGNAL_THRESHOLD, PSK31_BW_HZ,
};

const FS: f32 = 48_000.0;

// ── power_spectrum ───────────────────────────────────────────────────────────

#[test]
fn power_spectrum_returns_correct_bin_count() {
    let samples = tone(FS, 1000.0, 4096, 1.0);
    let (bins, bin_hz) = power_spectrum(&samples, FS);
    // FFT size = 4096 → 2049 bins
    assert_eq!(bins.len(), 2049);
    assert!((bin_hz - FS / 4096.0).abs() < 0.01);
}

#[test]
fn power_spectrum_peak_at_tone_frequency() {
    let freq = 5000.0;
    let samples = tone(FS, freq, 4096, 1.0);
    let (bins, bin_hz) = power_spectrum(&samples, FS);
    let expected_bin = (freq / bin_hz).round() as usize;
    // Peak should be at or within 1 bin of expected
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
    // Input shorter than 64 should still work (clamped to 64).
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
    // Pure noise (no tone) — SNR should be low
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
    // Pure tone should have very narrow bandwidth (a few bins)
    assert!(bw < 200.0, "expected narrow BW for pure tone, got {bw}");
}

#[test]
fn bw_minimum_for_short_input() {
    let samples = tone(FS, 1000.0, 64, 1.0);
    let bw = spectrum_bw_hz(&samples, FS, 1000.0, 7.0);
    // Should return something reasonable, not panic
    assert!(bw > 0.0);
}

// ── best_sync ────────────────────────────────────────────────────────────────

#[test]
fn best_sync_picks_earliest_near_carrier() {
    use orion_sdr::sync::psk31_sync::Psk31SyncResult;

    let results = vec![
        Psk31SyncResult { carrier_hz: 1000.0, time_sym: 10, freq_bin: 0, score: 1.0, soft_bits: vec![] },
        Psk31SyncResult { carrier_hz: 1001.0, time_sym: 5,  freq_bin: 0, score: 1.0, soft_bits: vec![] },
        Psk31SyncResult { carrier_hz: 5000.0, time_sym: 1,  freq_bin: 0, score: 1.0, soft_bits: vec![] },
    ];
    let baud = 31.25;
    let (hz, sym) = best_sync(&results, 1000.0, baud).unwrap();
    // Should pick time_sym=5 (earliest within 2×baud of 1000 Hz)
    assert_eq!(sym, 5);
    assert!((hz - 1001.0).abs() < 0.01);
}

#[test]
fn best_sync_none_when_no_match() {
    use orion_sdr::sync::psk31_sync::Psk31SyncResult;

    let results = vec![
        Psk31SyncResult { carrier_hz: 5000.0, time_sym: 1, freq_bin: 0, score: 1.0, soft_bits: vec![] },
    ];
    assert!(best_sync(&results, 1000.0, 31.25).is_none());
}

#[test]
fn best_sync_empty_input() {
    assert!(best_sync(&[], 1000.0, 31.25).is_none());
}

// ── Constants ────────────────────────────────────────────────────────────────

#[test]
fn signal_threshold_reasonable() {
    assert!(SIGNAL_THRESHOLD > 0.0);
    assert!(SIGNAL_THRESHOLD < 1.0);
}

#[test]
fn psk31_bw_is_twice_baud() {
    assert!((PSK31_BW_HZ - 62.5).abs() < 0.01);
}
