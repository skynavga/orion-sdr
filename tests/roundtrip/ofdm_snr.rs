// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// CI regression thresholds for OFDM BER vs SNR (known-start, no-CFO,
// flat-channel scope of this release). 50-trial Monte Carlo per SNR point,
// matching tests/performance/snr/ft8.rs's trial structure but asserting a
// pass/fail threshold rather than only printing a characterization table.

use crate::common::add_awgn;
use num_complex::Complex32 as C32;
use orion_sdr::core::Block;
use orion_sdr::demodulate::{OfdmDecider, OfdmDemod};
use orion_sdr::modulate::{ConstellationOrder, OfdmConfig, OfdmMod};
use orion_sdr::multicarrier::CarrierPlan;

const TRIALS: usize = 50;

fn plan(n_fft: usize, cp_len: usize) -> CarrierPlan {
    let half = (n_fft / 2) as i32;
    let data: Vec<i32> = (1..half).chain(-(half - 1)..0).collect();
    CarrierPlan::new(n_fft, cp_len).with_data_carriers(data)
}

fn config(n_fft: usize, cp_len: usize, constellation: ConstellationOrder) -> OfdmConfig {
    OfdmConfig::new(plan(n_fft, cp_len), 48_000.0, 0.0, 1.0, constellation)
}

/// Mean BER over `TRIALS` independent AWGN draws at `noise_scale` (relative
/// to the time-domain signal's own power).
fn mean_ber_at_noise_scale(cfg: &OfdmConfig, noise_scale: f32, seed_base: u64) -> f32 {
    let bps = cfg.bits_per_ofdm_symbol();
    let n_symbols = 20;
    let bits_in: Vec<u8> = (0..n_symbols * bps)
        .map(|i| ((i / 7 + i % 5) & 1) as u8)
        .collect();

    let mut modstage = OfdmMod::new(cfg);
    let clean_iq = modstage.modulate(&bits_in);
    let sig_power: f32 = clean_iq.iter().map(|s| s.norm_sqr()).sum::<f32>() / clean_iq.len() as f32;
    let noise_power = sig_power * noise_scale;

    let samples_per_symbol = cfg.samples_per_ofdm_symbol();
    let mut total_errors = 0usize;
    let mut total_bits = 0usize;

    for trial in 0..TRIALS {
        let mut iq = clean_iq.clone();
        let seed = seed_base
            .wrapping_add(trial as u64)
            .wrapping_add((noise_scale * 1_000_000.0) as u64);
        add_awgn(&mut iq, noise_power, seed);

        let mut demod = OfdmDemod::new(cfg);
        let mut decider = OfdmDecider::new(cfg);
        let num_data = demod.num_data_carriers();
        let mut soft = vec![C32::default(); num_data];
        let mut bits_out = vec![0u8; bits_in.len()];

        let mut in_off = 0usize;
        let mut out_off = 0usize;
        while in_off + samples_per_symbol <= iq.len() {
            demod.process(&iq[in_off..], &mut soft);
            decider.process(&soft, &mut bits_out[out_off..]);
            in_off += samples_per_symbol;
            out_off += bps;
        }

        total_errors += bits_in
            .iter()
            .zip(bits_out.iter())
            .filter(|(a, b)| a != b)
            .count();
        total_bits += bits_in.len();
    }

    total_errors as f32 / total_bits as f32
}

#[test]
fn ofdm_qpsk_ber_below_threshold_at_moderate_snr() {
    let cfg = config(64, 8, ConstellationOrder::Qpsk);
    let ber = mean_ber_at_noise_scale(&cfg, 0.02, 0x1234_5678_ABCD_0000);
    assert!(
        ber < 0.01,
        "OFDM QPSK mean BER {:.4} over {} trials exceeded regression threshold at moderate SNR",
        ber,
        TRIALS
    );
}

#[test]
fn ofdm_qpsk_ber_degrades_at_low_snr() {
    let cfg = config(64, 8, ConstellationOrder::Qpsk);
    let ber = mean_ber_at_noise_scale(&cfg, 2.0, 0x9ABC_DEF0_1234_0000);
    assert!(
        ber > 0.1,
        "expected OFDM QPSK mean BER to degrade substantially at low SNR, got {:.4}",
        ber
    );
}
