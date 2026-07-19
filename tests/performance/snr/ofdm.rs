// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// BER-vs-SNR characterization sweep for OFDM (known-start, no-CFO,
// flat-channel scope of this release).
//
// Feature-gated (`--features throughput`). Tests always pass -- they are
// measurement / characterisation runs, not assertions. Run with --nocapture
// to see the printed table.

use crate::common::add_awgn;
use num_complex::Complex32 as C32;
use orion_sdr::core::Block;
use orion_sdr::demodulate::{EqualizerMethod, OfdmDecider, OfdmDemod, OfdmEqualizer};
use orion_sdr::modulate::{ConstellationOrder, OfdmConfig, OfdmMod};
use orion_sdr::multicarrier::{
    CarrierGrid, CarrierPlan, CyclicPrefixRemove, FftBlock, GridExtract,
};
use orion_sdr::sync::{OfdmPreamble, generate_ofdm_preamble};

const TRIALS: usize = 50;

// Noise scale (relative to the time-domain signal's own power) levels to
// sweep. Not an absolute SNR-in-dB scale like the FT8 sweep -- OFDM has no
// single reference bandwidth here -- but monotonic and printed alongside an
// equivalent per-sample SNR estimate for readability.
const NOISE_SCALES: &[f32] = &[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0];

fn plan(n_fft: usize, cp_len: usize) -> CarrierPlan {
    let half = (n_fft / 2) as i32;
    let data: Vec<i32> = (1..half).chain(-(half - 1)..0).collect();
    CarrierPlan::new(n_fft, cp_len).with_data_carriers(data)
}

fn config(n_fft: usize, cp_len: usize, constellation: ConstellationOrder) -> OfdmConfig {
    OfdmConfig::new(plan(n_fft, cp_len), 48_000.0, 0.0, 1.0, constellation)
}

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

fn run_sweep(label: &str, constellation: ConstellationOrder, seed_base: u64) {
    let cfg = config(64, 8, constellation);

    println!("\n[OFDM {} BER sweep, {} trials/point]", label, TRIALS);
    println!(
        "{:>14} {:>12} {:>10}",
        "noise_scale", "equiv_snr_dB", "mean_BER"
    );
    println!("{}", "-".repeat(40));

    for &noise_scale in NOISE_SCALES {
        let ber = mean_ber_at_noise_scale(&cfg, noise_scale, seed_base);
        let equiv_snr_db = -10.0 * noise_scale.log10();
        println!("{:>14.4} {:>12.1} {:>10.5}", noise_scale, equiv_snr_db, ber);
    }
    println!();
    // Always passes -- this is a measurement run, not an assertion.
}

#[test]
fn snr_sweep_ofdm_qpsk() {
    run_sweep("QPSK", ConstellationOrder::Qpsk, 0x1111_0000_0000_0000);
}

#[test]
fn snr_sweep_ofdm_qam16() {
    run_sweep("QAM-16", ConstellationOrder::Qam16, 0x2222_0000_0000_0000);
}

#[test]
fn snr_sweep_ofdm_qam64() {
    run_sweep("QAM-64", ConstellationOrder::Qam64, 0x3333_0000_0000_0000);
}

/// Convolves `iq` with a short FIR channel (causal, `taps[0]` is the direct
/// path).
fn apply_fir_channel(iq: &[C32], taps: &[C32]) -> Vec<C32> {
    let mut out = vec![C32::default(); iq.len()];
    for (n, &x) in iq.iter().enumerate() {
        for (k, &h) in taps.iter().enumerate() {
            if n + k < out.len() {
                out[n + k] += x * h;
            }
        }
    }
    out
}

fn mean_ber_multipath_at_noise_scale(cfg: &OfdmConfig, noise_scale: f32, seed_base: u64) -> f32 {
    let n_fft = cfg.carrier_plan.n_fft();
    let cp_len = cfg.carrier_plan.cp_len();
    let bps = cfg.bits_per_ofdm_symbol();
    let n_symbols = 20;
    let bits_in: Vec<u8> = (0..n_symbols * bps)
        .map(|i| ((i / 7 + i % 5) & 1) as u8)
        .collect();

    let preamble = OfdmPreamble::new(4, 32).with_training_symbol(n_fft, cp_len);
    let channel_taps = [
        C32::new(0.8, 0.1),
        C32::new(0.0, 0.0),
        C32::new(0.25, -0.15),
    ];

    let mut modstage = OfdmMod::new(cfg);
    let data_iq = modstage.modulate(&bits_in);
    let preamble_iq = generate_ofdm_preamble(&preamble, cfg);

    let mut clean = Vec::with_capacity(preamble_iq.len() + data_iq.len());
    clean.extend_from_slice(&preamble_iq);
    clean.extend_from_slice(&data_iq);
    let channeled_clean = apply_fir_channel(&clean, &channel_taps);
    let sig_power: f32 =
        channeled_clean.iter().map(|s| s.norm_sqr()).sum::<f32>() / channeled_clean.len() as f32;
    let noise_power = sig_power * noise_scale;

    let training_start = preamble.num_repeats * preamble.repeat_len;
    let data_start = preamble.total_len();
    let samples_per_symbol = cfg.samples_per_ofdm_symbol();

    let mut total_errors = 0usize;
    let mut total_bits = 0usize;

    for trial in 0..TRIALS {
        let mut buf = channeled_clean.clone();
        let seed = seed_base
            .wrapping_add(trial as u64)
            .wrapping_add((noise_scale * 1_000_000.0) as u64);
        add_awgn(&mut buf, noise_power, seed);

        let grid = CarrierGrid::from_plan(&cfg.carrier_plan);
        let mut cp_remove = CyclicPrefixRemove::new(n_fft, cp_len);
        let mut fft = FftBlock::new(n_fft);
        let mut eq = OfdmEqualizer::new(cfg, EqualizerMethod::TrainingSymbolHold);

        let training_symbol = &buf[training_start..training_start + n_fft + cp_len];
        let mut training_time = vec![C32::default(); n_fft];
        cp_remove.process(training_symbol, &mut training_time);
        let mut training_freq = vec![C32::default(); n_fft];
        fft.process(&training_time, &mut training_freq);
        eq.estimate_from_training_symbol(&training_freq);

        let iq = &buf[data_start..data_start + data_iq.len()];
        let num_data = grid.num_data_carriers();
        let mut grid_extract = GridExtract::new(grid);
        let mut decider = OfdmDecider::new(cfg);
        let mut bits_out = vec![0u8; bits_in.len()];

        let mut in_off = 0usize;
        let mut out_off = 0usize;
        while in_off + samples_per_symbol <= iq.len() {
            let mut time = vec![C32::default(); n_fft];
            cp_remove.process(&iq[in_off..], &mut time);
            let mut freq = vec![C32::default(); n_fft];
            fft.process(&time, &mut freq);
            let mut equalized = vec![C32::default(); n_fft];
            eq.process(&freq, &mut equalized);
            let mut soft = vec![C32::default(); num_data];
            grid_extract.process(&equalized, &mut soft);
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

fn run_multipath_sweep(label: &str, constellation: ConstellationOrder, seed_base: u64) {
    let cfg = config(64, 8, constellation);

    println!(
        "\n[OFDM {} multipath BER sweep (TrainingSymbolHold equalizer), {} trials/point]",
        label, TRIALS
    );
    println!(
        "{:>14} {:>12} {:>10}",
        "noise_scale", "equiv_snr_dB", "mean_BER"
    );
    println!("{}", "-".repeat(40));

    for &noise_scale in NOISE_SCALES {
        let ber = mean_ber_multipath_at_noise_scale(&cfg, noise_scale, seed_base);
        let equiv_snr_db = -10.0 * noise_scale.log10();
        println!("{:>14.4} {:>12.1} {:>10.5}", noise_scale, equiv_snr_db, ber);
    }
    println!();
    // Always passes -- this is a measurement run, not an assertion.
}

#[test]
fn snr_sweep_ofdm_multipath_qpsk() {
    run_multipath_sweep("QPSK", ConstellationOrder::Qpsk, 0x4444_0000_0000_0000);
}

#[test]
fn snr_sweep_ofdm_multipath_qam16() {
    run_multipath_sweep("QAM-16", ConstellationOrder::Qam16, 0x5555_0000_0000_0000);
}
