// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// Acquisition-probability-vs-SNR characterization sweep for ofdm_sync
// (fractional CFO/timing acquisition scope of this release).
//
// Feature-gated (`--features throughput`). Tests always pass -- they are
// measurement / characterisation runs, not assertions. Run with --nocapture
// to see the printed table.

use crate::common::add_awgn;
use num_complex::Complex32 as C32;
use orion_sdr::dsp::Rotator;
use orion_sdr::modulate::{ConstellationOrder, OfdmConfig};
use orion_sdr::multicarrier::CarrierPlan;
use orion_sdr::sync::{OfdmPreamble, generate_ofdm_preamble, ofdm_sync};

const FS: f32 = 48_000.0;
const TRIALS: usize = 50;

const NOISE_SCALES: &[f32] = &[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0];

fn config() -> OfdmConfig {
    let plan = CarrierPlan::new(64, 8).with_data_carriers(1..32);
    OfdmConfig::new(plan, FS, 0.0, 1.0, ConstellationOrder::Qpsk)
}

#[test]
fn snr_sweep_ofdm_sync_acquisition() {
    let cfg = config();
    let preamble = OfdmPreamble::new(4, 32);
    let clean_preamble = generate_ofdm_preamble(&preamble, &cfg);
    let sig_power: f32 =
        clean_preamble.iter().map(|s| s.norm_sqr()).sum::<f32>() / clean_preamble.len() as f32;

    let time_offset = 50usize;
    let tail = 64usize;

    println!("\n[OFDM sync acquisition sweep, {} trials/point]", TRIALS);
    println!(
        "{:>14} {:>10} {:>10} {:>12}",
        "noise_scale", "trials", "locked", "success%"
    );
    println!("{}", "-".repeat(50));

    for &noise_scale in NOISE_SCALES {
        let noise_power = sig_power * noise_scale;
        let locked = (0..TRIALS)
            .filter(|&trial| {
                let mut buf = vec![C32::default(); time_offset];
                buf.extend_from_slice(&clean_preamble);
                buf.extend(vec![C32::default(); tail]);

                let seed = 0xF00D_0000_0000_0000_u64
                    .wrapping_add(trial as u64)
                    .wrapping_add((noise_scale * 1_000_000.0) as u64);
                add_awgn(&mut buf, noise_power, seed);

                let results = ofdm_sync(&buf, FS, &preamble, 0, buf.len());
                results
                    .first()
                    .map(|r| r.start_sample == time_offset && r.score > 0.5)
                    .unwrap_or(false)
            })
            .count();

        let pct = 100.0 * locked as f32 / TRIALS as f32;
        println!(
            "{:>14.3} {:>10} {:>10} {:>11.1}%",
            noise_scale, TRIALS, locked, pct
        );
    }
    println!();
    // Always passes -- this is a measurement run, not an assertion.
}

#[test]
fn snr_sweep_ofdm_sync_wide_cfo_acquisition() {
    let n_fft = 64;
    let cp_len = 8;
    let cfg = config();
    let preamble = OfdmPreamble::new(4, 32).with_training_symbol(n_fft, cp_len);
    let subcarrier_spacing_hz = FS / n_fft as f32;
    // Well beyond Release E's ±½-spacing fractional-only capture range.
    let applied_cfo = 5.0 * subcarrier_spacing_hz + 0.3 * subcarrier_spacing_hz;

    let clean_preamble = generate_ofdm_preamble(&preamble, &cfg);
    let sig_power: f32 =
        clean_preamble.iter().map(|s| s.norm_sqr()).sum::<f32>() / clean_preamble.len() as f32;

    let time_offset = 50usize;
    let tail = 64usize;

    println!(
        "\n[OFDM sync wide-CFO acquisition sweep, applied_cfo={:.1} Hz ({:.2} subcarrier spacings), {} trials/point]",
        applied_cfo,
        applied_cfo / subcarrier_spacing_hz,
        TRIALS
    );
    println!(
        "{:>14} {:>10} {:>10} {:>12}",
        "noise_scale", "trials", "locked", "success%"
    );
    println!("{}", "-".repeat(50));

    for &noise_scale in NOISE_SCALES {
        let noise_power = sig_power * noise_scale;
        let locked = (0..TRIALS)
            .filter(|&trial| {
                let mut buf = vec![C32::default(); time_offset];
                buf.extend_from_slice(&clean_preamble);
                buf.extend(vec![C32::default(); tail]);

                let mut rot = Rotator::new(applied_cfo, FS);
                let mut with_cfo = vec![C32::default(); buf.len()];
                rot.rotate_block(&buf, &mut with_cfo);

                let seed = 0xFEED_0000_0000_0000_u64
                    .wrapping_add(trial as u64)
                    .wrapping_add((noise_scale * 1_000_000.0) as u64);
                add_awgn(&mut with_cfo, noise_power, seed);

                let results = ofdm_sync(&with_cfo, FS, &preamble, 0, with_cfo.len());
                results
                    .first()
                    .map(|r| {
                        if r.start_sample != time_offset || r.score <= 0.5 {
                            return false;
                        }
                        let total_cfo =
                            r.cfo_hz + r.integer_cfo_bins as f32 * subcarrier_spacing_hz;
                        (total_cfo - applied_cfo).abs() < subcarrier_spacing_hz * 0.1
                    })
                    .unwrap_or(false)
            })
            .count();

        let pct = 100.0 * locked as f32 / TRIALS as f32;
        println!(
            "{:>14.3} {:>10} {:>10} {:>11.1}%",
            noise_scale, TRIALS, locked, pct
        );
    }
    println!();
    // Always passes -- this is a measurement run, not an assertion.
}
