// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// SNR sensitivity sweep for FT8 and FT4.
//
// Feature-gated (`--features throughput`).  Tests always pass -- they are
// measurement / characterisation runs, not assertions.  Run with --nocapture
// to see the printed table.

use crate::common::add_awgn;
use orion_sdr::codec::ft4::Ft4Bits;
use orion_sdr::codec::ft4::Ft4Codec;
use orion_sdr::codec::ft8::Ft8Bits;
use orion_sdr::codec::ft8::Ft8Codec;
use orion_sdr::modulate::ft4::FT4_FRAME_LEN;
use orion_sdr::modulate::ft8::FT8_FRAME_LEN;
use orion_sdr::modulate::{Ft4Mod, Ft8Mod};
use orion_sdr::sync::{ft4_sync, ft8_sync};

const FS: f32 = 12_000.0;
const REF_BW_HZ: f32 = 2_500.0;
const SIG_POWER: f32 = 0.5;
const TRIALS: usize = 50;

/// Convert SNR (dB, 2500 Hz reference BW) -> noise_power_per_sample.
fn snr_to_noise_power(snr_db: f32) -> f32 {
    SIG_POWER * FS / (REF_BW_HZ * 10.0_f32.powf(snr_db / 10.0))
}

// SNR levels to sweep: -26 dB to -10 dB in 1 dB steps.
const SNR_LEVELS: &[f32] = &[
    -26.0, -25.0, -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0, -16.0, -15.0, -14.0,
    -13.0, -12.0, -11.0, -10.0,
];

// -- helpers ---------------------------------------------------------------

fn make_ft8_frame_iq(base_hz: f32) -> (Vec<num_complex::Complex32>, Ft8Bits) {
    let payload: Ft8Bits = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60];
    let frame = Ft8Codec::encode(&payload);
    let iq = Ft8Mod::new(FS, base_hz, 0.0, 1.0).modulate(&frame);
    (iq, payload)
}

fn make_ft4_frame_iq(base_hz: f32) -> (Vec<num_complex::Complex32>, Ft4Bits) {
    let payload: Ft4Bits = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60];
    let frame = Ft4Codec::encode(&payload);
    let iq = Ft4Mod::new(FS, base_hz, 0.0, 1.0).modulate(&frame);
    (iq, payload)
}

/// Run one FT8 decode trial at the given noise_power. Returns true on success.
fn try_ft8(
    clean_iq: &[num_complex::Complex32],
    payload: &Ft8Bits,
    base_hz: f32,
    noise_power: f32,
    seed: u64,
) -> bool {
    let total = FT8_FRAME_LEN + FT8_FRAME_LEN / 4;
    let mut buf = vec![num_complex::Complex32::new(0.0, 0.0); total];
    buf[..FT8_FRAME_LEN].copy_from_slice(clean_iq);
    add_awgn(&mut buf, noise_power, seed);

    let results = ft8_sync(&buf, FS, base_hz - 6.25, base_hz + 50.0 + 6.25, 0, 0, 5);
    if results.is_empty() {
        return false;
    }
    Ft8Codec::decode_soft(&results[0].llr).as_ref() == Some(payload)
}

/// Run one FT4 decode trial at the given noise_power. Returns true on success.
fn try_ft4(
    clean_iq: &[num_complex::Complex32],
    payload: &Ft4Bits,
    base_hz: f32,
    noise_power: f32,
    seed: u64,
) -> bool {
    let total = FT4_FRAME_LEN + FT4_FRAME_LEN / 4;
    let mut buf = vec![num_complex::Complex32::new(0.0, 0.0); total];
    buf[..FT4_FRAME_LEN].copy_from_slice(clean_iq);
    add_awgn(&mut buf, noise_power, seed);

    let results = ft4_sync(
        &buf,
        FS,
        base_hz - 20.833,
        base_hz + 100.0 + 20.833,
        0,
        0,
        5,
    );
    if results.is_empty() {
        return false;
    }
    Ft4Codec::decode_soft(&results[0].llr).as_ref() == Some(payload)
}

// -- FT8 sweep ------------------------------------------------------------

#[test]
fn snr_sweep_ft8() {
    let base_hz = 1_000.0_f32;
    let (clean_iq, payload) = make_ft8_frame_iq(base_hz);

    println!(
        "\n[FT8 SNR sweep, ref BW = {:.0} Hz, {} trials/point]",
        REF_BW_HZ, TRIALS
    );
    println!(
        "{:>10} {:>8} {:>8} {:>9}",
        "SNR(dB)", "trials", "decoded", "success%"
    );
    println!("{}", "-".repeat(40));

    for &snr_db in SNR_LEVELS {
        let noise_power = snr_to_noise_power(snr_db);
        let decoded: usize = (0..TRIALS)
            .filter(|&i| {
                let seed = 0x1234_0000_0000_0000_u64
                    .wrapping_add(i as u64)
                    .wrapping_add((snr_db * 100.0) as u64);
                try_ft8(&clean_iq, &payload, base_hz, noise_power, seed)
            })
            .count();
        let pct = 100.0 * decoded as f32 / TRIALS as f32;
        println!(
            "{:>10.1} {:>8} {:>8} {:>8.1}%",
            snr_db, TRIALS, decoded, pct
        );
    }
    println!();
    // Always passes -- this is a measurement run, not an assertion.
}

// -- FT4 sweep ------------------------------------------------------------

#[test]
fn snr_sweep_ft4() {
    let base_hz = 1_000.0_f32;
    let (clean_iq, payload) = make_ft4_frame_iq(base_hz);

    println!(
        "\n[FT4 SNR sweep, ref BW = {:.0} Hz, {} trials/point]",
        REF_BW_HZ, TRIALS
    );
    println!(
        "{:>10} {:>8} {:>8} {:>9}",
        "SNR(dB)", "trials", "decoded", "success%"
    );
    println!("{}", "-".repeat(40));

    for &snr_db in SNR_LEVELS {
        let noise_power = snr_to_noise_power(snr_db);
        let decoded: usize = (0..TRIALS)
            .filter(|&i| {
                let seed = 0xABCD_0000_0000_0000_u64
                    .wrapping_add(i as u64)
                    .wrapping_add((snr_db * 100.0) as u64);
                try_ft4(&clean_iq, &payload, base_hz, noise_power, seed)
            })
            .count();
        let pct = 100.0 * decoded as f32 / TRIALS as f32;
        println!(
            "{:>10.1} {:>8} {:>8} {:>8.1}%",
            snr_db, TRIALS, decoded, pct
        );
    }
    println!();
}
