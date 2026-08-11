// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// Frame-error-rate-vs-noise characterization sweep for the COFDM frame (MAC)
// layer, over the two concatenations the throughput benchmarks also use.
//
// Feature-gated (`--features throughput`). Tests always pass -- they are
// measurement / characterisation runs, not assertions. Run with --nocapture
// to see the printed table.
//
// Backs the "COFDM frame-error-rate vs. noise scale" table in
// docs/performance.md, which was previously produced by a throwaway harness
// and had drifted a full noise decade out of date as a result.
//
// **Two methodology points, both load-bearing when comparing numbers:**
//
// - Decoding uses the *batch* `OfdmFrameDemod` at a known start, so this
//   measures the concatenated FEC rather than acquisition. Feeding the
//   streaming receiver instead folds in sync failures and reports a far worse
//   figure for the same link (0.35 rather than 0.000 at `noise_scale = 0.02`).
//   `snr::ofdm_sync` covers acquisition separately.
// - `noise_scale` is AWGN power relative to the **payload's** power, not the
//   whole buffer's mean. A buffer-mean reference is hostage to the preamble:
//   before it was band-limited it was ~30 dB hotter than the payload and
//   full-band, so the same nominal figure injected substantially more noise.

use crate::common::add_awgn;
use num_complex::Complex32 as C32;
use orion_sdr::demodulate::OfdmFrameDemod;
use orion_sdr::fec::{ConvCode, FrameMetadata, FramePacket, InnerFec, OuterFec, PunctureRate};
use orion_sdr::modulate::{ConstellationOrder, Mcs, McsTable, OfdmConfig, OfdmFrameMod};
use orion_sdr::multicarrier::CarrierPlan;
use orion_sdr::sync::OfdmPreamble;

const TRIALS: usize = 100;
const FRAME_N: usize = 64; // n_fft
const FRAME_CP: usize = 8;
const PAYLOAD_LEN: usize = 96;

/// Straddles the cliff both concatenations now fall off between 0.6 and 0.8.
/// The lower points are kept to show the FEC holding FER = 0 well past the
/// noise scale at which uncoded QPSK is already substantially errored.
const NOISE_SCALES: &[f32] = &[0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

/// The same plan the COFDM throughput benchmarks use, so the two tables
/// describe one link.
fn frame_config() -> OfdmConfig {
    let half = (FRAME_N / 2) as i32;
    let data: Vec<i32> = (1..half).chain(-(half - 1)..0).collect();
    let plan = CarrierPlan::new(FRAME_N, FRAME_CP).with_data_carriers(data);
    OfdmConfig::new(plan, 48_000.0, 0.0, 1.0, ConstellationOrder::Bpsk)
}

fn frame_preamble(cfg: &OfdmConfig) -> OfdmPreamble {
    OfdmPreamble::new(4, 16)
        .with_training_symbol(cfg.carrier_plan.n_fft(), cfg.carrier_plan.cp_len())
}

/// The DVB-style concatenation: punctured convolutional r1/2 + RS(60,52).
fn conv_rs_table() -> McsTable {
    McsTable::new(vec![Mcs::new(
        ConstellationOrder::Qpsk,
        InnerFec::Convolutional {
            rate: PunctureRate::R1_2,
            code: ConvCode::K5,
        },
        OuterFec::ReedSolomon { n: 60, n_parity: 8 },
    )])
}

/// Fraction of `TRIALS` frames whose payload does not come back byte-exact.
///
/// A frame counts as an error if it fails to decode *or* decodes to the wrong
/// bytes — a CRC collision would otherwise be scored as a success.
fn mean_fer(table: &McsTable, mcs_index: u8, noise_scale: f32, seed_base: u64) -> f32 {
    let cfg = frame_config();
    let pre = frame_preamble(&cfg);
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let payload: Vec<u8> = (0..PAYLOAD_LEN)
        .map(|i| ((i * 37 + 11) & 0xff) as u8)
        .collect();
    let frame = FramePacket::new(FrameMetadata::new(1, mcs_index), payload.clone());
    let frame_iq = modu.modulate_frame(&frame, 0);

    // Noise is referenced to the payload, so the figure means one thing
    // regardless of what the preamble is doing — see the module header.
    let body = &frame_iq[pre.total_len()..];
    let body_power: f32 = body.iter().map(|c| c.norm_sqr()).sum::<f32>() / body.len() as f32;

    let demod = OfdmFrameDemod::new(cfg, table.clone());
    let mut errors = 0usize;
    for trial in 0..TRIALS {
        let mut rx: Vec<C32> = body.to_vec();
        if noise_scale > 0.0 {
            add_awgn(&mut rx, body_power * noise_scale, seed_base + trial as u64);
        }
        let ok = demod
            .decode(&rx)
            .map(|f| f.payload == payload)
            .unwrap_or(false);
        if !ok {
            errors += 1;
        }
    }
    errors as f32 / TRIALS as f32
}

#[test]
fn snr_sweep_cofdm_frame_error_rate() {
    let ldpc_bch = McsTable::default_ladder(); // mcs 1 is the QPSK payload
    let conv_rs = conv_rs_table();

    println!(
        "\n[COFDM frame-error-rate sweep, n_fft={FRAME_N}, cp_len={FRAME_CP}, \
         QPSK payload, {PAYLOAD_LEN}-byte payload, {TRIALS} trials/point, flat channel]"
    );
    println!(
        "{:>14} {:>12} {:>14} {:>13}",
        "noise_scale", "equiv_snr_dB", "LDPC+BCH FER", "Conv+RS FER"
    );
    println!("{}", "-".repeat(56));

    for &noise_scale in NOISE_SCALES {
        let a = mean_fer(&ldpc_bch, 1, noise_scale, 0xF00D_0000_0000_0000);
        let b = mean_fer(&conv_rs, 0, noise_scale, 0xBEEF_0000_0000_0000);
        let equiv_snr_db = -10.0 * noise_scale.log10();
        println!("{noise_scale:>14.4} {equiv_snr_db:>12.1} {a:>14.3} {b:>13.3}");
    }
    println!();
    // Always passes -- this is a measurement run, not an assertion.
}
