// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// Frame-error-rate-vs-SNR characterization for the COFDM **streaming**
// receiver, `OfdmFrameStreamDemod` — acquisition, equalization and residual
// carrier tracking included.
//
// Feature-gated (`--features throughput`). Tests always pass -- they are
// measurement / characterisation runs, not assertions. Run with --nocapture
// to see the printed table.
//
// Backs the "COFDM streaming receiver frame-error rate" table in
// docs/performance.md.
//
// **Why this exists beside `snr::cofdm_fer`.** That sweep decodes with the
// *batch* demodulator at a known frame start, so it measures the concatenated
// FEC and nothing else. This one measures the link a caller actually gets. The
// gap between the two is the receiver's own overhead, and it used to be
// enormous: the Schmidl & Cox carrier estimate has variance, `TrainingSymbolHold`
// holds one channel estimate for the whole frame, and the residual therefore
// integrated into constellation rotation that nothing corrected. Frames failed
// 15-20 dB above the FEC's own limit, and the failures looked exactly like an
// FEC cliff. `remove_common_phase_error` closed most of that gap; this table is
// what keeps it closed.

use crate::common::add_awgn;
use num_complex::Complex32 as C32;
use orion_sdr::demodulate::OfdmFrameStreamDemod;
use orion_sdr::fec::{CrcKind, FrameMetadata, FramePacket};
use orion_sdr::modulate::{ConstellationOrder, McsTable, OfdmConfig, OfdmFrameMod};
use orion_sdr::multicarrier::CarrierPlan;
use orion_sdr::sync::OfdmPreamble;

const TRIALS: usize = 60;
const FRAME_N: usize = 64; // n_fft
const FRAME_CP: usize = 8;
const PAYLOAD_LEN: usize = 96;

/// In-band SNR points, in dB, spanning the region where the streaming receiver
/// used to fail and no longer does.
const SNR_DB: &[f32] = &[6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0];

/// The same plan `snr::cofdm_fer` and the COFDM throughput benchmarks use, so
/// all three tables describe one link.
fn frame_config() -> OfdmConfig {
    let half = (FRAME_N / 2) as i32;
    let data: Vec<i32> = (1..half).chain(-(half - 1)..0).collect();
    let plan = CarrierPlan::new(FRAME_N, FRAME_CP).with_data_carriers(data);
    OfdmConfig::new(plan, 48_000.0, 0.0, 1.0, ConstellationOrder::Bpsk)
        .with_payload_crc(CrcKind::Crc32)
        .with_header_crc(CrcKind::Crc16)
}

fn frame_preamble(cfg: &OfdmConfig) -> OfdmPreamble {
    OfdmPreamble::new(4, 16)
        .with_training_symbol(cfg.carrier_plan.n_fft(), cfg.carrier_plan.cp_len())
}

#[test]
fn snr_sweep_cofdm_stream_frame_error_rate() {
    let cfg = frame_config();
    let pre = frame_preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let payload: Vec<u8> = (0..PAYLOAD_LEN)
        .map(|i| ((i * 37 + 11) & 0xff) as u8)
        .collect();
    let frame = FramePacket::new(FrameMetadata::new(1, 1), payload.clone());
    let clean = modu.modulate_frame(&frame, 0);

    // Noise referenced to the PAYLOAD, not the buffer mean: the preamble is
    // deliberately hotter, so a buffer-mean reference injects more noise than
    // the nominal figure claims.
    let body = &clean[pre.total_len()..];
    let body_power: f32 = body.iter().map(|c| c.norm_sqr()).sum::<f32>() / body.len() as f32;
    let frame_ms = clean.len() as f32 / cfg.fs * 1.0e3;

    println!(
        "\n[COFDM streaming-receiver FER, n_fft={FRAME_N}, cp_len={FRAME_CP}, QPSK \
         payload, {PAYLOAD_LEN}-byte payload, {frame_ms:.1} ms frame, {TRIALS} \
         trials/point, flat channel]"
    );
    println!("{:>10} {:>10} {:>14}", "in-band SNR", "FER", "mean CBER");
    println!("{}", "-".repeat(36));

    for &snr_db in SNR_DB {
        let noise_power = body_power * 10f32.powf(-snr_db / 10.0);
        let mut errors = 0usize;
        let (mut cber_sum, mut cber_n) = (0.0f64, 0usize);
        for trial in 0..TRIALS {
            let mut rx_iq: Vec<C32> = clean.clone();
            add_awgn(&mut rx_iq, noise_power, 0xC0FD_0000 + trial as u64);
            rx_iq.extend(std::iter::repeat_n(C32::default(), 256));

            let mut demod =
                OfdmFrameStreamDemod::new(cfg.clone(), table.clone(), frame_preamble(&cfg))
                    .with_error_rates(true);
            let out = demod.feed(&rx_iq);
            let ok = out
                .iter()
                .any(|r| matches!(r, Ok(f) if f.packet.payload == payload));
            if !ok {
                errors += 1;
            }
            for f in out.iter().flatten() {
                if let Some(b) = f.diagnostics.channel_ber {
                    cber_sum += f64::from(b);
                    cber_n += 1;
                }
            }
        }
        let cber = if cber_n > 0 {
            cber_sum / cber_n as f64
        } else {
            f64::NAN
        };
        println!(
            "{snr_db:>10.0} {:>10.3} {cber:>14.5}",
            errors as f32 / TRIALS as f32
        );
    }
    println!();
    // Always passes -- this is a measurement run, not an assertion.
}
