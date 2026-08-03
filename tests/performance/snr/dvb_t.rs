// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// BER / frame-decode-vs-SNR characterization for the conformant DVB-T frame
// (modulate::dvb_t_frame ↔ demodulate::dvb_t_frame): TS payload + energy
// dispersal + RS(204,188) + K=7 conv + Forney interleaver + Figure-9a
// soft-decision + four-phase scattered pilots + TPS + guard-interval
// acquisition.
//
// Feature-gated (`--features throughput`). Run with --nocapture to see the
// printed table. SNR is defined per complex sample (Es/N0-like) relative to the
// frame's own average power, swept in dB.
//
// These runs print a full decode-vs-SNR curve AND carry a coarse robustness-
// ordering assertion: the robust QPSK r1/2 config must decode at a low SNR where
// the denser 16-QAM r3/4 config does not yet, and both must reach a full lock by
// a moderate SNR. The equalizer must exclude the 17 TPS carriers from its channel-
// reference set (the modulator transmits data-power DBPSK on them, not the boosted
// `w_k` pilot value), or the interpolated estimate is corrupted around each TPS
// carrier; the noiseless guard for that is
// `roundtrip::dvb_t::dvb_t_equalizer_noiseless_clean_*`.

use crate::common::add_awgn;
use num_complex::Complex32 as C32;
use orion_sdr::demodulate::dvb_t_frame_demodulate;
use orion_sdr::fec::PunctureRate;
use orion_sdr::modulate::{ConstellationOrder, dvb_t_frame_modulate};
use orion_sdr::waveform::dvb_t::{DvbTFrameParams, GuardInterval};

const TRIALS: usize = 30;
const SNR_DB_POINTS: &[f32] = &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0];

fn params(constellation: ConstellationOrder, code_rate: PunctureRate) -> DvbTFrameParams {
    DvbTFrameParams {
        // GI 1/8 (cp_len = 256). GI-sync locks the exact symbol offset regardless
        // of guard interval (verified 30/30).
        guard: GuardInterval::G1_8,
        constellation,
        code_rate,
        frame_number: 0,
        cell_id: 0,
    }
}

/// Runs `TRIALS` AWGN draws at `snr_db` and returns
/// `(frame_decode_success_rate, mean_payload_ber_over_successful)`.
fn measure_at_snr(
    params: DvbTFrameParams,
    payload: &[u8],
    snr_db: f32,
    seed_base: u64,
) -> (f32, f32) {
    let frame = dvb_t_frame_modulate(params, payload);
    let sig_power: f32 = frame.iq.iter().map(|s| s.norm_sqr()).sum::<f32>() / frame.iq.len() as f32;
    let noise_power = sig_power / 10f32.powf(snr_db / 10.0);

    let mut decoded = 0usize;
    let mut err_bits = 0usize;
    let mut tot_bits = 0usize;

    for trial in 0..TRIALS {
        let mut buf = vec![C32::default(); 200];
        buf.extend_from_slice(&frame.iq);
        buf.extend(vec![C32::default(); frame.samples_per_symbol]);
        let seed = seed_base
            .wrapping_add(trial as u64)
            .wrapping_add((snr_db * 1000.0) as u64);
        // Add noise only over the frame body (not the lead/trail silence), so the
        // SNR is defined against the signal, not the padding.
        add_awgn(&mut buf[200..200 + frame.iq.len()], noise_power, seed);

        if let Ok(rx) = dvb_t_frame_demodulate(params, &buf, frame.n_symbols, payload.len()) {
            decoded += 1;
            for (a, b) in rx.payload.iter().zip(payload.iter()) {
                err_bits += (a ^ b).count_ones() as usize;
            }
            tot_bits += payload.len() * 8;
        }
    }

    let success = decoded as f32 / TRIALS as f32;
    let ber = if tot_bits > 0 {
        err_bits as f32 / tot_bits as f32
    } else {
        0.5 // no frame decoded → treat as worst-case
    };
    (success, ber)
}

/// Sweeps `SNR_DB_POINTS`, prints the decode-vs-SNR curve, and returns the
/// frame-decode fraction at each point (parallel to `SNR_DB_POINTS`).
fn sweep(label: &str, p: DvbTFrameParams, payload: &[u8], seed: u64) -> Vec<f32> {
    println!("\n[DVB-T conformant frame — {label}, GI 1/8, {TRIALS} trials/pt]");
    println!("  SNR(dB)  frame-decode%   payload-BER");
    let mut decode = Vec::with_capacity(SNR_DB_POINTS.len());
    for &snr in SNR_DB_POINTS {
        let (success, ber) = measure_at_snr(p, payload, snr, seed);
        println!("  {snr:5.1}    {:6.1}%       {ber:.2e}", success * 100.0);
        decode.push(success);
    }
    decode
}

/// Decode fraction at the sweep point nearest `snr_db`.
fn at_snr(decode: &[f32], snr_db: f32) -> f32 {
    let i = SNR_DB_POINTS
        .iter()
        .position(|&s| (s - snr_db).abs() < 0.5)
        .expect("SNR point present in SNR_DB_POINTS");
    decode[i]
}

#[test]
fn dvb_t_frame_ber_vs_snr_qpsk_r12() {
    // The robust config: QPSK rate 1/2 — the 333 kHz-class baseline. It decodes
    // essentially from the bottom of the sweep; require a clean lock by a low SNR.
    let p = params(ConstellationOrder::Qpsk, PunctureRate::R1_2);
    let payload: Vec<u8> = (0..184).map(|i| ((i * 37 + 11) & 0xff) as u8).collect();
    let decode = sweep("QPSK r1/2", p, &payload, 0x5EED_0000);
    assert!(
        at_snr(&decode, 4.0) >= 0.9,
        "QPSK r1/2 must decode ≥90% by 4 dB (was {:.0}%)",
        at_snr(&decode, 4.0) * 100.0
    );
}

#[test]
fn dvb_t_frame_ber_vs_snr_qam16_r34() {
    // A denser config: 16-QAM rate 3/4 — the 2 MHz-class high-rate mode. Correctly
    // LESS robust than QPSK r1/2: it must reach a full lock by a moderate SNR but
    // not already be locked at the low SNR where QPSK r1/2 is (robustness ordering).
    let p = params(ConstellationOrder::Qam16, PunctureRate::R3_4);
    let payload: Vec<u8> = (0..184).map(|i| ((i * 29 + 7) & 0xff) as u8).collect();
    let decode = sweep("16-QAM r3/4", p, &payload, 0x1600_0000);
    assert!(
        at_snr(&decode, 10.0) >= 0.9,
        "16-QAM r3/4 must decode ≥90% by 10 dB (was {:.0}%)",
        at_snr(&decode, 10.0) * 100.0
    );

    // Robustness ordering: at a low SNR the robust QPSK r1/2 decodes essentially
    // fully while the denser 16-QAM r3/4 does not yet.
    let qpsk = params(ConstellationOrder::Qpsk, PunctureRate::R1_2);
    let qpsk_payload: Vec<u8> = (0..184).map(|i| ((i * 37 + 11) & 0xff) as u8).collect();
    let qpsk_decode = sweep(
        "QPSK r1/2 (ordering check)",
        qpsk,
        &qpsk_payload,
        0x5EED_0000,
    );
    let lo = 4.0;
    assert!(
        at_snr(&qpsk_decode, lo) > at_snr(&decode, lo),
        "at {lo} dB QPSK r1/2 ({:.0}%) must out-decode 16-QAM r3/4 ({:.0}%)",
        at_snr(&qpsk_decode, lo) * 100.0,
        at_snr(&decode, lo) * 100.0
    );
}
