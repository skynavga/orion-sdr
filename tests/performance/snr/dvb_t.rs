// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// BER / frame-decode-vs-SNR characterization for the conformant DVB-T frame
// (modulate::dvb_t_frame ↔ demodulate::dvb_t_frame): TS payload + energy
// dispersal + RS(204,188) + K=7 conv + Forney interleaver + Figure-9a
// soft-decision + four-phase scattered pilots + TPS + guard-interval
// acquisition.
//
// Feature-gated (`--features throughput`). These are CHARACTERIZATION runs, not
// pass/fail assertions. Run with --nocapture to see the printed table. SNR is
// defined per complex sample (Es/N0-like) relative to the frame's own average
// power, swept in dB.
//
// KNOWN LIMITATION (candidate Batch-5 optimization, documented in
// docs/performance.md): a small payload is padded to a full 68-symbol TPS frame,
// so the real coded data occupies only ~5–13 of 68 symbols (the rest are zero).
// With so few data-bearing symbols, the per-symbol scattered-pilot equalizer's
// band-edge residual (data carriers beyond the outermost scattered pilot hold
// the nearest pilot's estimate) can tank a whole decode, making frame-decode%
// erratic vs SNR and NOT ordered by nominal robustness (QPSK r1/2 can trail
// 16-QAM r3/4). Acquisition and TPS are unaffected — verified perfect (30/30
// exact symbol-offset lock, no TPS failures) at these SNRs; the effect is purely
// in the heavily-padded payload FEC path. Fixes: multi-codeword frame packing
// (less padding) and/or an MMSE / band-edge-aware equalizer.

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
        // GI 1/8 (cp_len = 256). At the shorter GI 1/32 (cp_len = 64) the effect
        // described in the module header is worse. Note: this is NOT an
        // acquisition issue — GI-sync locks the exact symbol offset regardless of
        // GI (verified 30/30); the longer GI simply gives more per-symbol energy
        // and a stronger equalizer estimate on the few data-bearing symbols. See
        // the module header and docs/performance.md.
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

#[test]
fn dvb_t_frame_ber_vs_snr_qpsk_r12() {
    // The robust config: QPSK rate 1/2 — the 333 kHz-class baseline.
    let p = params(ConstellationOrder::Qpsk, PunctureRate::R1_2);
    let payload: Vec<u8> = (0..184).map(|i| ((i * 37 + 11) & 0xff) as u8).collect();

    println!("\n[DVB-T conformant frame — QPSK r1/2, GI 1/8, {TRIALS} trials/pt]");
    println!("  SNR(dB)  frame-decode%   payload-BER");
    let mut any_decode = false;
    for &snr in SNR_DB_POINTS {
        let (success, ber) = measure_at_snr(p, &payload, snr, 0x5EED_0000);
        println!("  {snr:5.1}    {:6.1}%       {ber:.2e}", success * 100.0);
        any_decode |= success > 0.0;
    }
    // Characterization only: the sole invariant is that the frame decodes at all
    // somewhere in the sweep (guards a gross regression that breaks decode
    // entirely). The frame-decode% shape is the known padded-short-frame effect.
    assert!(
        any_decode,
        "QPSK r1/2 must decode somewhere in the SNR sweep"
    );
}

#[test]
fn dvb_t_frame_ber_vs_snr_qam16_r34() {
    // A denser config: 16-QAM rate 3/4 — the 2 MHz-class high-rate mode; needs a
    // higher SNR than QPSK r1/2 (characterization; no low-SNR assertion).
    let p = params(ConstellationOrder::Qam16, PunctureRate::R3_4);
    let payload: Vec<u8> = (0..184).map(|i| ((i * 29 + 7) & 0xff) as u8).collect();

    println!("\n[DVB-T conformant frame — 16-QAM r3/4, GI 1/8, {TRIALS} trials/pt]");
    println!("  SNR(dB)  frame-decode%   payload-BER");
    let mut any_decode = false;
    for &snr in SNR_DB_POINTS {
        let (success, ber) = measure_at_snr(p, &payload, snr, 0x1600_0000);
        println!("  {snr:5.1}    {:6.1}%       {ber:.2e}", success * 100.0);
        any_decode |= success > 0.0;
    }
    assert!(
        any_decode,
        "16-QAM r3/4 must decode somewhere in the SNR sweep"
    );
}
