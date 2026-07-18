// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use num_complex::Complex32 as C32;
use orion_sdr::core::Block;
use orion_sdr::demodulate::{OfdmDecider, OfdmDemod, build_ofdm_rx_frame};
use orion_sdr::modulate::{ConstellationOrder, OfdmConfig, OfdmMod};
use orion_sdr::multicarrier::CarrierPlan;
use orion_sdr::util::wb_spectrum_snr_db;
use rustfft::FftPlanner;

fn qpsk_plan(n_fft: usize, cp_len: usize) -> CarrierPlan {
    let half = (n_fft / 2) as i32;
    let data: Vec<i32> = (1..half).chain(-(half - 1)..0).collect();
    CarrierPlan::new(n_fft, cp_len).with_data_carriers(data)
}

fn qpsk_config(n_fft: usize, cp_len: usize, fs: f32, rf_hz: f32) -> OfdmConfig {
    OfdmConfig::new(
        qpsk_plan(n_fft, cp_len),
        fs,
        rf_hz,
        1.0,
        ConstellationOrder::Qpsk,
    )
}

/// Test-local reference FFT: strips the given CP length and runs a plain
/// forward FFT (unity gain) over the remaining n_fft samples. No library RX
/// code exists yet in this release, so this is intentionally independent of
/// `multicarrier::FftBlock`.
fn reference_fft(symbol: &[C32], n_fft: usize, cp_len: usize) -> Vec<C32> {
    let mut buf: Vec<rustfft::num_complex::Complex<f32>> = symbol[cp_len..cp_len + n_fft]
        .iter()
        .map(|c| rustfft::num_complex::Complex::new(c.re, c.im))
        .collect();
    FftPlanner::new().plan_fft_forward(n_fft).process(&mut buf);
    buf.into_iter().map(|c| C32::new(c.re, c.im)).collect()
}

#[test]
fn ofdm_mod_symbol_length() {
    let n_fft = 16;
    let cp_len = 4;
    let cfg = qpsk_config(n_fft, cp_len, 48_000.0, 0.0);
    let mut modstage = OfdmMod::new(&cfg);

    let bits = vec![0u8; cfg.bits_per_ofdm_symbol()];
    let mut out = vec![C32::default(); cfg.samples_per_ofdm_symbol()];

    let wr = modstage.process(&bits, &mut out);
    assert_eq!(wr.in_read, cfg.bits_per_ofdm_symbol());
    assert_eq!(wr.out_written, n_fft + cp_len);
    assert_eq!(cfg.samples_per_ofdm_symbol(), n_fft + cp_len);
}

#[test]
fn ofdm_mod_partial_bits_is_noop() {
    let cfg = qpsk_config(16, 4, 48_000.0, 0.0);
    let mut modstage = OfdmMod::new(&cfg);

    let bits = vec![0u8; cfg.bits_per_ofdm_symbol() - 1]; // one bit short
    let mut out = vec![C32::default(); cfg.samples_per_ofdm_symbol()];

    let wr = modstage.process(&bits, &mut out);
    assert_eq!(wr.in_read, 0);
    assert_eq!(wr.out_written, 0);
}

#[test]
fn ofdm_mod_multi_symbol_batch() {
    let cfg = qpsk_config(16, 4, 48_000.0, 0.0);
    let bps = cfg.bits_per_ofdm_symbol();
    let mut modstage = OfdmMod::new(&cfg);

    let n_symbols = 5;
    let bits: Vec<u8> = (0..n_symbols * bps).map(|i| (i & 1) as u8).collect();
    let out = modstage.modulate(&bits);

    assert_eq!(out.len(), n_symbols * cfg.samples_per_ofdm_symbol());
}

#[test]
fn ofdm_mod_null_carriers_are_silent() {
    let n_fft = 16;
    let cp_len = 4;
    let cfg = qpsk_config(n_fft, cp_len, 48_000.0, 0.0);
    let plan = qpsk_plan(n_fft, cp_len);
    let mut modstage = OfdmMod::new(&cfg);

    // All-ones bit pattern gives nonzero, non-degenerate QPSK symbols.
    let bits = vec![1u8; cfg.bits_per_ofdm_symbol()];
    let mut out = vec![C32::default(); cfg.samples_per_ofdm_symbol()];
    modstage.process(&bits, &mut out);

    let freq = reference_fft(&out, n_fft, cp_len);

    let data_bins: std::collections::HashSet<usize> = plan
        .data_carriers()
        .iter()
        .map(|&idx| idx.rem_euclid(n_fft as i32) as usize)
        .collect();

    let eps = 1e-3f32;
    for bin in 0..n_fft {
        if !data_bins.contains(&bin) {
            assert!(
                freq[bin].norm() < eps,
                "null bin {} not silent: {:?}",
                bin,
                freq[bin]
            );
        } else {
            assert!(
                freq[bin].norm() > eps,
                "data bin {} unexpectedly silent",
                bin
            );
        }
    }
}

#[test]
fn ofdm_mod_cp_matches_symbol_tail() {
    let n_fft = 16;
    let cp_len = 4;
    let cfg = qpsk_config(n_fft, cp_len, 48_000.0, 0.0);
    let mut modstage = OfdmMod::new(&cfg);

    let bits = vec![1u8; cfg.bits_per_ofdm_symbol()];
    let mut out = vec![C32::default(); cfg.samples_per_ofdm_symbol()];
    modstage.process(&bits, &mut out);

    assert_eq!(&out[..cp_len], &out[n_fft..n_fft + cp_len]);
}

#[test]
fn ofdm_mod_rf_upconversion_shifts_spectrum() {
    // A narrow cluster of active subcarriers well clear of DC, so the
    // occupied band is a small, unambiguous slice of the full spectrum both
    // at baseband and after upconversion.
    let n_fft = 256;
    let cp_len = 16;
    let fs = 48_000.0f32;
    let subcarrier_hz = fs / n_fft as f32;
    let rf_hz = 12_000.0f32;

    let active: Vec<i32> = (20..28).collect(); // 8 adjacent carriers
    let plan = CarrierPlan::new(n_fft, cp_len).with_data_carriers(active.clone());
    let cfg = OfdmConfig::new(plan, fs, rf_hz, 1.0, ConstellationOrder::Qpsk);
    let mut modstage = OfdmMod::new(&cfg);

    let n_symbols = 8;
    let bits: Vec<u8> = (0..n_symbols * cfg.bits_per_ofdm_symbol())
        .map(|i| ((i / 3) & 1) as u8)
        .collect();
    let iq = modstage.modulate(&bits);
    let real: Vec<f32> = iq.iter().map(|c| c.re).collect();

    // Occupied band: 8 subcarriers wide, centered at rf_hz + the cluster's
    // baseband center (carriers 20..27 -> center ~23.5 subcarrier spacings).
    let cluster_center_hz = 23.5 * subcarrier_hz;
    let occupied_hz = (active.len() as f32 + 2.0) * subcarrier_hz;
    let carrier_hz = rf_hz + cluster_center_hz;

    let snr = wb_spectrum_snr_db(&real, fs, carrier_hz, occupied_hz);
    assert!(
        snr > 6.0,
        "expected energy concentrated near {:.1} Hz (rf_hz={} + cluster offset), got SNR {:.2} dB",
        carrier_hz,
        rf_hz,
        snr
    );

    // Sanity: without upconversion (rf_hz=0), the same cluster should show
    // up at baseband instead, confirming the shift is really due to rf_hz.
    let cfg_bb = OfdmConfig::new(
        CarrierPlan::new(n_fft, cp_len).with_data_carriers(active),
        fs,
        0.0,
        1.0,
        ConstellationOrder::Qpsk,
    );
    let mut modstage_bb = OfdmMod::new(&cfg_bb);
    let iq_bb = modstage_bb.modulate(&bits);
    let real_bb: Vec<f32> = iq_bb.iter().map(|c| c.re).collect();
    let snr_bb_at_rf = wb_spectrum_snr_db(&real_bb, fs, carrier_hz, occupied_hz);
    assert!(
        snr_bb_at_rf < snr,
        "baseband signal should NOT show concentrated energy at the RF offset: {:.2} dB vs {:.2} dB",
        snr_bb_at_rf,
        snr
    );
}

#[test]
fn ofdm_demod_symbol_length() {
    let n_fft = 16;
    let cp_len = 4;
    let cfg = qpsk_config(n_fft, cp_len, 48_000.0, 0.0);

    let mut modstage = OfdmMod::new(&cfg);
    let bits = vec![1u8; cfg.bits_per_ofdm_symbol()];
    let iq = modstage.modulate(&bits);

    let mut demod = OfdmDemod::new(&cfg);
    let mut soft = vec![C32::default(); demod.num_data_carriers()];
    let wr = demod.process(&iq, &mut soft);

    assert_eq!(wr.in_read, cfg.samples_per_ofdm_symbol());
    assert_eq!(wr.out_written, demod.num_data_carriers());
    assert_eq!(
        demod.num_data_carriers(),
        qpsk_plan(n_fft, cp_len).data_carriers().len()
    );
}

#[test]
fn ofdm_demod_partial_chunk_is_noop() {
    let cfg = qpsk_config(16, 4, 48_000.0, 0.0);
    let mut demod = OfdmDemod::new(&cfg);

    let iq = vec![C32::default(); cfg.samples_per_ofdm_symbol() - 1]; // one sample short
    let mut soft = vec![C32::default(); demod.num_data_carriers()];

    let wr = demod.process(&iq, &mut soft);
    assert_eq!(wr.in_read, 0);
    assert_eq!(wr.out_written, 0);
}

#[test]
fn ofdm_rx_frame_evm_present_cfo_absent() {
    let n_fft = 16;
    let cp_len = 4;
    let cfg = qpsk_config(n_fft, cp_len, 48_000.0, 0.0);
    let bps = cfg.bits_per_ofdm_symbol();
    let n_symbols = 4;

    let bits_in: Vec<u8> = (0..n_symbols * bps).map(|i| (i & 1) as u8).collect();
    let mut modstage = OfdmMod::new(&cfg);
    let iq = modstage.modulate(&bits_in);

    let mut demod = OfdmDemod::new(&cfg);
    let mut decider = OfdmDecider::new(&cfg);
    let num_data = demod.num_data_carriers();
    let samples_per_symbol = cfg.samples_per_ofdm_symbol();

    let mut soft_all = vec![C32::default(); n_symbols * num_data];
    let mut bits_out = vec![0u8; bits_in.len()];
    let mut in_off = 0usize;
    let mut out_off = 0usize;
    while in_off + samples_per_symbol <= iq.len() {
        let mut soft_sym = vec![C32::default(); num_data];
        demod.process(&iq[in_off..], &mut soft_sym);
        soft_all[out_off / bps * num_data..out_off / bps * num_data + num_data]
            .copy_from_slice(&soft_sym);
        decider.process(&soft_sym, &mut bits_out[out_off..]);
        in_off += samples_per_symbol;
        out_off += bps;
    }

    let frame = build_ofdm_rx_frame(&cfg, &soft_all, bits_out.clone());

    assert_eq!(frame.bits, bits_out);
    assert_eq!(frame.num_symbols, n_symbols);
    assert!(
        frame.evm_db.is_some(),
        "evm_db should be populated in this release"
    );
    assert!(
        frame.evm_db.unwrap() < -20.0,
        "expected low EVM for a noiseless roundtrip, got {:?} dB",
        frame.evm_db
    );
    assert!(
        frame.cfo_hz.is_none(),
        "cfo_hz should be None until acquisition lands"
    );
    assert!(
        frame.timing_offset_samples.is_none(),
        "timing_offset_samples should be None until acquisition lands"
    );
    assert!(
        frame.channel_mse.is_none(),
        "channel_mse should be None until equalization lands"
    );
}
