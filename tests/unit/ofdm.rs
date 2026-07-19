// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use num_complex::Complex32 as C32;
use orion_sdr::core::Block;
use orion_sdr::demodulate::{
    EqualizerMethod, OfdmDecider, OfdmDemod, OfdmEqualizer, OfdmSoftDemod, build_ofdm_rx_frame,
};
use orion_sdr::modulate::{ConstellationOrder, OfdmConfig, OfdmMod};
use orion_sdr::multicarrier::{CarrierPlan, FftBlock};
use orion_sdr::sync::{OfdmPreamble, generate_ofdm_preamble};
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

#[test]
fn ofdm_equalizer_corrects_known_static_channel() {
    let n_fft = 16;
    let cp_len = 4;
    let cfg = qpsk_config(n_fft, cp_len, 48_000.0, 0.0);
    let preamble = OfdmPreamble::new(4, 4).with_training_symbol(n_fft, cp_len);

    // A synthetic per-bin static channel: distinct gain/phase per bin so a
    // no-op (identity) equalizer would clearly fail this test.
    let channel: Vec<C32> = (0..n_fft)
        .map(|k| {
            let mag = 0.3 + 0.05 * k as f32;
            let phase = 0.2 * k as f32;
            C32::from_polar(mag, phase)
        })
        .collect();

    // Run the training symbol through the synthetic channel, FFT it, and
    // estimate.
    let training_iq = generate_ofdm_preamble(&preamble, &cfg);
    let training_start = preamble.num_repeats * preamble.repeat_len;
    let training_symbol = &training_iq[training_start..training_start + n_fft + cp_len];
    let training_time = &training_symbol[cp_len..];
    let channeled_training: Vec<C32> = apply_bin_channel(training_time, &channel, n_fft);

    let mut fft = FftBlock::new(n_fft);
    let mut training_freq = vec![C32::default(); n_fft];
    fft.process(&channeled_training, &mut training_freq);

    let mut eq = OfdmEqualizer::new(&cfg, EqualizerMethod::TrainingSymbolHold);
    assert_eq!(eq.method(), EqualizerMethod::TrainingSymbolHold);
    eq.estimate_from_training_symbol(&training_freq);

    // Now run a data symbol through the SAME channel and confirm the
    // equalizer recovers (approximately) the original spectrum.
    let bits = vec![1u8; cfg.bits_per_ofdm_symbol()];
    let mut modstage = OfdmMod::new(&cfg);
    let mut data_iq = vec![C32::default(); cfg.samples_per_ofdm_symbol()];
    modstage.process(&bits, &mut data_iq);
    let data_time: Vec<C32> = data_iq[cp_len..].to_vec();
    let channeled_data = apply_bin_channel(&data_time, &channel, n_fft);

    let mut clean_freq = vec![C32::default(); n_fft];
    fft.process(&data_time, &mut clean_freq);
    let mut channeled_freq = vec![C32::default(); n_fft];
    fft.process(&channeled_data, &mut channeled_freq);

    let mut equalized = vec![C32::default(); n_fft];
    eq.process(&channeled_freq, &mut equalized);

    let eps = 0.05f32;
    for bin in 0..n_fft {
        assert!(
            (equalized[bin] - clean_freq[bin]).norm() < eps,
            "bin {} not corrected: got {:?}, expected {:?}",
            bin,
            equalized[bin],
            clean_freq[bin]
        );
    }
}

#[test]
fn ofdm_equalizer_interp_between_pilots() {
    let n_fft = 16;
    let cp_len = 4;
    // A handful of data carriers with pilots spread across the band, so
    // interpolation between them is exercised for the data bins in between.
    let plan = CarrierPlan::new(n_fft, cp_len)
        .with_data_carriers([1, 2, 3, 5, 6, 7])
        .with_pilot_carriers([(4i32, C32::new(1.0, 0.0)), (8i32, C32::new(1.0, 0.0))]);
    let cfg = OfdmConfig::new(plan, 48_000.0, 0.0, 1.0, ConstellationOrder::Qpsk);

    // Linear-in-bin-index channel gain between the two pilots (bins 4 and 8)
    // so exact linear interpolation should recover it closely at bins 5..7.
    let channel: Vec<C32> = (0..n_fft)
        .map(|k| C32::new(1.0 + 0.1 * k as f32, 0.0))
        .collect();

    let mut eq = OfdmEqualizer::new(&cfg, EqualizerMethod::PerSymbolPilotInterp);
    assert_eq!(eq.method(), EqualizerMethod::PerSymbolPilotInterp);

    // A frequency-domain vector where every bin already carries the known
    // pilot/data value pre-channel, then apply the synthetic channel
    // directly in the frequency domain (equivalent to a static per-bin
    // channel for this test's purposes).
    let mut freq = vec![C32::new(1.0, 0.0); n_fft];
    for bin in 0..n_fft {
        freq[bin] *= channel[bin];
    }

    let mut equalized = vec![C32::default(); n_fft];
    eq.process(&freq, &mut equalized);

    let eps = 0.15f32;
    for &bin in &[5usize, 6, 7] {
        assert!(
            (equalized[bin] - C32::new(1.0, 0.0)).norm() < eps,
            "bin {} not corrected via pilot interpolation: got {:?}",
            bin,
            equalized[bin]
        );
    }
}

/// Applies a per-bin frequency-domain channel to a time-domain symbol:
/// FFT, multiply by `channel[bin]`, IFFT back to time domain.
fn apply_bin_channel(time: &[C32], channel: &[C32], n_fft: usize) -> Vec<C32> {
    use orion_sdr::multicarrier::IfftBlock;

    let mut fft = FftBlock::new(n_fft);
    let mut freq = vec![C32::default(); n_fft];
    fft.process(time, &mut freq);
    for bin in 0..n_fft {
        freq[bin] *= channel[bin];
    }
    let mut ifft = IfftBlock::new(n_fft);
    let mut out = vec![C32::default(); n_fft];
    ifft.process(&freq, &mut out);
    out
}

fn ofdm_soft_llr_sign_matches_hard_decision_for(constellation: ConstellationOrder) {
    let n_fft = 64;
    let cp_len = 8;
    let plan = CarrierPlan::new(n_fft, cp_len).with_data_carriers(1..32);
    let cfg = OfdmConfig::new(plan, 48_000.0, 0.0, 1.0, constellation);
    let bps = cfg.bits_per_ofdm_symbol();

    // A mixed bit pattern (not all-0/all-1) so every axis exercises a
    // variety of constellation points, not just the outermost ones.
    let bits_in: Vec<u8> = (0..bps).map(|i| ((i * 5 + i / 3) & 1) as u8).collect();

    let mut modstage = OfdmMod::new(&cfg);
    let iq = modstage.modulate(&bits_in);

    let mut demod = OfdmDemod::new(&cfg);
    let num_data = demod.num_data_carriers();
    let mut soft = vec![C32::default(); num_data];
    demod.process(&iq, &mut soft);

    let mut decider = OfdmDecider::new(&cfg);
    let mut bits_hard = vec![0u8; bps];
    decider.process(&soft, &mut bits_hard);

    let mut soft_demod = OfdmSoftDemod::new(&cfg);
    let mut llrs = vec![0.0f32; bps];
    let wr = soft_demod.process(&soft, &mut llrs);
    assert_eq!(wr.in_read, num_data);
    assert_eq!(wr.out_written, bps);

    for i in 0..bps {
        let expected_bit = bits_hard[i];
        let llr_sign_bit = u8::from(llrs[i] < 0.0);
        assert_eq!(
            llr_sign_bit, expected_bit,
            "{:?} bit {}: LLR {} sign disagrees with hard decision {}",
            constellation, i, llrs[i], expected_bit
        );
    }
}

#[test]
fn ofdm_soft_llr_sign_matches_hard_decision() {
    for &order in &[
        ConstellationOrder::Bpsk,
        ConstellationOrder::Qpsk,
        ConstellationOrder::Qam16,
        ConstellationOrder::Qam64,
        ConstellationOrder::Qam256,
    ] {
        ofdm_soft_llr_sign_matches_hard_decision_for(order);
    }
}
