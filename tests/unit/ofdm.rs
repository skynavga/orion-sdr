// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use num_complex::Complex32 as C32;
use orion_sdr::core::Block;
use orion_sdr::demodulate::{
    EqualizerMethod, OfdmDecider, OfdmDemod, OfdmEqualizer, OfdmSoftDemod, build_ofdm_rx_frame,
};
use orion_sdr::modulate::{ConstellationOrder, OfdmConfig, OfdmMod};
use orion_sdr::multicarrier::{CarrierPlan, FftBlock, SymbolWindow, TxLowpass};
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
    for (bin, value) in freq.iter().enumerate() {
        if !data_bins.contains(&bin) {
            assert!(
                value.norm() < eps,
                "null bin {} not silent: {:?}",
                bin,
                value
            );
        } else {
            assert!(value.norm() > eps, "data bin {} unexpectedly silent", bin);
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

    // Anchor: the noiseless roundtrip must recover the transmitted bits
    // exactly. Without this, EVM is only self-consistent (soft symbols vs.
    // ideal points re-mapped from the *decided* bits) and a demod that was
    // wrong-but-self-consistent could still show low EVM. Pinning bits_out to
    // bits_in makes the ideal reference the true transmitted constellation.
    assert_eq!(
        bits_out, bits_in,
        "noiseless roundtrip must recover input bits"
    );

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
fn ofdm_rx_frame_evm_matches_known_error_magnitude() {
    // Pin the EVM dB *formula*, not just "very negative". Feed soft symbols
    // that are the ideal constellation points scaled by (1 + e) for a known
    // e, so the per-symbol error vector is exactly e·ideal and
    //   err_energy / ref_energy = e²  =>  evm_db = 10·log10(e²) = 20·log10(e).
    // With e = 0.1 the expected EVM is exactly −20.0 dB. A wrong formula
    // (e.g. 20·log10 of the ratio, or a missing normalization) would miss
    // this, whereas the noiseless-roundtrip test's −20 dB *threshold* would
    // not.
    let n_fft = 16;
    let cp_len = 4;
    let cfg = qpsk_config(n_fft, cp_len, 48_000.0, 0.0);
    let bps = cfg.bits_per_ofdm_symbol();

    let bits: Vec<u8> = (0..bps).map(|i| ((i / 2 + i % 3) & 1) as u8).collect();

    // Recover the ideal soft symbols via a noiseless demod (they sit exactly
    // on the constellation for a flat, noiseless channel).
    let mut modstage = OfdmMod::new(&cfg);
    let iq = modstage.modulate(&bits);
    let mut demod = OfdmDemod::new(&cfg);
    let num_data = demod.num_data_carriers();
    let mut ideal = vec![C32::default(); num_data];
    demod.process(&iq, &mut ideal);

    let e = 0.1f32;
    let scaled: Vec<C32> = ideal.iter().map(|&s| s * (1.0 + e)).collect();

    let frame = build_ofdm_rx_frame(&cfg, &scaled, bits.clone());
    let evm = frame.evm_db.expect("evm_db populated");
    let expected = 20.0 * e.log10(); // = -20.0 dB for e = 0.1
    assert!(
        (evm - expected).abs() < 0.2,
        "EVM {} dB should match 20·log10(e) = {} dB for a known {}× error",
        evm,
        expected,
        e
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
    for (bin, (&got, &want)) in equalized.iter().zip(clean_freq.iter()).enumerate() {
        assert!(
            (got - want).norm() < eps,
            "bin {} not corrected: got {:?}, expected {:?}",
            bin,
            got,
            want
        );
    }
}

#[test]
fn ofdm_equalizer_interp_between_pilots() {
    let n_fft = 16;
    let cp_len = 4;
    // Pilots at bins 2 and 6 (both in-range: valid signed indices for
    // n_fft=16 are -8..=7), with data carriers 3,4,5 strictly between them so
    // the linear `(Some, Some)` interpolation branch is exercised for those
    // bins.
    let plan = CarrierPlan::new(n_fft, cp_len)
        .with_data_carriers([1, 3, 4, 5, 7])
        .with_pilot_carriers([(2i32, C32::new(1.0, 0.0)), (6i32, C32::new(1.0, 0.0))]);
    let cfg = OfdmConfig::new(plan, 48_000.0, 0.0, 1.0, ConstellationOrder::Qpsk);

    // A channel that is linear *in bin index* across the pilot span in both
    // magnitude and phase, so exact linear interpolation of the complex
    // channel ratio should recover it closely at the in-between bins. Using a
    // complex (phase-rotating) channel — not a purely real gain — exercises
    // interpolation of the imaginary part too.
    let channel: Vec<C32> = (0..n_fft)
        .map(|k| C32::from_polar(1.0 + 0.05 * k as f32, 0.03 * k as f32))
        .collect();

    let mut eq = OfdmEqualizer::new(&cfg, EqualizerMethod::PerSymbolPilotInterp);
    assert_eq!(eq.method(), EqualizerMethod::PerSymbolPilotInterp);

    // A frequency-domain vector where every bin already carries the known
    // pilot/data value pre-channel, then apply the synthetic channel
    // directly in the frequency domain (equivalent to a static per-bin
    // channel for this test's purposes).
    let mut freq = vec![C32::new(1.0, 0.0); n_fft];
    for (f, &h) in freq.iter_mut().zip(channel.iter()) {
        *f *= h;
    }

    let mut equalized = vec![C32::default(); n_fft];
    eq.process(&freq, &mut equalized);

    // Between the pilots (bins 3,4,5), linear interpolation of the complex
    // channel ratio is exact, so equalization recovers the pre-channel value.
    let eps = 0.05f32;
    for &bin in &[3usize, 4, 5] {
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
    for (f, &h) in freq.iter_mut().zip(channel.iter()) {
        *f *= h;
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

    // Anchor against ground truth: for a noiseless flat channel the soft
    // symbols sit exactly on the transmitted constellation point, so the hard
    // decision MUST equal the transmitted bits. Checking this first turns the
    // LLR test from a mere soft-vs-hard *consistency* check (which a bug
    // shared by both — e.g. in the common Gray axis table — could pass) into a
    // check against the true bits.
    assert_eq!(
        bits_hard, bits_in,
        "{:?}: noiseless hard decision must recover the transmitted bits",
        constellation
    );

    for (i, (&llr, &bit_in)) in llrs.iter().zip(bits_in.iter()).enumerate() {
        // Positive LLR ⇒ bit more likely 0; so sign bit (llr < 0) is the
        // decided bit and must match the transmitted bit exactly.
        let llr_sign_bit = u8::from(llr < 0.0);
        assert_eq!(
            llr_sign_bit, bit_in,
            "{:?} bit {}: LLR {} sign disagrees with transmitted bit {}",
            constellation, i, llr, bit_in
        );
        // The LLR must also be confidently signed (not a near-zero tie) on a
        // noiseless channel — a magnitude sanity check the pure sign test
        // omitted.
        assert!(
            llr.abs() > 1e-3,
            "{:?} bit {}: LLR {} implausibly close to zero on a noiseless channel",
            constellation,
            i,
            llr
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

#[test]
fn ofdm_mod_applies_tx_gain_and_demod_inverts_it() {
    // TX gain scales the output IQ; RX gain scales the soft symbols. A round
    // trip with TX gain g and RX gain 1/g must recover the same soft symbols
    // as the unity-gain path, and gain=1.0 on both must be identical. This is
    // the only test that drives the non-unity gain branches on either side
    // (OfdmMod::process's rf/gain loop and OfdmDemod's `(g-1).abs()>EPSILON`
    // correction).
    let n_fft = 16;
    let cp_len = 4;
    let cfg = qpsk_config(n_fft, cp_len, 48_000.0, 0.0);
    let bps = cfg.bits_per_ofdm_symbol();
    let bits: Vec<u8> = (0..bps).map(|i| (i & 1) as u8).collect();

    // Reference: unity gain end to end.
    let mut mod_unity = OfdmMod::new(&cfg);
    let iq_unity = mod_unity.modulate(&bits);
    let mut demod_unity = OfdmDemod::new(&cfg);
    let num_data = demod_unity.num_data_carriers();
    let mut soft_ref = vec![C32::default(); num_data];
    demod_unity.process(&iq_unity, &mut soft_ref);

    // TX gain g: the emitted IQ must be exactly g times the unity IQ.
    let g = 3.0f32;
    let mut mod_gain = OfdmMod::new(&cfg);
    mod_gain.set_gain(g);
    let iq_gain = mod_gain.modulate(&bits);
    for (a, b) in iq_gain.iter().zip(iq_unity.iter()) {
        assert!(
            (*a - *b * g).norm() < 1e-4,
            "TX gain not applied: {:?} vs {:?}*{}",
            a,
            b,
            g
        );
    }

    // RX gain 1/g undoes it: soft symbols match the unity-gain reference.
    let mut demod_gain = OfdmDemod::new(&cfg);
    demod_gain.set_gain(1.0 / g);
    let mut soft_corrected = vec![C32::default(); num_data];
    demod_gain.process(&iq_gain, &mut soft_corrected);
    for (a, b) in soft_corrected.iter().zip(soft_ref.iter()) {
        assert!(
            (*a - *b).norm() < 1e-4,
            "RX gain did not invert TX gain: {:?} vs {:?}",
            a,
            b
        );
    }
}

#[test]
fn ofdm_mod_rf_upconversion_applies_gain() {
    // The rf_hz != 0.0 branch of OfdmMod::process has its own gain multiply
    // (separate from the baseband branch); confirm gain scales the
    // upconverted output too.
    let n_fft = 16;
    let cp_len = 4;
    let cfg_g1 = qpsk_config(n_fft, cp_len, 48_000.0, 6_000.0);
    let bps = cfg_g1.bits_per_ofdm_symbol();
    let bits: Vec<u8> = (0..bps).map(|i| ((i / 2) & 1) as u8).collect();

    let mut mod_g1 = OfdmMod::new(&cfg_g1);
    let iq_g1 = mod_g1.modulate(&bits);

    let g = 2.5f32;
    let mut mod_g = OfdmMod::new(&cfg_g1);
    mod_g.set_gain(g);
    let iq_g = mod_g.modulate(&bits);

    for (a, b) in iq_g.iter().zip(iq_g1.iter()) {
        assert!(
            (*a - *b * g).norm() < 1e-4,
            "gain not applied on the RF-upconversion path: {:?} vs {:?}*{}",
            a,
            b,
            g
        );
    }
}

#[test]
fn ifft_dc_bin_scale_is_one_over_n() {
    // Isolated pin on the IFFT's 1/N scale (fft.rs folds `scale = 1/n_fft`
    // into the output copy). Only ever tested via the cancelling FFT->IFFT
    // roundtrip elsewhere, where a compensating error in both would hide.
    // IFFT of a single DC bin [c, 0, 0, ...] must be the constant c/N in
    // every time sample.
    use orion_sdr::multicarrier::IfftBlock;

    let n_fft = 32;
    let c = C32::new(4.0, -2.0);
    let mut freq = vec![C32::default(); n_fft];
    freq[0] = c;

    let mut ifft = IfftBlock::new(n_fft);
    let mut time = vec![C32::default(); n_fft];
    ifft.process(&freq, &mut time);

    let expected = c / n_fft as f32;
    let eps = 1e-5f32;
    for (k, s) in time.iter().enumerate() {
        assert!(
            (*s - expected).norm() < eps,
            "IFFT DC scale wrong at sample {}: got {:?}, expected {:?} (=c/N)",
            k,
            s,
            expected
        );
    }
}

#[test]
fn ofdm_mod_zero_pads_final_partial_symbol() {
    // `modulate` zero-pads a final partial symbol up to a whole
    // bits_per_ofdm_symbol boundary. Feed 1.5 symbols' worth of bits and
    // confirm (a) the output is exactly 2 whole symbols long, and (b) the
    // second symbol equals what modulating the same partial bits explicitly
    // zero-padded produces.
    let cfg = qpsk_config(16, 4, 48_000.0, 0.0);
    let bps = cfg.bits_per_ofdm_symbol();
    let sps = cfg.samples_per_ofdm_symbol();

    let partial = bps + bps / 2; // 1.5 symbols
    let bits: Vec<u8> = (0..partial).map(|i| (i & 1) as u8).collect();

    let mut modstage = OfdmMod::new(&cfg);
    let iq = modstage.modulate(&bits);
    assert_eq!(
        iq.len(),
        2 * sps,
        "partial final symbol should be padded to a whole symbol"
    );

    // Reference: same bits, explicitly zero-padded to 2 whole symbols.
    let mut padded = bits.clone();
    padded.resize(2 * bps, 0);
    let mut mod_ref = OfdmMod::new(&cfg);
    let iq_ref = mod_ref.modulate(&padded);
    assert_eq!(
        iq, iq_ref,
        "zero-padding of the partial symbol is inconsistent"
    );
}

#[test]
fn ofdm_equalizer_pilot_interp_empty_pilots_is_noop() {
    // PerSymbolPilotInterp with zero in-band pilots must leave the held
    // estimate (identity, 1.0+0j) unchanged, per the documented fallback --
    // a valid config for the default-equalizer use case that specifies no
    // pilots. Equalization then passes the input through unchanged.
    let n_fft = 16;
    let cp_len = 4;
    let plan = CarrierPlan::new(n_fft, cp_len).with_data_carriers(1..8);
    let cfg = OfdmConfig::new(plan, 48_000.0, 0.0, 1.0, ConstellationOrder::Qpsk);

    let mut eq = OfdmEqualizer::new(&cfg, EqualizerMethod::PerSymbolPilotInterp);

    // An arbitrary (non-unit) frequency-domain vector. With no pilots the
    // equalizer divides by the identity estimate, so output == input.
    let freq: Vec<C32> = (0..n_fft)
        .map(|k| C32::new(0.5 + k as f32, -(k as f32)))
        .collect();
    let mut equalized = vec![C32::default(); n_fft];
    eq.process(&freq, &mut equalized);

    for (bin, (&got, &want)) in equalized.iter().zip(freq.iter()).enumerate() {
        assert!(
            (got - want).norm() < 1e-5,
            "empty-pilot interp should be a pass-through at bin {}: {:?} vs {:?}",
            bin,
            got,
            want
        );
    }
}

#[test]
fn ofdm_equalizer_pilot_interp_extrapolates_outside_pilot_span() {
    // Data bins that fall outside the [min pilot, max pilot] span exercise
    // the nearest-pilot fallback (a data bin below the lowest pilot or above
    // the highest), which the between-pilots test never reaches. With a
    // single distinct value per pilot, a bin outside the span takes the
    // nearest pilot's channel ratio.
    let n_fft = 16;
    let cp_len = 4;
    // Pilots at bins 3 and 6; data at 1 (below the span) and 7 (above it),
    // plus 4,5 inside. All in-range for n_fft=16 (-8..=7).
    let plan = CarrierPlan::new(n_fft, cp_len)
        .with_data_carriers([1, 4, 5, 7])
        .with_pilot_carriers([(3i32, C32::new(1.0, 0.0)), (6i32, C32::new(1.0, 0.0))]);
    let cfg = OfdmConfig::new(plan, 48_000.0, 0.0, 1.0, ConstellationOrder::Qpsk);

    // A per-bin channel that is constant across the whole band, so both the
    // interpolated and the nearest-pilot-extrapolated estimates equal that
    // constant and equalization recovers the pre-channel value everywhere,
    // including the out-of-span bins 1 and 7.
    let h = C32::from_polar(0.7, 0.4);
    let freq: Vec<C32> = (0..n_fft).map(|_| h).collect();

    let mut eq = OfdmEqualizer::new(&cfg, EqualizerMethod::PerSymbolPilotInterp);
    let mut equalized = vec![C32::default(); n_fft];
    eq.process(&freq, &mut equalized);

    let eps = 1e-4f32;
    for &bin in &[1usize, 4, 5, 7] {
        assert!(
            (equalized[bin] - C32::new(1.0, 0.0)).norm() < eps,
            "data bin {} not equalized (out-of-span fallback): {:?}",
            bin,
            equalized[bin]
        );
    }
}

// ── Edge-carrier guard band: out-of-band emission (Track A, R4) ─────────────

/// Full complex-baseband power spectrum in natural rustfft bin order (bin 0 =
/// DC, negative freqs in the upper half). `util::power_spectrum` is real-input
/// (one-sided) and would fold negative onto positive frequencies — wrong for a
/// complex OFDM signal — so this test measures the complex spectrum directly.
fn complex_power_db(samples: &[C32], n_fft: usize) -> Vec<f32> {
    let mut buf: Vec<rustfft::num_complex::Complex<f32>> = (0..n_fft)
        .map(|i| {
            let s = samples.get(i).copied().unwrap_or_default();
            rustfft::num_complex::Complex::new(s.re, s.im)
        })
        .collect();
    FftPlanner::new().plan_fft_forward(n_fft).process(&mut buf);
    buf.iter()
        .map(|c| 10.0 * ((c.re * c.re + c.im * c.im) + 1e-12).log10())
        .collect()
}

/// Mean power (dB) over the out-of-band bins: those signed indices with
/// `|idx| > band_half`, i.e. outside the occupied span. Averaged in the linear
/// domain to reflect actual leaked energy, then returned in dB.
fn mean_oob_power_db(power_db: &[f32], n_fft: usize, band_half: i32) -> f32 {
    let n = n_fft as i32;
    let mut acc = 0.0f64;
    let mut count = 0usize;
    for (bin, &pdb) in power_db.iter().enumerate().take(n_fft) {
        // rustfft bin -> signed index
        let signed = if (bin as i32) <= n / 2 {
            bin as i32
        } else {
            bin as i32 - n
        };
        if signed.abs() > band_half {
            acc += 10f64.powf(pdb as f64 / 10.0);
            count += 1;
        }
    }
    let mean_lin = acc / count.max(1) as f64;
    (10.0 * mean_lin.log10()) as f32
}

#[test]
fn edge_guard_reduces_out_of_band_power() {
    let n_fft = 256usize;
    let cp_len = n_fft / 4;
    let fs = 240_000.0f32;
    let edge_guard = 12usize; // ~5% of n_fft per edge

    // A deterministic bit pattern that exercises many subcarriers.
    let bits: Vec<u8> = (0..4096u32)
        .map(|i| ((i * 2654435761) >> 24) as u8)
        .collect();

    // Baseline: full-fill span (guard 0). Guarded: same but edge_guard nulls.
    let full = CarrierPlan::new(n_fft, cp_len).with_contiguous_data(0, false);
    let guarded = CarrierPlan::new(n_fft, cp_len).with_contiguous_data(edge_guard, false);

    let cfg_full = OfdmConfig::new(full, fs, 0.0, 1.0, ConstellationOrder::Qpsk);
    let cfg_guarded = OfdmConfig::new(guarded, fs, 0.0, 1.0, ConstellationOrder::Qpsk);

    let out_full = OfdmMod::new(&cfg_full).modulate(&bits);
    let out_guarded = OfdmMod::new(&cfg_guarded).modulate(&bits);

    // Analyze one interior symbol (skip the first, to avoid transient edges),
    // dropping its CP so the FFT sees exactly one n_fft-sample OFDM symbol.
    let sps = cfg_full.samples_per_ofdm_symbol();
    let sym_full = &out_full[sps + cp_len..sps + cp_len + n_fft];
    let sym_guarded = &out_guarded[sps + cp_len..sps + cp_len + n_fft];

    let pdb_full = complex_power_db(sym_full, n_fft);
    let pdb_guarded = complex_power_db(sym_guarded, n_fft);

    // Occupied half-width of the GUARDED signal: its outermost data carrier.
    let band_half = (n_fft as i32 / 2 - 1) - edge_guard as i32;

    let oob_full = mean_oob_power_db(&pdb_full, n_fft, band_half);
    let oob_guarded = mean_oob_power_db(&pdb_guarded, n_fft, band_half);

    // The guard should drop mean out-of-band power well below the full-fill
    // case (the outer carriers that were the loudest sinc generators are now
    // null). Demonstrated, not merely asserted: require a clear margin.
    let drop_db = oob_full - oob_guarded;
    assert!(
        drop_db > 10.0,
        "edge guard should cut mean OOB power by >10 dB \
         (full={oob_full:.1} dB, guarded={oob_guarded:.1} dB, drop={drop_db:.1} dB)"
    );
}

// ── Symbol-window roll-off: samples and the two beta conventions (R10) ──────

fn window_cfg(n_fft: usize, cp_len: usize) -> OfdmConfig {
    let plan = qpsk_plan(n_fft, cp_len);
    OfdmConfig::new(plan, 48_000.0, 0.0, 1.0, ConstellationOrder::Qpsk)
}

#[test]
fn symbol_window_roll_off_samples_and_default() {
    let cfg = window_cfg(64, 16);
    assert_eq!(cfg.carrier_plan.window_roll_off(), 0, "default is off");
    let cfg = cfg.with_symbol_window(5);
    assert_eq!(cfg.carrier_plan.window_roll_off(), 5);
}

#[test]
fn symbol_window_beta_guard_is_fraction_of_cp() {
    // beta * cp_len, rounded; beta=0.5 is the max-transparent cp_len/2.
    let n_fft = 64;
    let cp_len = 16;
    let cases = [(0.0f32, 0), (0.25, 4), (0.5, 8), (0.375, 6)];
    for (beta, expect) in cases {
        let cfg = window_cfg(n_fft, cp_len).with_symbol_window_beta_guard(beta);
        assert_eq!(
            cfg.carrier_plan.window_roll_off(),
            expect,
            "beta_guard={beta} -> roll_off"
        );
    }
    // Clamped to 0.5 (never exceeds cp_len/2).
    let cfg = window_cfg(n_fft, cp_len).with_symbol_window_beta_guard(0.9);
    assert_eq!(cfg.carrier_plan.window_roll_off(), cp_len / 2);
}

#[test]
fn symbol_window_beta_tu_is_fraction_of_n_fft() {
    // beta * n_fft, rounded (the DVB-family Tu-relative convention).
    let n_fft = 64;
    let cp_len = 16;
    let cases = [(0.0f32, 0), (1.0 / 32.0, 2), (1.0 / 16.0, 4), (0.125, 8)];
    for (beta, expect) in cases {
        let cfg = window_cfg(n_fft, cp_len).with_symbol_window_beta_tu(beta);
        assert_eq!(
            cfg.carrier_plan.window_roll_off(),
            expect,
            "beta_tu={beta} -> roll_off"
        );
    }
}

/// Mean power (dB, linear-averaged) over the `take`-point FFT bins whose
/// carrier-equivalent index (`bin * n_fft / take`, signed) lands in the
/// skirt band `[lo_k, hi_k]` outside the occupied cluster.
fn mean_skirt_power_db(pdb: &[f32], take: usize, n_fft: usize, lo_k: i32, hi_k: i32) -> f32 {
    let t = take as i32;
    let mut acc = 0.0f64;
    let mut count = 0usize;
    for (bin, &p) in pdb.iter().enumerate().take(take) {
        let signed = if (bin as i32) <= t / 2 {
            bin as i32
        } else {
            bin as i32 - t
        };
        // take-FFT bin -> carrier-equivalent index (in n_fft units)
        let k = (signed.abs() * n_fft as i32) / t;
        if k >= lo_k && k <= hi_k {
            acc += 10f64.powf(p as f64 / 10.0);
            count += 1;
        }
    }
    (10.0 * (acc / count.max(1) as f64).log10()) as f32
}

#[test]
fn symbol_windowing_reduces_skirt_power() {
    // Windowing softens the inter-symbol boundary discontinuity, pulling down the
    // `~1/f` spectral skirt of the concatenated stream. The effect is in the
    // sidelobe region a little outside the occupied band (not the immediate
    // main-lobe transition, nor the far noise floor); measure there and require a
    // clear reduction. Demonstrated, not asserted (mirrors R4). ~11 dB observed.
    let n_fft = 128usize;
    let cp_len = 32usize; // guard 1/4
    let fs = 240_000.0f32;
    let roll_off = cp_len / 2; // 16, the max-transparent taper

    let half = (n_fft / 2) as i32;
    let occupied = half / 2; // carriers -occupied..=occupied
    let data: Vec<i32> = (1..occupied).chain(-(occupied - 1)..0).collect();
    let plan = CarrierPlan::new(n_fft, cp_len).with_data_carriers(data);
    let cfg = OfdmConfig::new(plan, fs, 0.0, 1.0, ConstellationOrder::Qpsk);

    let bits: Vec<u8> = (0..8192u32)
        .map(|i| ((i.wrapping_mul(2654435761)) >> 24) as u8)
        .collect();
    let plain = OfdmMod::new(&cfg).modulate(&bits);

    // Window a copy in place, per symbol.
    let sps = cfg.samples_per_ofdm_symbol();
    let mut windowed = plain.clone();
    let mut win = SymbolWindow::new(sps, roll_off);
    let mut off = 0;
    while off + sps <= windowed.len() {
        let sym: Vec<C32> = windowed[off..off + sps].to_vec();
        win.process(&sym, &mut windowed[off..off + sps]);
        off += sps;
    }

    let take = 4096.min(plain.len() - sps);
    let pdb_plain = complex_power_db(&plain[sps..sps + take], take);
    let pdb_win = complex_power_db(&windowed[sps..sps + take], take);

    // Skirt band: a few carriers beyond the occupied edge, out to a modest
    // distance — where the sidelobe rolloff (not the main-lobe transition) lives.
    let lo_k = occupied + 4;
    let hi_k = occupied + 24;
    let skirt_plain = mean_skirt_power_db(&pdb_plain, take, n_fft, lo_k, hi_k);
    let skirt_win = mean_skirt_power_db(&pdb_win, take, n_fft, lo_k, hi_k);

    let drop_db = skirt_plain - skirt_win;
    assert!(
        drop_db > 5.0,
        "windowing should cut skirt power by >5 dB \
         (plain={skirt_plain:.1} dB, windowed={skirt_win:.1} dB, drop={drop_db:.1} dB)"
    );
}

/// Windows every `sps`-sample symbol of `stream` in place (used by the combined
/// Track A + Track B spectral test).
fn window_stream_in_place(stream: &mut [C32], sps: usize, roll_off: usize) {
    if roll_off == 0 {
        return;
    }
    let mut win = SymbolWindow::new(sps, roll_off);
    let mut off = 0;
    while off + sps <= stream.len() {
        let sym: Vec<C32> = stream[off..off + sps].to_vec();
        win.process(&sym, &mut stream[off..off + sps]);
        off += sps;
    }
}

// ── TX baseband low-pass: beyond the windowing ceiling (Track C, R15) ───────

/// The COFDM geometry used by the spectral-mask tests: a 256-point grid with a
/// 31-carrier edge guard (occupied half-width 96 of 128, so there is real
/// unoccupied bandwidth for a mask to work in) and a quarter guard interval —
/// `cp_len = 64`, which affords a 65-tap mask at back-off `cp_len/2`.
const MASK_GEOM: (usize, usize, usize) = (256, 64, 31);

fn mask_config() -> OfdmConfig {
    let (n_fft, cp_len, edge_guard) = MASK_GEOM;
    let plan = CarrierPlan::new(n_fft, cp_len).with_contiguous_data(edge_guard, false);
    OfdmConfig::new(plan, 240_000.0, 0.0, 1.0, ConstellationOrder::Qpsk)
}

/// The occupied half-width of [`mask_config`]'s plan, in carriers.
fn mask_occupied_half() -> i32 {
    let (n_fft, _, edge_guard) = MASK_GEOM;
    (n_fft as i32 / 2 - 1) - edge_guard as i32
}

/// Complex power spectrum measured through a 4-term Blackman–Harris window
/// (sidelobes ≈ −92 dB).
///
/// `complex_power_db` takes a raw rectangular slice, whose own `~1/f` leakage
/// floor sits around −35 dB below the in-band power — fine for the ~11 dB
/// effects the windowing tests measure, but it would *hide* a 60 dB mask
/// entirely: the leakage of the analysis, not the signal, would be what got
/// measured. Anything claiming deep stop-band attenuation must be read through
/// a window with sidelobes below the attenuation being claimed.
fn complex_power_db_bh(samples: &[C32], n_fft: usize) -> Vec<f32> {
    const A: [f32; 4] = [0.35875, 0.48829, 0.14128, 0.01168];
    let mut buf: Vec<rustfft::num_complex::Complex<f32>> = (0..n_fft)
        .map(|i| {
            let s = samples.get(i).copied().unwrap_or_default();
            let x = core::f32::consts::TAU * i as f32 / n_fft as f32;
            let w = A[0] - A[1] * x.cos() + A[2] * (2.0 * x).cos() - A[3] * (3.0 * x).cos();
            rustfft::num_complex::Complex::new(s.re * w, s.im * w)
        })
        .collect();
    FftPlanner::new().plan_fft_forward(n_fft).process(&mut buf);
    buf.iter()
        .map(|c| 10.0 * ((c.re * c.re + c.im * c.im) + 1e-12).log10())
        .collect()
}

#[test]
fn tx_lowpass_drops_out_of_band_below_the_windowing_floor() {
    // R15: symbol windowing works on the symbol seam and tops out around ~11 dB
    // in the skirt (R10). The baseband mask attacks the same energy directly in
    // the frequency domain, so it is not bound by that ceiling — and because the
    // two mechanisms are independent, stacking them beats either alone.
    //
    // Measured in the mask's STOP band (past its transition), which is how an
    // emission mask is specified in the first place: at a stated offset from the
    // band edge. The transition itself is deliberately unattenuated.
    let (n_fft, cp_len, _) = MASK_GEOM;
    let cfg = mask_config();
    let occupied = mask_occupied_half(); // 96
    let roll_off = cp_len / 2; // 32, the max-transparent taper
    // 65 taps -> group delay 32 = min(cp_len - b, b) at b = cp_len/2.
    let lowpass = TxLowpass::for_null_band(n_fft, occupied as usize, 65, 60.0);
    assert!(lowpass.fits_guard(cp_len, 0, cp_len / 2));
    assert!(lowpass.transition_fits(n_fft, occupied as usize));

    let bits: Vec<u8> = (0..32768u32)
        .map(|i| ((i.wrapping_mul(2654435761)) >> 24) as u8)
        .collect();
    let sps = cfg.samples_per_ofdm_symbol();
    let baseline = OfdmMod::new(&cfg).modulate(&bits);

    let mut window_only = baseline.clone();
    window_stream_in_place(&mut window_only, sps, roll_off);
    let mut lowpass_only = baseline.clone();
    lowpass.apply(&mut lowpass_only);
    let mut both = window_only.clone();
    lowpass.apply(&mut both);

    // Stop band: from the mask's stop-band edge out to Nyquist (all a
    // critically-sampled complex baseband spectrum can show — energy past fs/2
    // folds back).
    let take = 4096.min(baseline.len() - sps);
    let lo_k = (lowpass.stopband_edge_norm() * n_fft as f32).ceil() as i32;
    let hi_k = n_fft as i32 / 2;
    assert!(lo_k < hi_k, "stop band must be inside the null band");
    let skirt = |s: &[C32]| {
        let pdb = complex_power_db_bh(&s[sps..sps + take], take);
        mean_skirt_power_db(&pdb, take, n_fft, lo_k, hi_k)
    };

    let s_base = skirt(&baseline);
    let s_win = skirt(&window_only);
    let s_lpf = skirt(&lowpass_only);
    let s_both = skirt(&both);
    println!(
        "stop band k in [{lo_k}, {hi_k}] dB: base={s_base:.1} window={s_win:.1} \
         lowpass={s_lpf:.1} both={s_both:.1}"
    );

    // Observed on this geometry: base −30, window −62, mask −96, both −116 dB.
    // Assert roughly half of each observed margin, so the test states the effect
    // without pinning exact numbers. Note windowing buys far more here (~32 dB)
    // than the ~11 dB the R10 test measures a few carriers out: it changes the
    // skirt's decay *rate*, so its payoff grows with distance from the band edge.
    assert!(
        s_win < s_base - 10.0,
        "windowing should beat baseline ({s_win:.1} vs {s_base:.1})"
    );
    assert!(
        s_lpf < s_win - 15.0,
        "the mask must drop out-of-band power clearly BELOW the windowing floor \
         (lowpass={s_lpf:.1} dB, window-only={s_win:.1} dB)"
    );
    assert!(
        s_both < s_lpf - 8.0,
        "mask + taper should stack, beating the mask alone \
         ({s_both:.1} vs {s_lpf:.1})"
    );
}

#[test]
fn all_three_spectral_levers_stack() {
    // R17: the full Track A + B + C stack on one COFDM link, each lever added in
    // turn. They attack different things — the edge guard moves the loudest sinc
    // generators inward, the taper lowers the skirt's decay rate, the mask
    // attenuates what is left directly — so every addition must strictly improve
    // on the one before, and the full stack must beat every partial combination.
    let (n_fft, cp_len, edge_guard) = MASK_GEOM;
    let fs = 240_000.0f32;
    let occupied = mask_occupied_half(); // 96
    let roll_off = cp_len / 2; // 32
    let lowpass = TxLowpass::for_null_band(n_fft, occupied as usize, 65, 60.0);

    let plan_of = |guard: usize| CarrierPlan::new(n_fft, cp_len).with_contiguous_data(guard, false);
    let cfg_full = OfdmConfig::new(plan_of(0), fs, 0.0, 1.0, ConstellationOrder::Qpsk);
    let cfg_guard = OfdmConfig::new(plan_of(edge_guard), fs, 0.0, 1.0, ConstellationOrder::Qpsk);

    let bits: Vec<u8> = (0..32768u32)
        .map(|i| ((i.wrapping_mul(2654435761)) >> 24) as u8)
        .collect();
    let sps = cfg_full.samples_per_ofdm_symbol();

    let baseline = OfdmMod::new(&cfg_full).modulate(&bits);
    let guard_only = OfdmMod::new(&cfg_guard).modulate(&bits);
    let mut guard_win = guard_only.clone();
    window_stream_in_place(&mut guard_win, sps, roll_off);
    let mut guard_lpf = guard_only.clone();
    lowpass.apply(&mut guard_lpf);
    let mut all_three = guard_win.clone();
    lowpass.apply(&mut all_three);

    // Measured in two regions, because the taper and the mask do not act in the
    // same place. The mask leaves its own transition band deliberately
    // unattenuated, so close to the band edge the taper is the useful lever;
    // past the transition the mask takes over by tens of dB. Reporting only one
    // of the two would misrepresent either lever.
    let take = 4096.min(baseline.len() - sps);
    let stop_k = (lowpass.stopband_edge_norm() * n_fft as f32).ceil() as i32;
    let skirt = |s: &[C32], lo_k: i32, hi_k: i32| {
        let pdb = complex_power_db_bh(&s[sps..sps + take], take);
        mean_skirt_power_db(&pdb, take, n_fft, lo_k, hi_k)
    };
    let near = |s: &[C32]| skirt(s, occupied + 4, stop_k - 1);
    let far = |s: &[C32]| skirt(s, stop_k, n_fft as i32 / 2);

    for (label, band) in [
        ("near edge", &near as &dyn Fn(&[C32]) -> f32),
        ("stop band", &far),
    ] {
        let s_base = band(&baseline);
        let s_guard = band(&guard_only);
        let s_win = band(&guard_win);
        let s_lpf = band(&guard_lpf);
        let s_all = band(&all_three);
        println!(
            "{label}: base={s_base:.1} +guard={s_guard:.1} +window={s_win:.1} \
             +mask={s_lpf:.1} all={s_all:.1} dB"
        );

        assert!(
            s_guard < s_base,
            "{label}: edge guard should beat baseline ({s_guard:.1} vs {s_base:.1})"
        );
        assert!(
            s_win < s_guard,
            "{label}: adding the taper should beat the guard alone \
             ({s_win:.1} vs {s_guard:.1})"
        );
        assert!(
            s_lpf < s_guard,
            "{label}: adding the mask should beat the guard alone \
             ({s_lpf:.1} vs {s_guard:.1})"
        );
        assert!(
            s_all < s_win && s_all < s_lpf,
            "{label}: the full stack must beat every partial combination \
             (all={s_all:.1}, guard+window={s_win:.1}, guard+mask={s_lpf:.1})"
        );
    }

    // And the complementarity itself: inside the mask's transition the taper is
    // the better lever; beyond it the mask wins by a wide margin. This is the
    // reason to ship both rather than pick one.
    assert!(
        near(&guard_win) < near(&guard_lpf),
        "near the band edge the taper should beat the mask's transition band \
         ({:.1} vs {:.1})",
        near(&guard_win),
        near(&guard_lpf)
    );
    assert!(
        far(&guard_lpf) < far(&guard_win) - 15.0,
        "past its transition the mask should beat the taper by a wide margin \
         ({:.1} vs {:.1})",
        far(&guard_lpf),
        far(&guard_win)
    );
}

#[test]
fn edge_guard_and_windowing_combine() {
    // Track A (edge-carrier nulling) and Track B (symbol windowing) are
    // independent levers that compose: nulling moves the strongest sinc
    // generators inward, windowing lowers the boundary-discontinuity skirt. Both
    // together must beat either alone in the skirt region. Demonstrated (mirrors
    // R4), COFDM only.
    let n_fft = 128usize;
    let cp_len = 32usize; // guard 1/4
    let fs = 240_000.0f32;
    let edge_guard = 8usize;
    let roll_off = cp_len / 2;

    // Same active span with/without the edge guard: the guard nulls the outer
    // `edge_guard` carriers of a contiguous fill.
    let cfg_plain = OfdmConfig::new(
        CarrierPlan::new(n_fft, cp_len).with_contiguous_data(0, false),
        fs,
        0.0,
        1.0,
        ConstellationOrder::Qpsk,
    );
    let cfg_guard = OfdmConfig::new(
        CarrierPlan::new(n_fft, cp_len).with_contiguous_data(edge_guard, false),
        fs,
        0.0,
        1.0,
        ConstellationOrder::Qpsk,
    );

    let bits: Vec<u8> = (0..16384u32)
        .map(|i| ((i.wrapping_mul(2654435761)) >> 24) as u8)
        .collect();
    let sps = cfg_plain.samples_per_ofdm_symbol();

    // Four variants: baseline, guard-only, window-only, both.
    let baseline = OfdmMod::new(&cfg_plain).modulate(&bits);
    let guard_only = OfdmMod::new(&cfg_guard).modulate(&bits);
    let mut window_only = baseline.clone();
    window_stream_in_place(&mut window_only, sps, roll_off);
    let mut both = guard_only.clone();
    window_stream_in_place(&mut both, sps, roll_off);

    // Skirt band beyond the full-fill occupied edge (|k| ~ n_fft/2). Both levers
    // act just outside the band; measure a window there.
    let take = 4096.min(baseline.len() - sps);
    let band = n_fft as i32 / 2; // full-fill occupied half-width
    let lo_k = band - 20;
    let hi_k = band - 2;
    let skirt = |s: &[C32]| {
        let pdb = complex_power_db(&s[sps..sps + take], take);
        mean_skirt_power_db(&pdb, take, n_fft, lo_k, hi_k)
    };

    let s_base = skirt(&baseline);
    let s_guard = skirt(&guard_only);
    let s_win = skirt(&window_only);
    let s_both = skirt(&both);

    // Each lever alone beats the baseline, and both together beat either alone.
    assert!(
        s_guard < s_base,
        "guard should beat baseline ({s_guard} vs {s_base})"
    );
    assert!(
        s_win < s_base,
        "window should beat baseline ({s_win} vs {s_base})"
    );
    assert!(
        s_both < s_guard && s_both < s_win,
        "combined should beat either alone (both={s_both:.1}, guard={s_guard:.1}, \
         win={s_win:.1}, base={s_base:.1})"
    );
}
