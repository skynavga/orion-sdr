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
