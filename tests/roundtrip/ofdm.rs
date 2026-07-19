// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::common::add_awgn;
use num_complex::Complex32 as C32;
use orion_sdr::core::Block;
use orion_sdr::demodulate::{EqualizerMethod, OfdmDecider, OfdmDemod, OfdmEqualizer};
use orion_sdr::dsp::Rotator;
use orion_sdr::modulate::{ConstellationOrder, OfdmConfig, OfdmMod};
use orion_sdr::multicarrier::{
    CarrierGrid, CarrierPlan, CyclicPrefixRemove, FftBlock, GridExtract,
};
use orion_sdr::sync::{OfdmPreamble, generate_ofdm_preamble, ofdm_sync};

fn plan(n_fft: usize, cp_len: usize) -> CarrierPlan {
    let half = (n_fft / 2) as i32;
    let data: Vec<i32> = (1..half).chain(-(half - 1)..0).collect();
    CarrierPlan::new(n_fft, cp_len).with_data_carriers(data)
}

fn config(n_fft: usize, cp_len: usize, constellation: ConstellationOrder) -> OfdmConfig {
    OfdmConfig::new(plan(n_fft, cp_len), 48_000.0, 0.0, 1.0, constellation)
}

fn ofdm_roundtrip_noiseless(constellation: ConstellationOrder, n_symbols: usize) {
    let n_fft = 64;
    let cp_len = 8;
    let cfg = config(n_fft, cp_len, constellation);
    let bps = cfg.bits_per_ofdm_symbol();

    let bits_in: Vec<u8> = (0..n_symbols * bps)
        .map(|i| ((i / 5 + i % 3) & 1) as u8)
        .collect();

    let mut modstage = OfdmMod::new(&cfg);
    let iq = modstage.modulate(&bits_in);

    let mut demod = OfdmDemod::new(&cfg);
    let mut decider = OfdmDecider::new(&cfg);
    let mut bits_out = vec![0u8; bits_in.len()];

    let mut in_off = 0usize;
    let mut out_off = 0usize;
    let samples_per_symbol = cfg.samples_per_ofdm_symbol();
    let num_data = demod.num_data_carriers();
    let mut soft = vec![C32::default(); num_data];

    while in_off + samples_per_symbol <= iq.len() {
        let wr_demod = demod.process(&iq[in_off..], &mut soft);
        assert_eq!(wr_demod.in_read, samples_per_symbol);
        let wr_decide = decider.process(&soft, &mut bits_out[out_off..]);
        assert_eq!(wr_decide.out_written, bps);
        in_off += samples_per_symbol;
        out_off += bps;
    }

    assert_eq!(
        bits_in, bits_out,
        "OFDM {:?} noiseless roundtrip failed",
        constellation
    );
}

#[test]
fn roundtrip_ofdm_qpsk_noiseless() {
    ofdm_roundtrip_noiseless(ConstellationOrder::Qpsk, 8);
}

#[test]
fn roundtrip_ofdm_qam16_noiseless() {
    ofdm_roundtrip_noiseless(ConstellationOrder::Qam16, 8);
}

#[test]
fn roundtrip_ofdm_qam64_noiseless() {
    ofdm_roundtrip_noiseless(ConstellationOrder::Qam64, 8);
}

#[test]
fn roundtrip_ofdm_awgn_flat_channel() {
    let n_fft = 64;
    let cp_len = 8;
    let cfg = config(n_fft, cp_len, ConstellationOrder::Qpsk);
    let bps = cfg.bits_per_ofdm_symbol();
    let n_symbols = 20;

    let bits_in: Vec<u8> = (0..n_symbols * bps)
        .map(|i| ((i / 7 + i % 5) & 1) as u8)
        .collect();

    let mut modstage = OfdmMod::new(&cfg);
    let mut iq = modstage.modulate(&bits_in);

    // Mild AWGN over an otherwise flat (unity, unfiltered) channel — well
    // within this release's stated scope. Noise power is scaled relative to
    // the time-domain signal's own power, since IFFT output energy is spread
    // across n_fft samples (much lower per-sample amplitude than the
    // frequency-domain unit-energy QPSK symbols).
    let sig_power: f32 = iq.iter().map(|s| s.norm_sqr()).sum::<f32>() / iq.len() as f32;
    add_awgn(&mut iq, sig_power * 0.05, 0xA5A5_5A5A_1234_5678);

    let mut demod = OfdmDemod::new(&cfg);
    let mut decider = OfdmDecider::new(&cfg);
    let mut bits_out = vec![0u8; bits_in.len()];

    let samples_per_symbol = cfg.samples_per_ofdm_symbol();
    let num_data = demod.num_data_carriers();
    let mut soft = vec![C32::default(); num_data];

    let mut in_off = 0usize;
    let mut out_off = 0usize;
    while in_off + samples_per_symbol <= iq.len() {
        demod.process(&iq[in_off..], &mut soft);
        decider.process(&soft, &mut bits_out[out_off..]);
        in_off += samples_per_symbol;
        out_off += bps;
    }

    let errors = bits_in
        .iter()
        .zip(bits_out.iter())
        .filter(|(a, b)| a != b)
        .count();
    let ber = errors as f32 / bits_in.len() as f32;
    assert!(
        ber < 0.02,
        "OFDM AWGN flat-channel roundtrip BER too high: {:.4} ({} / {})",
        ber,
        errors,
        bits_in.len()
    );
}

#[test]
fn roundtrip_ofdm_with_cfo_and_unknown_start() {
    // First release without the "known start" shortcut: sync locates the
    // preamble and estimates fractional CFO, which is corrected via
    // Rotator before the known-scope OfdmDemod pipeline takes over.
    let n_fft = 64;
    let cp_len = 8;
    let fs = 48_000.0f32;
    let cfg = config(n_fft, cp_len, ConstellationOrder::Qpsk);
    let bps = cfg.bits_per_ofdm_symbol();
    let n_symbols = 8;

    let preamble = OfdmPreamble::new(4, 32);
    let capture_hz = fs / (2.0 * preamble.repeat_len as f32);
    let applied_cfo = capture_hz * 0.3; // within the fractional capture range

    let bits_in: Vec<u8> = (0..n_symbols * bps)
        .map(|i| ((i / 5 + i % 3) & 1) as u8)
        .collect();

    let mut modstage = OfdmMod::new(&cfg);
    let data_iq = modstage.modulate(&bits_in);
    let preamble_iq = generate_ofdm_preamble(&preamble, &cfg);

    let unknown_start = 137usize;
    let mut buf = vec![C32::default(); unknown_start];
    buf.extend_from_slice(&preamble_iq);
    buf.extend_from_slice(&data_iq);
    buf.extend(vec![C32::default(); 32]);

    let mut rot = Rotator::new(applied_cfo, fs);
    let mut with_cfo = vec![C32::default(); buf.len()];
    rot.rotate_block(&buf, &mut with_cfo);

    let sync_results = ofdm_sync(&with_cfo, fs, &preamble, 0, with_cfo.len());
    assert!(!sync_results.is_empty(), "sync failed to find the preamble");
    let best = sync_results[0];
    assert_eq!(best.start_sample, unknown_start);

    // Correct the estimated CFO, then demod starting right after the
    // preamble (a fixed, protocol-known offset from the sync point).
    let mut correction = Rotator::new(-best.cfo_hz, fs);
    let mut corrected = vec![C32::default(); with_cfo.len()];
    correction.rotate_block(&with_cfo, &mut corrected);

    let data_start = best.start_sample + preamble.total_len();
    let iq = &corrected[data_start..data_start + data_iq.len()];

    let mut demod = OfdmDemod::new(&cfg);
    let mut decider = OfdmDecider::new(&cfg);
    let samples_per_symbol = cfg.samples_per_ofdm_symbol();
    let num_data = demod.num_data_carriers();
    let mut soft = vec![C32::default(); num_data];
    let mut bits_out = vec![0u8; bits_in.len()];

    let mut in_off = 0usize;
    let mut out_off = 0usize;
    while in_off + samples_per_symbol <= iq.len() {
        demod.process(&iq[in_off..], &mut soft);
        decider.process(&soft, &mut bits_out[out_off..]);
        in_off += samples_per_symbol;
        out_off += bps;
    }

    assert_eq!(
        bits_in, bits_out,
        "OFDM roundtrip with CFO + unknown start failed"
    );
}

#[test]
fn roundtrip_ofdm_large_cfo_wide_band() {
    // CFO well beyond ±½ subcarrier spacing — the case Release E's
    // fractional-only ofdm_sync explicitly could not handle. Release F's
    // integer stage (via the training symbol) resolves the multi-spacing
    // component; the combined fractional+integer correction recovers exact
    // bits through the same known-scope OfdmDemod pipeline.
    let n_fft = 64;
    let cp_len = 8;
    let fs = 48_000.0f32;
    let cfg = config(n_fft, cp_len, ConstellationOrder::Qpsk);
    let bps = cfg.bits_per_ofdm_symbol();
    let n_symbols = 8;

    let preamble = OfdmPreamble::new(4, 32).with_training_symbol(n_fft, cp_len);
    let subcarrier_spacing_hz = fs / n_fft as f32;
    // Several whole subcarrier spacings plus a fractional remainder: far
    // beyond Release E's ±½-spacing capture range.
    let applied_cfo = 5.0 * subcarrier_spacing_hz + 0.3 * subcarrier_spacing_hz;

    let bits_in: Vec<u8> = (0..n_symbols * bps)
        .map(|i| ((i / 5 + i % 3) & 1) as u8)
        .collect();

    let mut modstage = OfdmMod::new(&cfg);
    let data_iq = modstage.modulate(&bits_in);
    let preamble_iq = generate_ofdm_preamble(&preamble, &cfg);

    let unknown_start = 137usize;
    let mut buf = vec![C32::default(); unknown_start];
    buf.extend_from_slice(&preamble_iq);
    buf.extend_from_slice(&data_iq);
    buf.extend(vec![C32::default(); 32]);

    let mut rot = Rotator::new(applied_cfo, fs);
    let mut with_cfo = vec![C32::default(); buf.len()];
    rot.rotate_block(&buf, &mut with_cfo);

    let sync_results = ofdm_sync(&with_cfo, fs, &preamble, 0, with_cfo.len());
    assert!(!sync_results.is_empty(), "sync failed to find the preamble");
    let best = sync_results[0];
    assert_eq!(best.start_sample, unknown_start);
    assert_ne!(
        best.integer_cfo_bins, 0,
        "expected the integer stage to recover a nonzero multi-spacing shift"
    );

    let total_cfo_hz = best.cfo_hz + best.integer_cfo_bins as f32 * subcarrier_spacing_hz;
    let mut correction = Rotator::new(-total_cfo_hz, fs);
    let mut corrected = vec![C32::default(); with_cfo.len()];
    correction.rotate_block(&with_cfo, &mut corrected);

    let data_start = best.start_sample + preamble.total_len();
    let iq = &corrected[data_start..data_start + data_iq.len()];

    let mut demod = OfdmDemod::new(&cfg);
    let mut decider = OfdmDecider::new(&cfg);
    let samples_per_symbol = cfg.samples_per_ofdm_symbol();
    let num_data = demod.num_data_carriers();
    let mut soft = vec![C32::default(); num_data];
    let mut bits_out = vec![0u8; bits_in.len()];

    let mut in_off = 0usize;
    let mut out_off = 0usize;
    while in_off + samples_per_symbol <= iq.len() {
        demod.process(&iq[in_off..], &mut soft);
        decider.process(&soft, &mut bits_out[out_off..]);
        in_off += samples_per_symbol;
        out_off += bps;
    }

    assert_eq!(
        bits_in, bits_out,
        "OFDM roundtrip with large multi-spacing CFO failed"
    );
}

/// Convolves `iq` with a short FIR channel (causal, `taps[0]` is the direct
/// path). Delay spread of `taps.len() - 1` samples must stay within the
/// symbol's cyclic prefix for the OFDM per-bin equalization model to hold —
/// the explicit scope limit of this release.
fn apply_fir_channel(iq: &[C32], taps: &[C32]) -> Vec<C32> {
    let mut out = vec![C32::default(); iq.len()];
    for (n, &x) in iq.iter().enumerate() {
        for (k, &h) in taps.iter().enumerate() {
            if n + k < out.len() {
                out[n + k] += x * h;
            }
        }
    }
    out
}

#[test]
fn roundtrip_ofdm_multipath_channel() {
    // Synthetic 2-tap frequency-selective channel with delay spread <=
    // cp_len (this release's explicit scope limit -- longer delay spreads
    // cause inter-symbol interference this per-bin equalizer does not
    // model). TrainingSymbolHold: one channel estimate per packet, held
    // constant, driven manually through CyclicPrefixRemove -> FftBlock ->
    // OfdmEqualizer -> GridExtract -> OfdmDecider since OfdmEqualizer is a
    // standalone composable stage, not fused into OfdmDemod.
    let n_fft = 64;
    let cp_len = 8;
    let fs = 48_000.0f32;
    let cfg = config(n_fft, cp_len, ConstellationOrder::Qpsk);
    let bps = cfg.bits_per_ofdm_symbol();
    let n_symbols = 8;

    let preamble = OfdmPreamble::new(4, 32).with_training_symbol(n_fft, cp_len);
    let channel_taps = [
        C32::new(0.8, 0.1),
        C32::new(0.0, 0.0),
        C32::new(0.25, -0.15),
    ];
    assert!(channel_taps.len() - 1 <= cp_len);

    let bits_in: Vec<u8> = (0..n_symbols * bps)
        .map(|i| ((i / 5 + i % 3) & 1) as u8)
        .collect();

    let mut modstage = OfdmMod::new(&cfg);
    let data_iq = modstage.modulate(&bits_in);
    let preamble_iq = generate_ofdm_preamble(&preamble, &cfg);

    let time_offset = 40usize;
    let mut buf = vec![C32::default(); time_offset];
    buf.extend_from_slice(&preamble_iq);
    buf.extend_from_slice(&data_iq);
    buf.extend(vec![C32::default(); 32]);

    let channeled = apply_fir_channel(&buf, &channel_taps);

    let sync_results = ofdm_sync(&channeled, fs, &preamble, 0, channeled.len());
    assert!(!sync_results.is_empty(), "sync failed to find the preamble");
    let best = sync_results[0];
    assert_eq!(best.start_sample, time_offset);

    // Estimate the channel from the training symbol.
    let grid = CarrierGrid::from_plan(&cfg.carrier_plan);
    let mut cp_remove = CyclicPrefixRemove::new(n_fft, cp_len);
    let mut fft = FftBlock::new(n_fft);
    let mut eq = OfdmEqualizer::new(&cfg, EqualizerMethod::TrainingSymbolHold);

    let training_start = best.start_sample + preamble.num_repeats * preamble.repeat_len;
    let training_symbol = &channeled[training_start..training_start + n_fft + cp_len];
    let mut training_time = vec![C32::default(); n_fft];
    cp_remove.process(training_symbol, &mut training_time);
    let mut training_freq = vec![C32::default(); n_fft];
    fft.process(&training_time, &mut training_freq);
    eq.estimate_from_training_symbol(&training_freq);

    // Demod the data symbols through CP-remove -> FFT -> equalizer ->
    // grid-extract -> decider.
    let data_start = best.start_sample + preamble.total_len();
    let iq = &channeled[data_start..data_start + data_iq.len()];
    let samples_per_symbol = cfg.samples_per_ofdm_symbol();
    let num_data = grid.num_data_carriers();

    let mut grid_extract = GridExtract::new(grid);
    let mut decider = OfdmDecider::new(&cfg);
    let mut bits_out = vec![0u8; bits_in.len()];

    let mut in_off = 0usize;
    let mut out_off = 0usize;
    while in_off + samples_per_symbol <= iq.len() {
        let mut time = vec![C32::default(); n_fft];
        cp_remove.process(&iq[in_off..], &mut time);
        let mut freq = vec![C32::default(); n_fft];
        fft.process(&time, &mut freq);
        let mut equalized = vec![C32::default(); n_fft];
        eq.process(&freq, &mut equalized);
        let mut soft = vec![C32::default(); num_data];
        grid_extract.process(&equalized, &mut soft);
        decider.process(&soft, &mut bits_out[out_off..]);

        in_off += samples_per_symbol;
        out_off += bps;
    }

    assert_eq!(
        bits_in, bits_out,
        "OFDM roundtrip through a static multipath channel failed"
    );
}
