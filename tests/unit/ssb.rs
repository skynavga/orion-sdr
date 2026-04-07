// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0


use orion_sdr::demodulate::SsbProductDemod;
use orion_sdr::core::Block;
use orion_sdr::util::gen_complex_tone;
use super::helpers::dft_power;

#[test]
fn ssb_product_demod_yields_strong_tone_and_low_dc() {
    let fs = 48_000.0;
    let n = 16_384;
    let f_tone = 1_000.0;
    let iq = gen_complex_tone(fs, f_tone, n);

    let mut det = SsbProductDemod::new(fs, 0.0, 2_800.0);
    let mut audio = vec![0.0f32; n];
    let _rep = det.process(&iq, &mut audio);

    let mean = audio.iter().copied().sum::<f32>() / (audio.len() as f32);
    assert!(mean.abs() < 1e-3, "DC too high: {}", mean);

    let p_sig = dft_power(&audio, fs, f_tone);
    let p_off = dft_power(&audio, fs, 700.0);
    let snr_db = 10.0 * (p_sig / (p_off + 1e-20)).log10();

    assert!(
        snr_db > 25.0,
        "Expected >25 dB at 1 kHz vs 700 Hz, got {:.2} dB (p_sig={}, p_off={})",
        snr_db, p_sig, p_off
    );
}
