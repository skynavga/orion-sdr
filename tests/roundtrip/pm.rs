// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0


use orion_sdr::core::{AudioToIqChain, IqToAudioChain};
use orion_sdr::modulate::PmDirectPhaseMod;
use orion_sdr::demodulate::PmQuadratureDemod;
use crate::common::snr_db_at;
use super::helpers::{real_tone, tail};

#[test]
fn roundtrip_pm_quadrature() {
    let fs = 48_000.0;
    let n = 32_768;
    let f_mod = 900.0;

    let audio_in = real_tone(fs, f_mod, n, 0.5);

    let mut tx = AudioToIqChain::new(PmDirectPhaseMod::new(fs, 0.9, 0.0));
    let iq = tx.process(audio_in.clone());

    let mut dem = IqToAudioChain::new(PmQuadratureDemod::new(fs, 0.9, 5_000.0));
    let audio_out = dem.process(iq);

    let s = tail(&audio_out);
    let snr = snr_db_at(fs, f_mod, s);
    assert!(snr > 18.0, "PM roundtrip SNR too low: {snr:.1} dB");
}
