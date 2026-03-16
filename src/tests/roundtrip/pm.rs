
use crate::core::{AudioToIqChain, IqToAudioChain};
use crate::modulate::PmDirectPhaseMod;
use crate::demodulate::PmQuadratureDemod;
use super::{real_tone, snr_db_at, tail};

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
