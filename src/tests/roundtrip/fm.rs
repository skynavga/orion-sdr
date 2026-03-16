
use crate::core::{AudioToIqChain, IqToAudioChain};
use crate::modulate::FmPhaseAccumMod;
use crate::demodulate::FmQuadratureDemod;
use super::{real_tone, snr_db_at, tail};

#[test]
fn roundtrip_fm_quadrature() {
    let fs = 48_000.0;
    let n = 32_768;
    let f_mod = 1_000.0;

    let audio_in = real_tone(fs, f_mod, n, 0.5);

    let mut tx = AudioToIqChain::new(FmPhaseAccumMod::new(fs, 2_500.0, 0.0));
    let iq = tx.process(audio_in.clone());

    let mut dem = IqToAudioChain::new(FmQuadratureDemod::new(fs, 2_500.0, 5_000.0));
    let audio_out = dem.process(iq);

    let s = tail(&audio_out);
    let snr = snr_db_at(fs, f_mod, s);
    assert!(snr > 20.0, "FM roundtrip SNR too low: {snr:.1} dB");
}
