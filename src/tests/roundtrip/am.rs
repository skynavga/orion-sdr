
use crate::core::{AudioToIqChain, IqToAudioChain};
use crate::modulate::AmDsbMod;
use crate::demodulate::AmEnvelopeDemod;
use super::{real_tone, snr_db_at, tail};

#[test]
fn roundtrip_am_envelope() {
    let fs = 48_000.0;
    let n = 32_768;
    let f_mod = 1_000.0;

    let audio_in = real_tone(fs, f_mod, n, 0.5);

    let mut tx = AudioToIqChain::new(AmDsbMod::new(fs, 0.0, 0.8, 0.5));
    let iq = tx.process(audio_in.clone());

    let mut dem = IqToAudioChain::new(AmEnvelopeDemod::new(fs, 5_000.0));
    let audio_out = dem.process(iq);

    let s = tail(&audio_out);
    let snr = snr_db_at(fs, f_mod, s);
    assert!(snr > 24.0, "AM roundtrip SNR too low: {snr:.1} dB");
}
