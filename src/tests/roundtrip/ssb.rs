
use crate::core::{AudioToIqChain, IqToAudioChain};
use crate::modulate::SsbPhasingMod;
use crate::demodulate::SsbProductDemod;
use super::snr_db_at;

#[test]
fn roundtrip_ssb_usb_product() {
    let fs = 48_000.0;
    let n = 32_768;
    let f_audio = 1_200.0;

    let audio_in: Vec<f32> = (0..n)
        .map(|k| 0.4 * (std::f32::consts::TAU * f_audio * (k as f32) / fs).sin())
        .collect();

    let audio_bw_hz = 2_800.0;
    let audio_if_hz = 1_500.0;

    let mut tx = AudioToIqChain::new(SsbPhasingMod::new(fs, audio_bw_hz, audio_if_hz, 0.0, true));
    let iq = tx.process(audio_in.clone());

    let mut rx = IqToAudioChain::new(SsbProductDemod::new(fs, audio_if_hz, audio_bw_hz));
    let audio_out = rx.process(iq);

    let skip = (0.120 * fs) as usize;
    let s = &audio_out[skip.min(audio_out.len())..];

    let snr = snr_db_at(fs, f_audio, s);
    assert!(snr > 18.0, "SSB roundtrip SNR too low: {snr:.1} dB");
}
