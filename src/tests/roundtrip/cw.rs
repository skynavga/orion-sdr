
use crate::core::{AudioToIqChain, IqToAudioChain};
use crate::modulate::CwKeyedMod;
use crate::demodulate::CwEnvelopeDemod;
use super::rms;

#[test]
fn roundtrip_cw_envelope() {
    let fs = 48_000.0;
    let n = 24_000;
    let pitch_hz = 700.0;

    let key_f = 5.0;
    let key_env: Vec<f32> = (0..n)
        .map(|k| ((k as f32 * key_f / fs).fract() < 0.5) as i32 as f32)
        .collect();

    let mut tx = AudioToIqChain::new(CwKeyedMod::new(fs, pitch_hz, 3.0, 3.0));
    let iq = tx.process(key_env.clone());

    let mut rx = IqToAudioChain::new(CwEnvelopeDemod::new(fs, pitch_hz, 300.0));
    let audio = rx.process(iq);

    let skip = (0.100 * fs) as usize;
    let a = &audio[skip.min(audio.len())..];
    let k = &key_env[skip.min(key_env.len())..];

    let mut on: Vec<f32> = Vec::with_capacity(a.len());
    let mut off: Vec<f32> = Vec::with_capacity(a.len());
    for (&y, &ke) in a.iter().zip(k.iter()) {
        if ke > 0.5 { on.push(y); } else { off.push(y); }
    }

    let on_rms  = rms(&on);
    let off_rms = rms(&off);

    let contrast_db = 20.0 * (on_rms / (off_rms + 1e-12)).log10();
    assert!(
        contrast_db > 14.0,
        "CW envelope ON/OFF contrast too low: {contrast_db:.1} dB (on_rms={on_rms:.4}, off_rms={off_rms:.4})"
    );
}
