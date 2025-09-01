
// helpers

fn real_tone(fs: f32, f_hz: f32, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|k| amp * (std::f32::consts::TAU * f_hz * (k as f32) / fs).sin())
        .collect()
}

/// crude single-bin SNR estimate around f0 against a distant off-bin
fn snr_db_at(fs: f32, f0: f32, x: &[f32]) -> f32 {
    let n = x.len().max(1);
    let proj = |f: f32| {
        let w = -std::f32::consts::TAU * f / fs;
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (k, &s) in x.iter().enumerate() {
            let t = w * (k as f32);
            re += s * t.cos();
            im += s * t.sin();
        }
        (re * re + im * im) / ((n as f32) * (n as f32))
    };
    let p_sig = proj(f0);
    let p_off = proj(f0 * 0.73) + 1e-20;
    10.0 * (p_sig / p_off).log10()
}

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    (sum_sq / x.len() as f32).sqrt()
}

// Trim initial transient to avoid startup bias in SNR checks
fn tail<'a, T>(x: &'a [T]) -> &'a [T] {
    &x[x.len() / 4..]
}

// === CW round-trip: CwKeyedMod -> CwEnvelopeDemod ========================

#[test]
fn roundtrip_cw_envelope() {
    use crate::core::{AudioToIqChain, IqToAudioChain};
    use crate::modulate::CwKeyedMod;
    use crate::demodulate::CwEnvelopeDemod;

    let fs = 48_000.0;
    let n = 24_000; // 0.5 s
    let pitch_hz = 700.0;

    // 50% duty square @ 5 Hz as keying envelope (0 or 1)
    let key_f = 5.0;
    let key_env: Vec<f32> = (0..n)
        .map(|k| ((k as f32 * key_f / fs).fract() < 0.5) as i32 as f32)
        .collect();

    // TX: keyed CW
    let mut tx = AudioToIqChain::new(CwKeyedMod::new(fs, pitch_hz, 3.0, 3.0));
    let iq = tx.process(key_env.clone());

    // RX: envelope demod (recovers keying envelope)
    let mut rx = IqToAudioChain::new(CwEnvelopeDemod::new(fs, pitch_hz, 300.0));
    let audio = rx.process(iq);

    // ---- drop a fixed time (e.g., 100 ms) of startup transient --------------
    let skip = (0.100 * fs) as usize;
    let a = &audio[skip.min(audio.len())..];
    let k = &key_env[skip.min(key_env.len())..];

    // Split by ON/OFF windows and compute RMS using the shared helper
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

// === AM round-trip: AmDsbMod -> AmEnvelopeDemod ==========================

#[test]
fn roundtrip_am_envelope() {
    use crate::core::{AudioToIqChain, IqToAudioChain};
    use crate::modulate::AmDsbMod;
    use crate::demodulate::AmEnvelopeDemod;

    let fs = 48_000.0;
    let n = 32_768;
    let f_mod = 1_000.0;

    let audio_in = real_tone(fs, f_mod, n, 0.5);

    // TX: baseband AM with carrier (carrier_level > 0 → conventional AM)
    let mut tx = AudioToIqChain::new(AmDsbMod::new(fs, 0.0, 0.8, 0.5));
    let iq = tx.process(audio_in.clone());

    // RX: envelope demod → audio
    let mut dem = IqToAudioChain::new(AmEnvelopeDemod::new(fs, 5_000.0));
    let audio_out = dem.process(iq);

    let s = tail(&audio_out);
    let snr = snr_db_at(fs, f_mod, s);
    assert!(snr > 24.0, "AM roundtrip SNR too low: {snr:.1} dB");
}

// === SSB round-trip: SsbPhasingMod(USB) -> SsbProductDemod(USB) =========

#[test]
fn roundtrip_ssb_usb_product() {
    use crate::core::{AudioToIqChain, IqToAudioChain};
    use crate::modulate::SsbPhasingMod;
    use crate::demodulate::SsbProductDemod;

    let fs = 48_000.0;
    let n = 32_768;
    let f_audio = 1_200.0;

    let audio_in: Vec<f32> = (0..n)
        .map(|k| 0.4 * (std::f32::consts::TAU * f_audio * (k as f32) / fs).sin())
        .collect();

    // Weaver/phasing SSB at baseband, USB, audio BW 2.8 kHz, IF 1.5 kHz
    let audio_bw_hz = 2_800.0;
    let audio_if_hz = 1_500.0;

    let mut tx = AudioToIqChain::new(SsbPhasingMod::new(fs, audio_bw_hz, audio_if_hz, 0.0, true));
    let iq = tx.process(audio_in.clone());

    // Product-detector SSB demod (USB) with BFO set to audio_if_hz
    let mut rx = IqToAudioChain::new(SsbProductDemod::new(fs, audio_if_hz, audio_bw_hz));
    let audio_out = rx.process(iq);

    // Skip 120 ms of startup transient
    let skip = (0.120 * fs) as usize;
    let s = &audio_out[skip.min(audio_out.len())..];

    // Expect a strong tone at f_audio
    let snr = snr_db_at(fs, f_audio, s);
    assert!(snr > 18.0, "SSB roundtrip SNR too low: {snr:.1} dB");
}

// === FM round-trip: FmPhaseAccumMod -> FmQuadratureDemod =================

#[test]
fn roundtrip_fm_quadrature() {
    use crate::core::{AudioToIqChain, IqToAudioChain};
    use crate::modulate::FmPhaseAccumMod;
    use crate::demodulate::FmQuadratureDemod;

    let fs = 48_000.0;
    let n = 32_768;
    let f_mod = 1_000.0;

    let audio_in = real_tone(fs, f_mod, n, 0.5);

    // TX: baseband NBFM with 2.5 kHz deviation
    let mut tx = AudioToIqChain::new(FmPhaseAccumMod::new(fs, 2_500.0, 0.0));
    let iq = tx.process(audio_in.clone());

    // RX: quadrature FM demod with audio low-pass ~5 kHz
    let mut dem = IqToAudioChain::new(FmQuadratureDemod::new(fs, 2_500.0, 5_000.0));
    let audio_out = dem.process(iq);

    let s = tail(&audio_out);
    let snr = snr_db_at(fs, f_mod, s);
    assert!(snr > 20.0, "FM roundtrip SNR too low: {snr:.1} dB");
}

// === PM round-trip: PmDirectPhaseMod -> PmQuadratureDemod ================

#[test]
fn roundtrip_pm_quadrature() {
    use crate::core::{AudioToIqChain, IqToAudioChain};
    use crate::modulate::PmDirectPhaseMod;
    use crate::demodulate::PmQuadratureDemod;

    let fs = 48_000.0;
    let n = 32_768;
    let f_mod = 900.0;

    let audio_in = real_tone(fs, f_mod, n, 0.5);

    // TX: baseband PM with kp = 0.9 rad per peak unit
    let mut tx = AudioToIqChain::new(PmDirectPhaseMod::new(fs, 0.9, 0.0));
    let iq = tx.process(audio_in.clone());

    // RX: PM demod (quadrature/dφ), audio LP ~5 kHz
    let mut dem = IqToAudioChain::new(PmQuadratureDemod::new(0.9, 5_000.0, fs));
    let audio_out = dem.process(iq);

    let s = tail(&audio_out);
    let snr = snr_db_at(fs, f_mod, s);
    assert!(snr > 18.0, "PM roundtrip SNR too low: {snr:.1} dB");
}
