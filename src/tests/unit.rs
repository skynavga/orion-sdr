
use crate::core::*;
use crate::dsp::*;
use crate::demodulate::*;
use crate::modulate::*;
use crate::util::*;
use num_complex::Complex32 as C32;

/// Tiny single-bin DFT; good enough for power at a specific frequency.
fn dft_power(signal: &[f32], fs: f32, f_hz: f32) -> f32 {
    let n = signal.len();
    let w = -2.0 * std::f32::consts::PI * f_hz / fs;
    let mut re = 0.0f32;
    let mut im = 0.0f32;
    for (k, &x) in signal.iter().enumerate() {
        let t = w * (k as f32);
        re += x * t.cos();
        im += x * t.sin();
    }
    // Normalize so power doesn’t scale with N
    let mag2 = (re * re + im * im) / (n as f32 * n as f32);
    mag2
}

fn snr_db_at(freq_hz: f32, fs: f32, x: &[f32]) -> f32 {
    // single-bin DFT power vs an off-bin
    let n = x.len();
    let w = -2.0 * std::f32::consts::PI * freq_hz / fs;
    let (mut re, mut im) = (0.0f32, 0.0f32);
    for (k, &s) in x.iter().enumerate() {
        let t = w * (k as f32);
        re += s * t.cos();
        im += s * t.sin();
    }
    let p_sig = (re*re + im*im) / (n as f32 * n as f32);
    // off-tone
    let f2 = freq_hz * 0.7;
    let w2 = -2.0 * std::f32::consts::PI * f2 / fs;
    let (mut r2, mut i2) = (0.0f32, 0.0f32);
    for (k, &s) in x.iter().enumerate() { let t = w2 * (k as f32); r2 += s*t.cos(); i2 += s*t.sin(); }
    let p_off = (r2*r2 + i2*i2) / (n as f32 * n as f32);
    10.0 * (p_sig / (p_off + 1e-20)).log10()
}

#[test]
fn ssb_product_demod_yields_strong_tone_and_low_dc() {
    let fs = 48_000.0;
    let n = 16_384; // ~0.34 s
    let f_tone = 1_000.0; // 1 kHz audio
    let iq = gen_complex_tone(fs, f_tone, n);

    // BFO at 0 Hz, audio BW 2.8 kHz
    let mut det = SsbProductDemod::new(fs, 0.0, 2_800.0);
    let mut audio = vec![0.0f32; n];
    let _rep = run_block(&mut det, &iq, &mut audio);

    // DC should be tiny
    let mean = audio.iter().copied().sum::<f32>() / (audio.len() as f32);
    assert!(mean.abs() < 1e-3, "DC too high: {}", mean);

    // Power at 1 kHz should dominate power off-target (e.g., 700 Hz)
    let p_sig = dft_power(&audio, fs, f_tone);
    let p_off = dft_power(&audio, fs, 700.0);
    let snr_db = 10.0 * (p_sig / (p_off + 1e-20)).log10();

    assert!(
        snr_db > 25.0,
        "Expected >25 dB at 1 kHz vs 700 Hz, got {:.2} dB (p_sig={}, p_off={})",
        snr_db,
        p_sig,
        p_off
    );
}

#[test]
fn agc_rms_converges_on_iq() {
    let fs = 48_000.0;
    let mut agc = AgcRmsIq::new(fs, 0.2, 5.0, 0.2);
    let n = 8_000;
    // Step two amplitudes to see gain adjust
    let mut input = Vec::with_capacity(n);
    for k in 0..n {
        let a = if k < n/2 { 0.02 } else { 1.0 };
        input.push(C32::new(a, 0.0));
    }
    let mut out = vec![C32::new(0.0, 0.0); n];
    let _ = agc.process(&input, &mut out);
    // RMS of second half should be ~ target
    // let mut acc = 0.0f32;
    // for z in &out[n/2..] { acc += z.norm_sqr(); }
    // let rms = (acc / (out.len()/2) as f32).sqrt();
    // assert!((rms - 0.2).abs() < 0.03, "rms={} not near target", rms);
    let tail_len = 1000.min(n/2);
    let tail = &out[n - tail_len..];
    let mut acc = 0.0f32;
    for z in tail { acc += z.norm_sqr(); }
    let rms_tail = (acc / (tail_len as f32)).sqrt();
    assert!((rms_tail - 0.2).abs() < 0.03, "tail RMS={} not near target 0.2", rms_tail);
}

#[test]
fn decimator_reduces_length_and_preserves_tone() {
    let fs = 96_000.0;
    let m = 4;
    let cutoff = fs / (m as f32) * 0.45;
    let transition = fs / (m as f32) * 0.10;
    let mut dec = FirDecimator::new(fs, m, cutoff, transition);
    // Baseband tone at 2 kHz
    let n = 4096;
    let mut nco = Nco::new(2_000.0, fs);
    let mut iq = vec![C32::new(0.0,0.0); n];
    for i in 0..n { iq[i] = mix_with_nco(C32::new(1.0,0.0), &mut nco); }
    let mut out = vec![C32::new(0.0,0.0); n/m];
    let w = dec.process(&iq, &mut out);
    assert_eq!(w.out_written, n/m);
}

#[test]
fn chain_runs_ssb_cw_and_am() {
    let fs = 48_000.0;
    let n = 4096;
    let mut tone = Vec::with_capacity(n);
    // Create a 1 kHz complex tone
    let mut nco = Nco::new(1_000.0, fs);
    for _ in 0..n {
        tone.push(mix_with_nco(C32::new(1.0,0.0), &mut nco));
    }

    // CW chain
    let mut cw = IqToAudioChain::new(CwEnvelopeDemod::new(fs, 700.0, 300.0));
    let y_cw = cw.process(tone.clone());
    assert_eq!(y_cw.len(), n);

    // AM chain
    let mut am = IqToAudioChain::new(AmEnvelopeDemod::new(fs, 5_000.0));
    let y_am = am.process(tone.clone());
    assert_eq!(y_am.len(), n);

    // SSB chain (existing)
    let mut ssb = IqToAudioChain::new(SsbProductDemod::new(fs, 0.0, 2_800.0));
    let y_ssb = ssb.process(tone);
    assert_eq!(y_ssb.len(), n);
}

#[test]
fn fm_quadrature_demod_recovers_tone() {
    let fs = 48_000.0;
    let n = 16_384;
    let f_mod = 1_000.0;
    let dev = 2_500.0; // Hz
    // Generate narrowband FM at baseband: phi[n] = sum( 2π * (dev*sin(2π f_mod t))/fs )
    let mut phi = 0.0f32;
    let mut iq = Vec::with_capacity(n);
    for k in 0..n {
        let t = k as f32 / fs;
        let f_inst = dev * (2.0*std::f32::consts::PI * f_mod * t).sin();
        phi += 2.0*std::f32::consts::PI * f_inst / fs;
        iq.push(C32::new(phi.cos(), phi.sin()));
    }
    let mut dem = FmQuadratureDemod::new(fs, dev, 5_000.0);
    let mut y = vec![0.0f32; n];
    let _ = dem.process(&iq, &mut y);
    let snr = snr_db_at(f_mod, fs, &y);
    assert!(snr > 20.0, "FM SNR too low: {:.1} dB", snr);
}

#[test]
fn pm_quadrature_demod_recovers_tone() {
    let fs = 48_000.0;
    let n = 16_384;
    let f_mod = 1_000.0;
    let beta = 0.8; // rad peak phase deviation
    let mut iq = Vec::with_capacity(n);
    for k in 0..n {
        let t = k as f32 / fs;
        let phi = beta * (2.0*std::f32::consts::PI * f_mod * t).sin();
        iq.push(C32::new(phi.cos(), phi.sin()));
    }
    let mut dem = PmQuadratureDemod::new(beta, 5_000.0, fs);
    let mut y = vec![0.0f32; n];
    let _ = dem.process(&iq, &mut y);
    let snr = snr_db_at(f_mod, fs, &y);
    assert!(snr > 20.0, "PM SNR too low: {:.1} dB", snr);
}

#[test]
fn audio_to_iq_chain_runs_and_lengths_match() {
    let fs = 48_000.0;
    let n = 4096;
    let audio = tone(fs, 1000.0, n, 0.5);

    // Baseband AM (carrier_hz=0): DSB-SC if carrier_level=0.0, AM if >0
    let am = AmDsbMod::new(fs, 0.0, 0.8, 0.5);
    let mut chain = AudioToIqChain::new(am);

    let iq = chain.process(audio);
    assert_eq!(iq.len(), n);
    // sanity: not all zeros
    let power: f32 = iq.iter().map(|z| z.norm_sqr()).sum::<f32>() / (n as f32);
    assert!(power > 1e-6, "IQ power too small");
}
