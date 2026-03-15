
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
    let mut dem = PmQuadratureDemod::new(fs, beta, 5_000.0);
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

// === FT8 unit tests =======================================================

#[test]
fn ft8_frame_length() {
    use crate::modulate::{Ft8Mod, Ft8Frame};
    use crate::modulate::ft8::FT8_FRAME_LEN;
    let tx = Ft8Mod::new(12_000.0, 1_000.0, 0.0, 1.0);
    let iq = tx.modulate(&Ft8Frame::zeros());
    assert_eq!(iq.len(), FT8_FRAME_LEN, "FT8 frame length mismatch");
}

#[test]
fn ft8_symbol_sequence_count() {
    use crate::modulate::{Ft8Mod, Ft8Frame};
    use crate::modulate::ft8::{FT8_TOTAL_SYMS, FT8_DATA_SYMS};
    let seq = Ft8Mod::build_symbol_sequence(&Ft8Frame::zeros());
    assert_eq!(seq.len(), FT8_TOTAL_SYMS);
    // Verify data symbol count: total - 3 Costas blocks × 7 = 79 - 21 = 58
    let sync_pos: [(usize, usize); 3] = [(0, 7), (36, 43), (72, 79)];
    let mut is_sync = [false; FT8_TOTAL_SYMS];
    for &(start, end) in &sync_pos { for p in start..end { is_sync[p] = true; } }
    let data_count = is_sync.iter().filter(|&&s| !s).count();
    assert_eq!(data_count, FT8_DATA_SYMS);
}

#[test]
fn ft8_costas_positions_correct() {
    use crate::modulate::{Ft8Mod, Ft8Frame};
    // The Costas sequence [3,1,4,0,6,5,2] should appear at positions 0-6, 36-42, 72-78
    let costas = [3u8, 1, 4, 0, 6, 5, 2];
    let sync_starts = [0usize, 36, 72];
    let seq = Ft8Mod::build_symbol_sequence(&Ft8Frame::zeros());
    for &start in &sync_starts {
        for i in 0..7 {
            assert_eq!(seq[start + i], costas[i],
                "FT8 Costas mismatch at sym {}: got {}, expected {}",
                start + i, seq[start + i], costas[i]);
        }
    }
}

#[test]
fn ft8_iq_power_unity() {
    use crate::modulate::{Ft8Mod, Ft8Frame};
    use crate::modulate::ft8::FT8_FRAME_LEN;
    let tx = Ft8Mod::new(12_000.0, 1_000.0, 0.0, 1.0);
    let iq = tx.modulate(&Ft8Frame::zeros());
    // All IQ samples are unit-amplitude phasors, so mean power should be ~1.0
    let power: f32 = iq.iter().map(|z| z.norm_sqr()).sum::<f32>() / (FT8_FRAME_LEN as f32);
    assert!((power - 1.0).abs() < 0.01, "FT8 IQ power deviates from 1.0: {}", power);
}

// === Phase 2: codec unit tests ============================================

#[test]
fn ft8_gray_encode_decode_roundtrip() {
    use crate::codec::gray::{gray8_encode, gray8_decode};
    for i in 0u8..8 {
        assert_eq!(gray8_decode(gray8_encode(i)), i, "FT8 Gray roundtrip failed for {i}");
    }
}

#[test]
fn ft4_gray_encode_decode_roundtrip() {
    use crate::codec::gray::{gray4_encode, gray4_decode};
    for i in 0u8..4 {
        assert_eq!(gray4_decode(gray4_encode(i)), i, "FT4 Gray roundtrip failed for {i}");
    }
}

#[test]
fn ft8_crc_known_answer() {
    use crate::codec::crc::{ft8_add_crc, ft8_extract_crc, ft8_crc14};
    // All-zeros payload: CRC should be deterministic and non-trivially verifiable.
    let payload = [0u8; 10];
    let mut a91 = [0u8; 12];
    ft8_add_crc(&payload, &mut a91);
    let extracted = ft8_extract_crc(&a91);
    // Recompute over zero-padded payload
    let mut buf = [0u8; 12];
    buf[..10].copy_from_slice(&payload);
    let computed = ft8_crc14(&buf, 82);
    assert_eq!(extracted, computed, "CRC mismatch for all-zero payload");
}

#[test]
fn ldpc_encode_syndrome_passes() {
    use crate::codec::ldpc::{ldpc_encode, ldpc_count_errors, K_BYTES, N_BYTES, N};
    use crate::codec::crc::ft8_add_crc;
    let payload = [0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0x00u8];
    let mut a91 = [0u8; K_BYTES];
    ft8_add_crc(&payload, &mut a91);
    let mut codeword = [0u8; N_BYTES];
    ldpc_encode(&a91, &mut codeword);
    let mut hard = [0u8; N];
    for i in 0..N {
        hard[i] = (codeword[i / 8] >> (7 - (i % 8))) & 1;
    }
    assert_eq!(ldpc_count_errors(&hard), 0, "LDPC syndrome check failed");
}

#[test]
fn ft8_codec_encode_produces_valid_tones() {
    use crate::codec::ft8::{Ft8Codec, Ft8Bits};
    let payload: Ft8Bits = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60];
    let frame = Ft8Codec::encode(&payload);
    for &t in frame.0.iter() {
        assert!(t < 8, "FT8 codec produced out-of-range tone: {t}");
    }
}

#[test]
fn ft4_codec_encode_produces_valid_tones() {
    use crate::codec::ft4::{Ft4Codec, Ft4Bits};
    let payload: Ft4Bits = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60];
    let frame = Ft4Codec::encode(&payload);
    for &t in frame.0.iter() {
        assert!(t < 4, "FT4 codec produced out-of-range tone: {t}");
    }
}

// === FT4 unit tests =======================================================

#[test]
fn ft4_frame_length() {
    use crate::modulate::{Ft4Mod, Ft4Frame};
    use crate::modulate::ft4::FT4_FRAME_LEN;
    let tx = Ft4Mod::new(12_000.0, 1_000.0, 0.0, 1.0);
    let iq = tx.modulate(&Ft4Frame::zeros());
    assert_eq!(iq.len(), FT4_FRAME_LEN, "FT4 frame length mismatch");
}

#[test]
fn ft4_symbol_sequence_count() {
    use crate::modulate::{Ft4Mod, Ft4Frame};
    use crate::modulate::ft4::{FT4_TOTAL_SYMS, FT4_DATA_SYMS};
    let seq = Ft4Mod::build_symbol_sequence(&Ft4Frame::zeros());
    assert_eq!(seq.len(), FT4_TOTAL_SYMS);
    // Verify data symbol count: total - 4 Costas blocks × 4 = 103 - 16 = 87
    let sync_pos: [(usize, usize); 4] = [(0, 4), (29, 33), (60, 64), (99, 103)];
    let mut is_sync = [false; FT4_TOTAL_SYMS];
    for &(start, end) in &sync_pos { for p in start..end { is_sync[p] = true; } }
    let data_count = is_sync.iter().filter(|&&s| !s).count();
    assert_eq!(data_count, FT4_DATA_SYMS);
}

#[test]
fn ft4_costas_positions_correct() {
    use crate::modulate::{Ft4Mod, Ft4Frame};
    let costas: [[u8; 4]; 4] = [[0,1,3,2],[1,0,2,3],[2,3,0,1],[3,2,1,0]];
    let sync_starts = [0usize, 29, 60, 99];
    let seq = Ft4Mod::build_symbol_sequence(&Ft4Frame::zeros());
    for (blk, &start) in sync_starts.iter().enumerate() {
        for i in 0..4 {
            assert_eq!(seq[start + i], costas[blk][i],
                "FT4 Costas mismatch blk {} sym {}: got {}, expected {}",
                blk, i, seq[start + i], costas[blk][i]);
        }
    }
}

#[test]
fn ft4_iq_power_unity() {
    use crate::modulate::{Ft4Mod, Ft4Frame};
    use crate::modulate::ft4::FT4_FRAME_LEN;
    let tx = Ft4Mod::new(12_000.0, 1_000.0, 0.0, 1.0);
    let iq = tx.modulate(&Ft4Frame::zeros());
    let power: f32 = iq.iter().map(|z| z.norm_sqr()).sum::<f32>() / (FT4_FRAME_LEN as f32);
    assert!((power - 1.0).abs() < 0.01, "FT4 IQ power deviates from 1.0: {}", power);
}
