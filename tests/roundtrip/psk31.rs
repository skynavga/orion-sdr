// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0


use num_complex::Complex32 as C32;
use orion_sdr::modulate::psk31::{Bpsk31Mod, Qpsk31Mod, psk31_sps, PSK31_BAUD};
use orion_sdr::demodulate::psk31::{Bpsk31Demod, Bpsk31Decider, Qpsk31Demod, Qpsk31Decider};
use orion_sdr::codec::psk31::Psk31Stream;
use orion_sdr::codec::varicode::{varicode_encode, VaricodeDecoder};
use orion_sdr::sync::psk31_sync::psk31_sync;
use orion_sdr::core::Block;
use crate::common::add_awgn;

// -- BPSK31 roundtrip tests ------------------------------------------------

#[test]
fn roundtrip_bpsk31_bits_noiseless() {
    // Generate 128 random-ish differential bits.
    let bits_in: Vec<u8> = (0..128).map(|i| ((i * 7 + 3) & 1) as u8).collect();

    let fs = 8000.0;
    let mut modulator = Bpsk31Mod::new(fs, 0.0, 1.0);
    let iq = modulator.modulate_bits(&bits_in);

    let mut demod = Bpsk31Demod::new(fs, 0.0, 1.0);
    let mut soft = vec![0.0f32; bits_in.len() + 4];
    let wr = demod.process(&iq, &mut soft);
    soft.truncate(wr.out_written);

    let mut decider = Bpsk31Decider::new();
    let mut bits_out = vec![0u8; soft.len()];
    let dr = decider.process(&soft, &mut bits_out);
    bits_out.truncate(dr.out_written);

    // Drop first symbol (differential decode startup transient).
    let skip = 1;
    let n = bits_in.len().min(bits_out.len()).saturating_sub(skip);
    let errors: usize = (0..n)
        .filter(|&i| bits_in[i + skip] != bits_out[i + skip])
        .count();
    assert_eq!(errors, 0, "BER non-zero: {} errors in {} bits", errors, n);
}

#[test]
fn roundtrip_bpsk31_text() {
    let text = b"HELLO TEST";
    let fs = 8000.0;
    let mut modulator = Bpsk31Mod::new(fs, 0.0, 1.0);
    // Use a long postamble to ensure the last character is flushed through
    // the Varicode decoder's "00" boundary detection.
    let iq = modulator.modulate_text(text, 32, 64);

    // Demodulate
    let mut demod = Bpsk31Demod::new(fs, 0.0, 1.0);
    let mut soft = vec![0.0f32; iq.len()];
    let wr = demod.process(&iq, &mut soft);
    soft.truncate(wr.out_written);

    // Hard decide
    let mut bits = vec![0u8; soft.len()];
    let dr = Bpsk31Decider::new().process(&soft, &mut bits);
    bits.truncate(dr.out_written);

    // Varicode decode -- push "00" at end to flush the last character.
    let mut vdec = VaricodeDecoder::new();
    for &b in &bits {
        vdec.push_bit(b);
    }
    vdec.push_bit(0);
    vdec.push_bit(0);
    let mut decoded = Vec::new();
    while let Some(c) = vdec.pop_char() {
        decoded.push(c);
    }

    // The decoded text should contain the original somewhere (leading preamble skipped).
    let decoded_str = String::from_utf8_lossy(&decoded);
    let expected = String::from_utf8_lossy(text);
    assert!(
        decoded_str.contains(expected.as_ref()),
        "decoded '{}' does not contain '{}'",
        decoded_str,
        expected
    );
}

#[test]
fn roundtrip_bpsk31_pulse_zero_crossing() {
    // A phase-flip symbol (bit=0) should have near-zero amplitude at the midpoint.
    let fs = 8000.0;
    let sps = psk31_sps(fs);
    let mut modulator = Bpsk31Mod::new(fs, 0.0, 1.0);

    // Modulate a single phase-flip bit (bit = 0).
    // We need a previous symbol first so the cross-fade is defined.
    // Use two bits: [1, 0] -- first bit is no-flip (steady), second is flip.
    let iq = modulator.modulate_bits(&[1u8, 0u8]);
    // The second symbol (indices sps..2*sps) is the phase-flip symbol.
    let mid = sps + sps / 2;
    let sample = iq[mid];
    let amplitude = (sample.re * sample.re + sample.im * sample.im).sqrt();
    assert!(
        amplitude < 0.15,
        "expected near-zero at phase-flip midpoint, got amplitude {}",
        amplitude
    );
}

#[test]
fn roundtrip_bpsk31_no_flip_constant() {
    // A no-flip symbol (bit=1) should have approximately constant amplitude.
    let fs = 8000.0;
    let sps = psk31_sps(fs);
    let mut modulator = Bpsk31Mod::new(fs, 0.0, 1.0);

    // All 1-bits: no phase changes.
    let iq = modulator.modulate_bits(&[1u8; 4]);
    // Check the middle two symbols have consistent amplitude.
    let amp: Vec<f32> = iq[sps..3*sps]
        .iter()
        .map(|s| (s.re * s.re + s.im * s.im).sqrt())
        .collect();
    let mean = amp.iter().sum::<f32>() / amp.len() as f32;
    let var = amp.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / amp.len() as f32;
    assert!(
        var < 0.01,
        "expected constant amplitude for no-flip symbols, variance = {}",
        var
    );
}

// -- QPSK31 roundtrip tests ------------------------------------------------

#[test]
fn roundtrip_qpsk31_bits_noiseless() {
    let bits_in: Vec<u8> = (0..64).map(|i| (i & 1) as u8).collect();

    let fs = 8000.0;
    let mut modulator = Qpsk31Mod::new(fs, 0.0, 1.0);
    let iq = modulator.modulate_bits(&bits_in);

    let mut demod = Qpsk31Demod::new(fs, 0.0, 1.0);
    let mut soft = vec![0.0f32; bits_in.len() * 4]; // 2 per symbol, generous
    let wr = demod.process(&iq, &mut soft);
    soft.truncate(wr.out_written);

    let mut decider = Qpsk31Decider::new();
    decider.process(&soft, &mut []);
    let mut bits_out = Vec::new();
    decider.flush(&mut bits_out);

    // Compare ignoring first few bits (Viterbi startup).
    let skip = 5;
    let n = bits_in.len().min(bits_out.len()).saturating_sub(skip);
    let errors: usize = (0..n)
        .filter(|&i| bits_in[i + skip] != bits_out[i + skip])
        .count();
    assert_eq!(errors, 0, "BER non-zero: {} errors in {} bits", errors, n);
}

#[test]
fn roundtrip_qpsk31_text() {
    let text = b"HELLO TEST";
    let fs = 8000.0;
    let mut modulator = Qpsk31Mod::new(fs, 0.0, 1.0);
    let iq = modulator.modulate_text(text, 32, 64);

    let mut demod = Qpsk31Demod::new(fs, 0.0, 1.0);
    let mut soft = vec![0.0f32; iq.len()];
    let wr = demod.process(&iq, &mut soft);
    soft.truncate(wr.out_written);

    let mut decider = Qpsk31Decider::new();
    decider.process(&soft, &mut []);
    let mut decoded_bits = Vec::new();
    decider.flush(&mut decoded_bits);

    let mut vdec = VaricodeDecoder::new();
    for &b in &decoded_bits {
        vdec.push_bit(b);
    }
    vdec.push_bit(0);
    vdec.push_bit(0);
    let mut decoded = Vec::new();
    while let Some(c) = vdec.pop_char() {
        decoded.push(c);
    }

    let decoded_str = String::from_utf8_lossy(&decoded);
    let expected = String::from_utf8_lossy(text);
    assert!(
        decoded_str.contains(expected.as_ref()),
        "QPSK31 decoded '{}' does not contain '{}'",
        decoded_str,
        expected
    );
}

// -- Full ASCII roundtrip --------------------------------------------------

/// Modulate -> demodulate -> varicode roundtrip for ALL 128 ASCII code points.
/// Verifies that every byte 0-127 survives the full BPSK31 signal chain at
/// baseband (carrier = 0 Hz, no noise).
#[test]
fn roundtrip_bpsk31_all_ascii() {
    let text: Vec<u8> = (0u8..128u8).collect();
    let fs = 8000.0;
    let mut modulator = Bpsk31Mod::new(fs, 0.0, 1.0);
    let iq = modulator.modulate_text(&text, 64, 64);

    let mut demod = Bpsk31Demod::new(fs, 0.0, 1.0);
    let mut soft = vec![0.0f32; iq.len()];
    let wr = demod.process(&iq, &mut soft);
    soft.truncate(wr.out_written);

    let mut bits = vec![0u8; soft.len()];
    let dr = Bpsk31Decider::new().process(&soft, &mut bits);
    bits.truncate(dr.out_written);

    let mut vdec = VaricodeDecoder::new();
    for &b in &bits {
        vdec.push_bit(b);
    }
    vdec.push_bit(0);
    vdec.push_bit(0);
    let mut decoded = Vec::new();
    while let Some(c) = vdec.pop_char() {
        decoded.push(c);
    }

    assert_eq!(decoded.len(), 128,
        "expected 128 decoded bytes, got {} -- decoded: {:?}",
        decoded.len(), &decoded[..decoded.len().min(20)]);
    for (i, (&got, &exp)) in decoded.iter().zip(text.iter()).enumerate() {
        assert_eq!(got, exp,
            "mismatch at index {}: expected 0x{:02X}, got 0x{:02X}", i, exp, got);
    }
}

// -- PSK31 sync roundtrip tests --------------------------------------------

#[test]
fn roundtrip_psk31_sync_finds_bpsk31() {
    use orion_sdr::sync::psk31_sync::psk31_sync;
    use orion_sdr::modulate::psk31::PSK31_BAUD;

    let fs = 8000.0;
    let base_hz = 900.0;
    // Use a carrier that aligns exactly with a waterfall bin (base + k * baud).
    let carrier_hz = base_hz + 3.0 * PSK31_BAUD; // 993.75 Hz, bin 3

    let text = b"CQ CQ";

    // Modulate BPSK31 at carrier_hz.
    let mut modulator = Bpsk31Mod::new(fs, carrier_hz, 1.0);
    let iq = modulator.modulate_text(text, 64, 32);

    // Build a 4-second IQ buffer with the signal embedded at 0.
    let total_samples = (fs * 4.0) as usize;
    let mut buf = vec![C32::new(0.0, 0.0); total_samples];
    for (i, s) in iq.iter().enumerate() {
        if i < total_samples {
            buf[i] = *s;
        }
    }

    // Run sync.
    let results = psk31_sync(&buf, fs, base_hz, base_hz + 200.0, 4, 3.0, 256, 5);

    assert!(!results.is_empty(), "psk31_sync should find the BPSK31 carrier");
    let best = &results[0];
    assert!(
        (best.carrier_hz - carrier_hz).abs() < 40.0,
        "carrier_hz {} too far from expected {}",
        best.carrier_hz,
        carrier_hz
    );
    assert!(!best.soft_bits.is_empty(), "soft_bits should not be empty");
}

#[test]
fn roundtrip_psk31_sync_finds_qpsk31() {
    use orion_sdr::sync::psk31_sync::psk31_sync;
    use orion_sdr::modulate::psk31::PSK31_BAUD;

    let fs = 8000.0;
    let base_hz = 1400.0;
    // Exact bin alignment: base + 2 * baud = 1462.5 Hz
    let carrier_hz = base_hz + 2.0 * PSK31_BAUD; // 1462.5 Hz, bin 2

    let text = b"TEST";

    let mut modulator = Qpsk31Mod::new(fs, carrier_hz, 1.0);
    let iq = modulator.modulate_text(text, 64, 32);

    let total_samples = (fs * 4.0) as usize;
    let mut buf = vec![C32::new(0.0, 0.0); total_samples];
    for (i, s) in iq.iter().enumerate() {
        if i < total_samples {
            buf[i] = *s;
        }
    }

    let results = psk31_sync(&buf, fs, base_hz, base_hz + 200.0, 4, 3.0, 256, 5);

    assert!(!results.is_empty(), "psk31_sync should find the QPSK31 carrier");
    let best = &results[0];
    assert!(
        (best.carrier_hz - carrier_hz).abs() < 40.0,
        "carrier_hz {} too far from expected {}",
        best.carrier_hz,
        carrier_hz
    );
}

// -- SNR sensitivity regression tests -----------------------------------------

const SNR_FS: f32 = 8_000.0;
const REF_BW_HZ: f32 = 2_500.0;
const SIG_POWER: f32 = 1.0;

fn snr_to_noise_power(snr_db: f32) -> f32 {
    SIG_POWER * SNR_FS / (REF_BW_HZ * 10.0_f32.powf(snr_db / 10.0))
}

fn try_bpsk31(text: &[u8], snr_db: f32, seed: u64) -> bool {
    let base_hz = 900.0_f32;
    let carrier_hz = base_hz + 3.0 * PSK31_BAUD;

    let sig_iq = Bpsk31Mod::new(SNR_FS, carrier_hz, 1.0)
        .modulate_text(text, 64, 32);

    let sig_len = sig_iq.len();
    let total = sig_len + SNR_FS as usize;
    let mut buf = sig_iq;
    buf.resize(total, C32::new(0.0, 0.0));
    add_awgn(&mut buf, snr_to_noise_power(snr_db), seed);

    let results = psk31_sync(&buf, SNR_FS, base_hz, base_hz + 200.0, 4, 3.0, 32, 5);
    if results.is_empty() { return false; }
    if (results[0].carrier_hz - carrier_hz).abs() > 40.0 { return false; }

    let found_hz = results[0].carrier_hz;
    let mut demod = Bpsk31Demod::new(SNR_FS, found_hz, 1.0);
    let max_syms = sig_len / (SNR_FS / PSK31_BAUD).round() as usize + 2;
    let mut soft = vec![0.0f32; max_syms];
    let wr = demod.process(&buf[..sig_len], &mut soft);
    soft.truncate(wr.out_written);

    let mut bits = vec![0u8; soft.len()];
    let dr = Bpsk31Decider::new().process(&soft, &mut bits);
    bits.truncate(dr.out_written);

    let mut vdec = VaricodeDecoder::new();
    for &b in &bits { vdec.push_bit(b); }
    vdec.push_bit(0); vdec.push_bit(0);
    let mut decoded = Vec::new();
    while let Some(c) = vdec.pop_char() { decoded.push(c); }

    decoded.windows(text.len()).any(|w| w == text)
}

fn try_qpsk31(text: &[u8], snr_db: f32, seed: u64) -> bool {
    let base_hz = 1400.0_f32;
    let carrier_hz = base_hz + 2.0 * PSK31_BAUD;

    let sig_iq = Qpsk31Mod::new(SNR_FS, carrier_hz, 1.0)
        .modulate_text(text, 64, 32);

    let sig_len = sig_iq.len();
    let total = sig_len + SNR_FS as usize;
    let mut buf = sig_iq;
    buf.resize(total, C32::new(0.0, 0.0));
    add_awgn(&mut buf, snr_to_noise_power(snr_db), seed);

    let results = psk31_sync(&buf, SNR_FS, base_hz, base_hz + 200.0, 4, 3.0, 32, 5);
    if results.is_empty() { return false; }
    let best = results.iter().min_by(|a, b| {
        let da = (a.carrier_hz - carrier_hz).abs();
        let db = (b.carrier_hz - carrier_hz).abs();
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    }).unwrap();
    if (best.carrier_hz - carrier_hz).abs() > 2.0 * PSK31_BAUD { return false; }

    let sps = (SNR_FS / PSK31_BAUD).round() as usize;
    let _ = (best, sps);
    let mut demod = Qpsk31Demod::new(SNR_FS, carrier_hz, 1.0);
    let max_soft = (sig_len / sps + 2) * 2;
    let mut soft = vec![0.0f32; max_soft];
    let wr = demod.process(&buf[..sig_len], &mut soft);
    soft.truncate(wr.out_written);

    let mut decider = Qpsk31Decider::new();
    decider.process(&soft, &mut []);
    let mut decoded_bits = Vec::new();
    decider.flush(&mut decoded_bits);

    let mut vdec = VaricodeDecoder::new();
    for &b in &decoded_bits { vdec.push_bit(b); }
    vdec.push_bit(0); vdec.push_bit(0);
    let mut decoded = Vec::new();
    while let Some(c) = vdec.pop_char() { decoded.push(c); }

    decoded.windows(text.len()).any(|w| w == text)
}

#[test]
fn bpsk31_decodes_at_minus_5db_snr_2500hz() {
    let snr_db = -5.0_f32;
    let text = b"CQ TEST";
    let noise_power = snr_to_noise_power(snr_db);
    assert!(
        try_bpsk31(text, snr_db, 0x1234_5678_9ABC_DEF0),
        "BPSK31 failed to decode at {} dB SNR/2500 Hz (noise_power={:.5})",
        snr_db, noise_power
    );
}

#[test]
fn qpsk31_decodes_at_minus_6db_snr_2500hz() {
    let snr_db = -6.0_f32;
    let text = b"CQ TEST";
    let noise_power = snr_to_noise_power(snr_db);
    assert!(
        try_qpsk31(text, snr_db, 0xFEDC_BA98_7654_3210),
        "QPSK31 failed to decode at {} dB SNR/2500 Hz (noise_power={:.5})",
        snr_db, noise_power
    );
}

// -- Psk31Stream roundtrip tests ----------------------------------------------

const STREAM_FS: f32 = 48_000.0;
const STREAM_CARRIER: f32 = 1000.0;

fn message_to_bits(msg: &str) -> Vec<u8> {
    let mut bits = Vec::new();
    bits.extend(std::iter::repeat_n(0u8, 32));
    for &byte in msg.as_bytes() {
        let (cw, len) = varicode_encode(byte);
        for i in (0..len).rev() {
            bits.push(((cw >> i) & 1) as u8);
        }
        bits.push(0);
        bits.push(0);
    }
    bits.extend(std::iter::repeat_n(0u8, 32));
    bits
}

fn stream_modulate_bpsk(bits: &[u8]) -> Vec<C32> {
    let sps = psk31_sps(STREAM_FS);
    let mut modulator = Bpsk31Mod::new(STREAM_FS, STREAM_CARRIER, 1.0);
    let mut iq = vec![C32::new(0.0, 0.0); bits.len() * sps + sps];
    let wr = modulator.process(bits, &mut iq);
    iq.truncate(wr.out_written);
    iq
}

fn stream_modulate_qpsk(bits: &[u8]) -> Vec<C32> {
    let sps = psk31_sps(STREAM_FS);
    let mut modulator = Qpsk31Mod::new(STREAM_FS, STREAM_CARRIER, 1.0);
    let mut iq = vec![C32::new(0.0, 0.0); bits.len() * sps + sps];
    let wr = modulator.process(bits, &mut iq);
    iq.truncate(wr.out_written);
    iq
}

#[test]
fn bpsk31_stream_decodes_hello() {
    let msg = "HELLO";
    let bits = message_to_bits(msg);
    let iq = stream_modulate_bpsk(&bits);

    let mut stream = Psk31Stream::new_bpsk(STREAM_FS, STREAM_CARRIER, 1.0);
    let mut decoded = stream.feed(&iq);
    decoded += &stream.flush();

    assert!(decoded.contains(msg),
        "expected '{}' in decoded output '{}'", msg, decoded);
}

#[test]
fn qpsk31_stream_decodes_hello() {
    let msg = "HELLO";
    let bits = message_to_bits(msg);
    let iq = stream_modulate_qpsk(&bits);

    let mut stream = Psk31Stream::new_qpsk(STREAM_FS, STREAM_CARRIER, 1.0);
    let mut decoded = stream.feed(&iq);
    decoded += &stream.flush();

    assert!(decoded.contains(msg),
        "expected '{}' in decoded output '{}'", msg, decoded);
}

#[test]
fn bpsk31_stream_decodes_all_printable_ascii() {
    let msg: String = (0x20u8..=0x7eu8).map(|b| b as char).collect();
    let bits = message_to_bits(&msg);
    let iq = stream_modulate_bpsk(&bits);

    let mut stream = Psk31Stream::new_bpsk(STREAM_FS, STREAM_CARRIER, 1.0);
    let mut decoded = stream.feed(&iq);
    decoded += &stream.flush();

    for ch in msg.chars() {
        assert!(decoded.contains(ch),
            "missing char '{}' (0x{:02x}) in decoded output", ch, ch as u8);
    }
}

#[test]
fn qpsk31_stream_decodes_all_printable_ascii() {
    let msg: String = (0x20u8..=0x7eu8).map(|b| b as char).collect();
    let bits = message_to_bits(&msg);
    let iq = stream_modulate_qpsk(&bits);

    let mut stream = Psk31Stream::new_qpsk(STREAM_FS, STREAM_CARRIER, 1.0);
    let mut decoded = stream.feed(&iq);
    decoded += &stream.flush();

    for ch in msg.chars() {
        assert!(decoded.contains(ch),
            "missing char '{}' (0x{:02x}) in decoded output", ch, ch as u8);
    }
}

#[test]
fn bpsk31_stream_incremental_feed() {
    let msg = "CQ CQ DE TEST";
    let bits = message_to_bits(msg);
    let iq = stream_modulate_bpsk(&bits);

    let mut stream = Psk31Stream::new_bpsk(STREAM_FS, STREAM_CARRIER, 1.0);
    let chunk_size = psk31_sps(STREAM_FS) * 4;
    let mut decoded = String::new();
    for chunk in iq.chunks(chunk_size) {
        decoded += &stream.feed(chunk);
    }
    decoded += &stream.flush();

    assert!(decoded.contains(msg),
        "expected '{}' in decoded output '{}'", msg, decoded);
}
