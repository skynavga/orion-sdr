
use num_complex::Complex32 as C32;
use crate::modulate::psk31::{Bpsk31Mod, Qpsk31Mod, psk31_sps};
use crate::demodulate::psk31::{Bpsk31Demod, Bpsk31Decider, Qpsk31Demod, Qpsk31Decider};
use crate::codec::varicode::VaricodeDecoder;
use crate::core::Block;

// ── BPSK31 roundtrip tests ────────────────────────────────────────────────────

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

    // Varicode decode — push "00" at end to flush the last character.
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
    // Use two bits: [1, 0] — first bit is no-flip (steady), second is flip.
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

// ── QPSK31 roundtrip tests ────────────────────────────────────────────────────

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
    decider.process(&soft, &mut vec![]);
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
    decider.process(&soft, &mut vec![]);
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

// ── PSK31 sync roundtrip tests ────────────────────────────────────────────────

#[test]
fn roundtrip_psk31_sync_finds_bpsk31() {
    use crate::sync::psk31_sync::psk31_sync;
    use crate::modulate::psk31::PSK31_BAUD;

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
    use crate::sync::psk31_sync::psk31_sync;
    use crate::modulate::psk31::PSK31_BAUD;

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
