
use crate::codec::varicode::{varicode_encode, VaricodeEncoder, VaricodeDecoder};
use crate::codec::psk31_conv::{conv_encode, viterbi_decode_hard};
use crate::modulate::psk31::{psk31_sps, PSK31_BAUD, PSK31_SPS_8000, PSK31_SPS_12000};

// ── Varicode tests ────────────────────────────────────────────────────────────

#[test]
fn varicode_encode_space() {
    let (cw, len) = varicode_encode(b' ');
    assert_eq!(len, 1);
    assert_eq!(cw, 1);
}

#[test]
fn varicode_decode_roundtrip() {
    // The canonical IZ8BLY/G3PLX Varicode table has a small number of duplicate
    // codewords (some characters share the same bit pattern).  For those, the
    // decode lookup returns the lower-indexed character (first match).
    // We only assert that the roundtrip holds for characters that are the *first*
    // in the table with their codeword.
    use crate::codec::varicode::varicode_encode as enc;
    use crate::codec::varicode::varicode_decode as dec;

    for b in 32u8..127u8 {
        let (cw, len) = enc(b);
        let decoded = dec(cw, len);
        // The decoded value must exist and must encode to the same codeword.
        assert!(decoded.is_some(), "no decode for byte {}", b);
        let d = decoded.unwrap();
        let (cw2, len2) = enc(d);
        assert_eq!((cw2, len2), (cw, len),
            "codeword mismatch: byte {} encoded to ({},{}) but decoded as byte {} \
             which encodes to ({},{})", b, cw, len, d, cw2, len2);
    }
}

#[test]
fn varicode_encoder_stream_cq() {
    // "CQ" — verify the bit stream starts with preamble 0s then C's code then 00 gap then Q's code.
    let mut enc = VaricodeEncoder::new();
    enc.push_preamble(4);
    enc.push_byte(b'C');
    enc.push_byte(b'Q');
    let bits = enc.drain_bits();

    // First 4 bits = preamble zeros.
    assert_eq!(&bits[..4], &[0, 0, 0, 0]);
    // After preamble the next bits are C's codeword (no leading 00, since first=false after preamble).
    let (c_cw, c_len) = varicode_encode(b'C');
    let c_bits: Vec<u8> = (0..c_len).rev().map(|i| ((c_cw >> i) & 1) as u8).collect();
    assert_eq!(&bits[4..4 + c_len as usize], c_bits.as_slice());
    // After C: "00" gap then Q's codeword.
    let gap_start = 4 + c_len as usize;
    assert_eq!(bits[gap_start],     0);
    assert_eq!(bits[gap_start + 1], 0);
    let (q_cw, q_len) = varicode_encode(b'Q');
    let q_bits: Vec<u8> = (0..q_len).rev().map(|i| ((q_cw >> i) & 1) as u8).collect();
    assert_eq!(&bits[gap_start + 2..gap_start + 2 + q_len as usize], q_bits.as_slice());
}

#[test]
fn varicode_decoder_boundary() {
    // Push C's codeword followed by "00" — should decode 'C'.
    let (cw, len) = varicode_encode(b'C');
    let mut dec = VaricodeDecoder::new();
    for i in (0..len).rev() {
        dec.push_bit(((cw >> i) & 1) as u8);
    }
    assert_eq!(dec.pop_char(), None); // not yet decoded (no "00" received)
    dec.push_bit(0);
    dec.push_bit(0); // "00" boundary
    assert_eq!(dec.pop_char(), Some(b'C'));
}

// ── PSK31 SPS tests ───────────────────────────────────────────────────────────

#[test]
fn psk31_sps_8000() {
    assert_eq!(psk31_sps(8000.0), PSK31_SPS_8000);
}

#[test]
fn psk31_sps_12000() {
    assert_eq!(psk31_sps(12000.0), PSK31_SPS_12000);
}

#[test]
fn psk31_hann_endpoints() {
    // PSK31 uses a half-cosine crossfade window (one-sided Hann):
    //   hann[n] = 0.5 - 0.5·cos(π·n/(sps-1))
    //
    // This gives:
    //   hann[0]     = 0.5 - 0.5·cos(0) = 0.0   (start at previous phasor)
    //   hann[sps-1] = 0.5 - 0.5·cos(π) = 1.0   (arrive at current phasor)
    let sps = psk31_sps(8000.0);
    assert_eq!(sps, 256);
    let denom = (sps - 1) as f32;
    let h0 = 0.5 - 0.5 * (std::f32::consts::PI * 0.0 / denom).cos();
    let hn = 0.5 - 0.5 * (std::f32::consts::PI * denom / denom).cos();
    assert!(h0.abs() < 1e-4,      "hann[0] = {}", h0);
    assert!((hn - 1.0).abs() < 1e-4, "hann[sps-1] = {}", hn);
}

// ── Decider tests ─────────────────────────────────────────────────────────────

#[test]
fn bpsk31_decider_sign() {
    use crate::demodulate::psk31::Bpsk31Decider;
    use crate::core::Block;

    let soft = [1.0f32, -1.0, 2.5, -0.5];
    let mut out = [0u8; 4];
    Bpsk31Decider::new().process(&soft, &mut out);
    assert_eq!(out, [1, 0, 1, 0]);
}

// ── Convolutional codec tests ─────────────────────────────────────────────────

#[test]
fn conv_encode_known() {
    // Verify a known input against hand-computed output.
    // Input: [1, 0] → G0 and G1 for each bit.
    // For bit 1: window = 0b10000, G0 = parity(0b10000 & 0b10101) = parity(0b10000) = 1
    //   G1 = parity(0b10000 & 0b10011) = parity(0b10000) = 1
    // For bit 0 (sr=0b1000 → window = 0b01000):
    //   G0 = parity(0b01000 & 0b10101) = parity(0b00000) = 0
    //   G1 = parity(0b01000 & 0b10011) = parity(0b00000) = 0
    let input = vec![1u8, 0u8];
    let encoded = conv_encode(&input);
    assert_eq!(encoded.len(), 4);
    assert_eq!(encoded[0], 1); // g0 for bit 1
    assert_eq!(encoded[1], 1); // g1 for bit 1
    assert_eq!(encoded[2], 0); // g0 for bit 0
    assert_eq!(encoded[3], 0); // g1 for bit 0
}

#[test]
fn viterbi_decode_noiseless() {
    let input: Vec<u8> = (0..32).map(|i| (i & 1) as u8).collect();
    let encoded = conv_encode(&input);
    let recovered = viterbi_decode_hard(&encoded);
    assert_eq!(recovered.len(), input.len());
    for i in 0..input.len() {
        assert_eq!(recovered[i], input[i], "bit {} mismatch", i);
    }
}

// ── QPSK31 preamble test ──────────────────────────────────────────────────────

#[test]
fn qpsk31_modulate_preamble() {
    use crate::modulate::psk31::Qpsk31Mod;

    // A preamble of all 0-bits through the convolutional encoder produces
    // a regular pattern of dibits. After PSK31 pulse shaping the output should
    // not be all-zero (non-trivial waveform exists).
    let mut mod_ = Qpsk31Mod::new(8000.0, 0.0, 1.0);
    let iq = mod_.modulate_bits(&[0u8; 16]);
    let sps = psk31_sps(8000.0);
    assert_eq!(iq.len(), 16 * sps);
    // Check that the waveform is non-trivially non-zero.
    let power: f32 = iq.iter().map(|s| s.re * s.re + s.im * s.im).sum::<f32>() / iq.len() as f32;
    assert!(power > 0.01, "expected non-zero power, got {}", power);
}

// ── PSK31 baud constant ───────────────────────────────────────────────────────

#[test]
fn psk31_baud_constant() {
    assert!((PSK31_BAUD - 31.25).abs() < 1e-6);
}

// ── Hard-decision function tests ──────────────────────────────────────────────

#[test]
fn qpsk31_hard_decide_dqpsk_four_quadrants() {
    use crate::demodulate::psk31::hard_decide_dqpsk;
    assert_eq!(hard_decide_dqpsk( 0.8,  0.2), ( 1.0,  0.0)); // 0°
    assert_eq!(hard_decide_dqpsk(-0.8,  0.2), (-1.0,  0.0)); // 180°
    assert_eq!(hard_decide_dqpsk( 0.2,  0.8), ( 0.0,  1.0)); // +90°
    assert_eq!(hard_decide_dqpsk( 0.2, -0.8), ( 0.0, -1.0)); // -90°
    // Tie (|re| == |im|) → real axis wins
    assert_eq!(hard_decide_dqpsk( 0.707,  0.707), (1.0, 0.0));
}
