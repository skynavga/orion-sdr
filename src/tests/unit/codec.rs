
// CRC tests (moved from src/codec/crc.rs)

use crate::codec::crc::{ft8_crc14, ft8_add_crc, ft8_extract_crc};

fn recompute_crc(payload: &[u8; 10]) -> u16 {
    let mut buf = [0u8; 12];
    for i in 0..10 { buf[i] = payload[i]; }
    buf[9] &= 0xF8;
    ft8_crc14(&buf, 82)
}

#[test]
fn crc_roundtrip_all_zeros() {
    let payload = [0u8; 10];
    let mut a91 = [0u8; 12];
    ft8_add_crc(&payload, &mut a91);
    let extracted = ft8_extract_crc(&a91);
    let computed = recompute_crc(&payload);
    assert_eq!(extracted, computed);
}

#[test]
fn crc_roundtrip_all_ones() {
    let mut payload = [0xFFu8; 10];
    payload[9] = 0xF8;
    let mut a91 = [0u8; 12];
    ft8_add_crc(&payload, &mut a91);
    let extracted = ft8_extract_crc(&a91);
    let computed = recompute_crc(&payload);
    assert_eq!(extracted, computed);
}

#[test]
fn crc_is_nonzero_for_nonzero_payload() {
    let mut payload = [0u8; 10];
    payload[0] = 0xAB;
    let mut a91 = [0u8; 12];
    ft8_add_crc(&payload, &mut a91);
    let extracted = ft8_extract_crc(&a91);
    assert_ne!(extracted, 0, "CRC should be non-zero for non-zero payload");
}

#[test]
fn crc_changes_with_payload() {
    let mut payload_a = [0u8; 10];
    payload_a[0] = 0x01;
    let mut payload_b = [0u8; 10];
    payload_b[0] = 0x02;
    let mut a91_a = [0u8; 12];
    let mut a91_b = [0u8; 12];
    ft8_add_crc(&payload_a, &mut a91_a);
    ft8_add_crc(&payload_b, &mut a91_b);
    assert_ne!(ft8_extract_crc(&a91_a), ft8_extract_crc(&a91_b));
}

// Gray tests (moved from src/codec/gray.rs)

use crate::codec::gray::{gray8_encode, gray8_decode, gray4_encode, gray4_decode};

#[test]
fn ft8_gray_roundtrip() {
    for i in 0u8..8 {
        assert_eq!(gray8_decode(gray8_encode(i)), i, "FT8 Gray roundtrip failed for {i}");
    }
}

#[test]
fn ft4_gray_roundtrip() {
    for i in 0u8..4 {
        assert_eq!(gray4_decode(gray4_encode(i)), i, "FT4 Gray roundtrip failed for {i}");
    }
}

#[test]
fn ft8_gray_table_matches_spec() {
    let expected: [u8; 8] = [0, 1, 3, 2, 5, 6, 4, 7];
    for (i, &e) in expected.iter().enumerate() {
        assert_eq!(gray8_encode(i as u8), e, "FT8 Gray table mismatch at {i}");
    }
}

#[test]
fn ft4_gray_table_matches_spec() {
    let expected: [u8; 4] = [0, 1, 3, 2];
    for (i, &e) in expected.iter().enumerate() {
        assert_eq!(gray4_encode(i as u8), e, "FT4 Gray table mismatch at {i}");
    }
}

// LDPC tests (moved from src/codec/ldpc.rs)

use crate::codec::ldpc::{ldpc_encode, ldpc_decode_soft, ldpc_count_errors, K_BYTES, N_BYTES, N, K};

fn make_a91(payload: &[u8; 10]) -> [u8; K_BYTES] {
    let mut a91 = [0u8; K_BYTES];
    ft8_add_crc(payload, &mut a91);
    a91
}

#[test]
fn ldpc_encode_syndrome_zero() {
    let payload = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x00u8];
    let a91 = make_a91(&payload);
    let mut codeword = [0u8; N_BYTES];
    ldpc_encode(&a91, &mut codeword);
    let mut hard = [0u8; N];
    for i in 0..N {
        hard[i] = (codeword[i / 8] >> (7 - (i % 8))) & 1;
    }
    assert_eq!(ldpc_count_errors(&hard), 0, "Syndrome check failed on encoded codeword");
}

#[test]
fn ldpc_hard_roundtrip() {
    let payload = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60u8];
    let a91 = make_a91(&payload);
    let mut codeword = [0u8; N_BYTES];
    ldpc_encode(&a91, &mut codeword);
    let mut llr = [0.0f32; N];
    for i in 0..N {
        let bit = (codeword[i / 8] >> (7 - (i % 8))) & 1;
        llr[i] = if bit == 0 { 10.0 } else { -10.0 };
    }
    let mut plain = [0u8; N];
    let errors = ldpc_decode_soft(&llr, 20, &mut plain);
    assert_eq!(errors, 0, "LDPC decode failed with hard LLRs");
    for i in 0..K {
        let expected = (a91[i / 8] >> (7 - (i % 8))) & 1;
        assert_eq!(plain[i], expected, "Bit {i} mismatch after decode");
    }
    let mut recovered_a91 = [0u8; K_BYTES];
    for i in 0..K {
        if plain[i] == 1 {
            recovered_a91[i / 8] |= 0x80 >> (i % 8);
        }
    }
    let crc_in_message = ft8_extract_crc(&recovered_a91);
    let mut buf_for_crc = recovered_a91;
    buf_for_crc[9] &= 0xF8;
    buf_for_crc[10] = 0;
    buf_for_crc[11] = 0;
    let crc_recomputed = ft8_crc14(&buf_for_crc, 82);
    assert_eq!(crc_in_message, crc_recomputed, "CRC mismatch in decoded message");
}

#[test]
fn ldpc_encode_preserves_message_bits() {
    let payload = [0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0x00u8];
    let a91 = make_a91(&payload);
    let mut codeword = [0u8; N_BYTES];
    ldpc_encode(&a91, &mut codeword);
    assert_eq!(&codeword[..K_BYTES - 1], &a91[..K_BYTES - 1]);
    let msg_mask: u8 = 0xE0;
    assert_eq!(codeword[K_BYTES - 1] & msg_mask, a91[K_BYTES - 1] & msg_mask,
        "Top 3 bits of codeword byte 11 must match a91 byte 11");
}

// FT8 codec tests (moved from src/codec/ft8.rs)

use crate::codec::ft8::{Ft8Codec, Ft8Bits};

#[test]
fn ft8_codec_roundtrip_zeros() {
    let payload: Ft8Bits = [0u8; 10];
    let frame = Ft8Codec::encode(&payload);
    let decoded = Ft8Codec::decode_hard(&frame).expect("FT8 hard decode failed (zeros)");
    assert_eq!(payload, decoded);
}

#[test]
fn ft8_codec_roundtrip_all_ones() {
    let mut payload: Ft8Bits = [0xFFu8; 10];
    payload[9] = 0xF8;
    let frame = Ft8Codec::encode(&payload);
    let decoded = Ft8Codec::decode_hard(&frame).expect("FT8 hard decode failed (all-ones)");
    assert_eq!(payload, decoded);
}

#[test]
fn ft8_codec_roundtrip_pattern() {
    let payload: Ft8Bits = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60];
    let frame = Ft8Codec::encode(&payload);
    let decoded = Ft8Codec::decode_hard(&frame).expect("FT8 hard decode failed (pattern)");
    assert_eq!(payload, decoded);
}

#[test]
fn ft8_codec_frame_has_valid_tones() {
    let payload: Ft8Bits = [0x55; 10];
    let frame = Ft8Codec::encode(&payload);
    for &t in frame.0.iter() {
        assert!(t < 8, "Tone index out of range: {t}");
    }
}

// FT4 codec tests (moved from src/codec/ft4.rs)

use crate::codec::ft4::{Ft4Codec, Ft4Bits};

#[test]
fn ft4_codec_roundtrip_zeros() {
    let payload: Ft4Bits = [0u8; 10];
    let frame = Ft4Codec::encode(&payload);
    let decoded = Ft4Codec::decode_hard(&frame).expect("FT4 hard decode failed (zeros)");
    assert_eq!(payload, decoded);
}

#[test]
fn ft4_codec_roundtrip_pattern() {
    let payload: Ft4Bits = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60];
    let frame = Ft4Codec::encode(&payload);
    let decoded = Ft4Codec::decode_hard(&frame).expect("FT4 hard decode failed (pattern)");
    assert_eq!(payload, decoded);
}

#[test]
fn ft4_codec_frame_has_valid_tones() {
    let payload: Ft4Bits = [0xAA; 10];
    let frame = Ft4Codec::encode(&payload);
    for &t in frame.0.iter() {
        assert!(t < 4, "FT4 tone index out of range: {t}");
    }
}

#[test]
fn ft4_xor_scramble_changes_frame() {
    let payload_a: Ft4Bits = [0u8; 10];
    let payload_b: Ft4Bits = [0x4A, 0x5E, 0x89, 0xB4, 0xB0, 0x8A, 0x79, 0x55, 0xBE, 0x28];
    let frame_a = Ft4Codec::encode(&payload_a);
    let frame_b = Ft4Codec::encode(&payload_b);
    assert_ne!(frame_a, frame_b, "XOR scramble should change the encoded frame");
}
