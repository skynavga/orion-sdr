// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0


// CRC tests (moved from src/codec/crc.rs)

use orion_sdr::codec::crc::{ft8_crc14, ft8_add_crc, ft8_extract_crc};

fn recompute_crc(payload: &[u8; 10]) -> u16 {
    let mut buf = [0u8; 12];
    buf[..10].copy_from_slice(payload);
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

// LDPC tests (moved from src/codec/ldpc.rs)

use orion_sdr::codec::ldpc::{ldpc_encode, ldpc_decode_soft, ldpc_count_errors, K_BYTES, N_BYTES, N, K};

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

