// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0


use orion_sdr::modulate::{Ft4Mod, Ft4Frame};
use orion_sdr::modulate::ft4::{FT4_FRAME_LEN, FT4_TOTAL_SYMS, FT4_DATA_SYMS};
use orion_sdr::codec::ft4::{Ft4Codec, Ft4Bits};
use orion_sdr::codec::gray::{gray4_encode, gray4_decode};

#[test]
fn ft4_frame_length() {
    let tx = Ft4Mod::new(12_000.0, 1_000.0, 0.0, 1.0);
    let iq = tx.modulate(&Ft4Frame::zeros());
    assert_eq!(iq.len(), FT4_FRAME_LEN, "FT4 frame length mismatch");
}

#[test]
fn ft4_symbol_sequence_count() {
    let seq = Ft4Mod::build_symbol_sequence(&Ft4Frame::zeros());
    assert_eq!(seq.len(), FT4_TOTAL_SYMS);
    let sync_pos: [(usize, usize); 4] = [(1, 5), (34, 38), (67, 71), (100, 104)];
    let mut is_reserved = [false; FT4_TOTAL_SYMS];
    is_reserved[0] = true;
    is_reserved[104] = true;
    for &(start, end) in &sync_pos { is_reserved[start..end].fill(true); }
    let data_count = is_reserved.iter().filter(|&&s| !s).count();
    assert_eq!(data_count, FT4_DATA_SYMS);
}

#[test]
fn ft4_costas_positions_correct() {
    let costas: [[u8; 4]; 4] = [[0,1,3,2],[1,0,2,3],[2,3,1,0],[3,2,0,1]];
    let sync_starts = [1usize, 34, 67, 100];
    let seq = Ft4Mod::build_symbol_sequence(&Ft4Frame::zeros());
    assert_eq!(seq[0], 0, "FT4 ramp at position 0 should be tone 0");
    assert_eq!(seq[104], 0, "FT4 ramp at position 104 should be tone 0");
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
    let tx = Ft4Mod::new(12_000.0, 1_000.0, 0.0, 1.0);
    let iq = tx.modulate(&Ft4Frame::zeros());
    let power: f32 = iq.iter().map(|z| z.norm_sqr()).sum::<f32>() / (FT4_FRAME_LEN as f32);
    assert!((power - 1.0).abs() < 0.01, "FT4 IQ power deviates from 1.0: {}", power);
}

// -- FT4 codec tests ----------------------------------------------------------

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

// -- Gray code tests ----------------------------------------------------------

#[test]
fn ft4_gray_roundtrip() {
    for i in 0u8..4 {
        assert_eq!(gray4_decode(gray4_encode(i)), i, "FT4 Gray roundtrip failed for {i}");
    }
}

#[test]
fn ft4_gray_table_matches_spec() {
    let expected: [u8; 4] = [0, 1, 3, 2];
    for (i, &e) in expected.iter().enumerate() {
        assert_eq!(gray4_encode(i as u8), e, "FT4 Gray table mismatch at {i}");
    }
}
