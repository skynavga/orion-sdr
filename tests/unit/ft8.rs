// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0


use orion_sdr::modulate::{Ft8Mod, Ft8Frame};
use orion_sdr::modulate::ft8::{FT8_FRAME_LEN, FT8_TOTAL_SYMS, FT8_DATA_SYMS};
use orion_sdr::codec::ft8::{Ft8Codec, Ft8Bits, Ft8StreamDecoder};
use orion_sdr::codec::gray::{gray8_encode, gray8_decode};

#[test]
fn ft8_frame_length() {
    let tx = Ft8Mod::new(12_000.0, 1_000.0, 0.0, 1.0);
    let iq = tx.modulate(&Ft8Frame::zeros());
    assert_eq!(iq.len(), FT8_FRAME_LEN, "FT8 frame length mismatch");
}

#[test]
fn ft8_symbol_sequence_count() {
    let seq = Ft8Mod::build_symbol_sequence(&Ft8Frame::zeros());
    assert_eq!(seq.len(), FT8_TOTAL_SYMS);
    let sync_pos: [(usize, usize); 3] = [(0, 7), (36, 43), (72, 79)];
    let mut is_sync = [false; FT8_TOTAL_SYMS];
    for &(start, end) in &sync_pos { is_sync[start..end].fill(true); }
    let data_count = is_sync.iter().filter(|&&s| !s).count();
    assert_eq!(data_count, FT8_DATA_SYMS);
}

#[test]
fn ft8_costas_positions_correct() {
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
    let tx = Ft8Mod::new(12_000.0, 1_000.0, 0.0, 1.0);
    let iq = tx.modulate(&Ft8Frame::zeros());
    let power: f32 = iq.iter().map(|z| z.norm_sqr()).sum::<f32>() / (FT8_FRAME_LEN as f32);
    assert!((power - 1.0).abs() < 0.01, "FT8 IQ power deviates from 1.0: {}", power);
}

// -- FT8 codec tests ----------------------------------------------------------

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

// -- Gray code tests ----------------------------------------------------------

#[test]
fn ft8_gray_roundtrip() {
    for i in 0u8..8 {
        assert_eq!(gray8_decode(gray8_encode(i)), i, "FT8 Gray roundtrip failed for {i}");
    }
}

#[test]
fn ft8_gray_table_matches_spec() {
    let expected: [u8; 8] = [0, 1, 3, 2, 5, 6, 4, 7];
    for (i, &e) in expected.iter().enumerate() {
        assert_eq!(gray8_encode(i as u8), e, "FT8 Gray table mismatch at {i}");
    }
}

// -- Ft8StreamDecoder unit tests ----------------------------------------------

#[test]
fn stream_decoder_starts_empty() {
    let dec = Ft8StreamDecoder::new_ft8(12_000.0, 900.0, 1_100.0, 5);
    assert_eq!(dec.len(), 0);
    assert!(dec.is_empty());
}

#[test]
fn stream_decoder_accumulates_without_decode() {
    use num_complex::Complex32;
    let mut dec = Ft8StreamDecoder::new_ft8(12_000.0, 900.0, 1_100.0, 5);
    let chunk = vec![Complex32::new(0.0, 0.0); 1024];
    let results = dec.feed(&chunk);
    assert!(results.is_empty(), "Should not decode before frame_len samples");
    assert_eq!(dec.len(), 1024);
}

#[test]
fn stream_decoder_clear_resets_buffer() {
    use num_complex::Complex32;
    let mut dec = Ft8StreamDecoder::new_ft8(12_000.0, 900.0, 1_100.0, 5);
    dec.feed(&vec![Complex32::new(0.1, 0.0); 4096]);
    assert!(!dec.is_empty());
    dec.clear();
    assert!(dec.is_empty());
    assert_eq!(dec.len(), 0);
}

#[test]
fn stream_decoder_flush_empty_returns_nothing() {
    let mut dec = Ft8StreamDecoder::new_ft8(12_000.0, 900.0, 1_100.0, 5);
    let results = dec.flush();
    assert!(results.is_empty());
}

#[test]
fn stream_decoder_feed_full_frame_noiseless() {
    let base_hz = 1_000.0_f32;
    let payload: Ft8Bits = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60];
    let frame = Ft8Codec::encode(&payload);
    let iq = Ft8Mod::new(12_000.0, base_hz, 0.0, 1.0).modulate(&frame);

    let mut dec = Ft8StreamDecoder::new_ft8(
        12_000.0,
        base_hz - 6.25,
        base_hz + 50.0 + 6.25,
        5,
    );

    // Feed in chunks; final chunk triggers decode.
    let chunk_size = 8192;
    let mut results = Vec::new();
    let mut pos = 0;
    while pos < iq.len() {
        let end = (pos + chunk_size).min(iq.len());
        let r = dec.feed(&iq[pos..end]);
        results.extend(r);
        pos = end;
    }
    // If feed didn't trigger (buffer < frame_len after last chunk), flush.
    if results.is_empty() {
        results = dec.flush();
    }

    assert!(!results.is_empty(), "Ft8StreamDecoder: no decode result from noiseless frame");
    // The payload encodes to a NonStd message (raw bytes) — just assert something decoded.
    let _ = &results[0].message;
}

#[test]
fn stream_decoder_flush_partial_frame_noiseless() {
    let base_hz = 1_000.0_f32;
    let payload: Ft8Bits = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x20];
    let frame = Ft8Codec::encode(&payload);
    let iq = Ft8Mod::new(12_000.0, base_hz, 0.0, 1.0).modulate(&frame);

    // Feed only 90% of the frame — decoder won't auto-trigger.
    let partial_len = iq.len() * 9 / 10;
    let mut dec = Ft8StreamDecoder::new_ft8(
        12_000.0,
        base_hz - 6.25,
        base_hz + 50.0 + 6.25,
        5,
    );
    let r = dec.feed(&iq[..partial_len]);
    assert!(r.is_empty(), "Should not decode before frame_len reached");
    assert_eq!(dec.len(), partial_len);
    // flush() should still attempt a decode on the partial buffer.
    let results = dec.flush();
    // A partial frame at high SNR may or may not decode — just assert no panic.
    let _ = results;
}
