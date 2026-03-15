
use crate::modulate::{Ft4Mod, Ft4Frame};
use crate::demodulate::Ft4Demod;
use crate::modulate::ft4::{FT4_DATA_SYMS, FT4_TONES};
use crate::codec::ft4::{Ft4Codec, Ft4Bits};
use crate::sync::ft4_sync;
use super::{make_ft4_test_buffer};

#[test]
fn roundtrip_ft4_noiseless() {
    let mut tones = [0u8; FT4_DATA_SYMS];
    for (i, t) in tones.iter_mut().enumerate() {
        *t = (i % FT4_TONES) as u8;
    }
    let frame_in = Ft4Frame::new(tones);

    let tx = Ft4Mod::new(12_000.0, 1_000.0, 0.0, 1.0);
    let iq = tx.modulate(&frame_in);

    let rx = Ft4Demod::new(12_000.0, 1_000.0);
    let frame_out = rx.demodulate(&iq).expect("FT4 demodulate returned None");

    assert_eq!(frame_in, frame_out, "FT4 noiseless roundtrip failed");
}

#[test]
fn roundtrip_ft4_all_tones() {
    let frame_in = Ft4Frame::new([3u8; FT4_DATA_SYMS]);
    let tx = Ft4Mod::new(12_000.0, 100.0, 0.0, 1.0);
    let iq = tx.modulate(&frame_in);
    let rx = Ft4Demod::new(12_000.0, 100.0);
    let frame_out = rx.demodulate(&iq).expect("FT4 all-3 demodulate failed");
    assert_eq!(frame_in, frame_out, "FT4 all-tone-3 roundtrip failed");
}

#[test]
fn roundtrip_ft4_codec_noiseless() {
    let payload: Ft4Bits = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60];
    let frame = Ft4Codec::encode(&payload);

    let tx = Ft4Mod::new(12_000.0, 1_000.0, 0.0, 1.0);
    let iq = tx.modulate(&frame);

    let rx = Ft4Demod::new(12_000.0, 1_000.0);
    let frame_out = rx.demodulate(&iq).expect("FT4 demodulate returned None");

    let decoded = Ft4Codec::decode_hard(&frame_out)
        .expect("FT4 codec decode failed (noiseless)");
    assert_eq!(payload, decoded, "FT4 full-stack codec roundtrip failed");
}

#[test]
fn roundtrip_ft4_codec_zeros() {
    let payload: Ft4Bits = [0u8; 10];
    let frame = Ft4Codec::encode(&payload);
    let iq = Ft4Mod::new(12_000.0, 500.0, 0.0, 1.0).modulate(&frame);
    let frame_out = Ft4Demod::new(12_000.0, 500.0).demodulate(&iq).unwrap();
    let decoded = Ft4Codec::decode_hard(&frame_out).expect("FT4 codec decode failed (zeros)");
    assert_eq!(payload, decoded);
}

#[test]
fn sync_ft4_noiseless_aligned() {
    let base_hz = 1_000.0_f32;
    let (buf, payload) = make_ft4_test_buffer(0, base_hz, 0.0);

    let results = ft4_sync(
        &buf,
        12_000.0,
        base_hz - 20.833,
        base_hz + 100.0 + 20.833,
        0,
        0,
        5,
    );

    assert!(!results.is_empty(), "FT4 sync returned no candidates");
    let best = &results[0];
    let decoded = Ft4Codec::decode_soft(&best.llr)
        .expect("FT4 sync+decode_soft failed (noiseless, aligned)");
    assert_eq!(payload, decoded, "FT4 sync noiseless aligned: payload mismatch");
}

#[test]
fn sync_ft4_noiseless_time_offset() {
    use crate::modulate::ft4::FT4_SAMPLES_PER_SYM;

    let base_hz = 900.0_f32;
    let offset = 3 * FT4_SAMPLES_PER_SYM;
    let (buf, payload) = make_ft4_test_buffer(offset, base_hz, 0.0);

    let results = ft4_sync(
        &buf,
        12_000.0,
        base_hz - 20.833,
        base_hz + 100.0 + 20.833,
        0,
        5,
        5,
    );

    assert!(!results.is_empty(), "FT4 sync (time offset) returned no candidates");
    let best = &results[0];
    let decoded = Ft4Codec::decode_soft(&best.llr)
        .expect("FT4 sync+decode_soft failed (noiseless, time offset)");
    assert_eq!(payload, decoded, "FT4 sync noiseless time-offset: payload mismatch");
}
