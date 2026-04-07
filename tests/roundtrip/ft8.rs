// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0


use orion_sdr::modulate::{Ft8Mod, Ft8Frame};
use orion_sdr::demodulate::Ft8Demod;
use orion_sdr::modulate::ft8::{FT8_DATA_SYMS, FT8_TONES};
use orion_sdr::codec::ft8::{Ft8Codec, Ft8Bits};
use orion_sdr::message::{CallsignHashTable, GridField, Ft8Message, pack77, unpack77};
use orion_sdr::sync::ft8_sync;
use super::helpers::make_ft8_test_buffer;

#[test]
fn roundtrip_ft8_noiseless() {
    let mut tones = [0u8; FT8_DATA_SYMS];
    for (i, t) in tones.iter_mut().enumerate() {
        *t = (i % FT8_TONES) as u8;
    }
    let frame_in = Ft8Frame::new(tones);

    let tx = Ft8Mod::new(12_000.0, 1_000.0, 0.0, 1.0);
    let iq = tx.modulate(&frame_in);

    let rx = Ft8Demod::new(12_000.0, 1_000.0);
    let frame_out = rx.demodulate(&iq).expect("FT8 demodulate returned None");

    assert_eq!(frame_in, frame_out, "FT8 noiseless roundtrip failed");
}

#[test]
fn roundtrip_ft8_all_tones() {
    let frame_in = Ft8Frame::new([7u8; FT8_DATA_SYMS]);
    let tx = Ft8Mod::new(12_000.0, 100.0, 0.0, 1.0);
    let iq = tx.modulate(&frame_in);
    let rx = Ft8Demod::new(12_000.0, 100.0);
    let frame_out = rx.demodulate(&iq).expect("FT8 all-7 demodulate failed");
    assert_eq!(frame_in, frame_out, "FT8 all-tone-7 roundtrip failed");
}

#[test]
fn roundtrip_ft8_codec_noiseless() {
    let payload: Ft8Bits = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60];
    let frame = Ft8Codec::encode(&payload);

    let tx = Ft8Mod::new(12_000.0, 1_000.0, 0.0, 1.0);
    let iq = tx.modulate(&frame);

    let rx = Ft8Demod::new(12_000.0, 1_000.0);
    let frame_out = rx.demodulate(&iq).expect("FT8 demodulate returned None");

    let decoded = Ft8Codec::decode_hard(&frame_out)
        .expect("FT8 codec decode failed (noiseless)");
    assert_eq!(payload, decoded, "FT8 full-stack codec roundtrip failed");
}

#[test]
fn roundtrip_ft8_codec_zeros() {
    let payload: Ft8Bits = [0u8; 10];
    let frame = Ft8Codec::encode(&payload);
    let iq = Ft8Mod::new(12_000.0, 500.0, 0.0, 1.0).modulate(&frame);
    let frame_out = Ft8Demod::new(12_000.0, 500.0).demodulate(&iq).unwrap();
    let decoded = Ft8Codec::decode_hard(&frame_out).expect("FT8 codec decode failed (zeros)");
    assert_eq!(payload, decoded);
}

#[test]
fn sync_ft8_noiseless_aligned() {
    let base_hz = 1_000.0_f32;
    let (buf, payload) = make_ft8_test_buffer(0, base_hz, 0.0);

    let results = ft8_sync(
        &buf,
        12_000.0,
        base_hz - 6.25,
        base_hz + 50.0 + 6.25,
        0,
        0,
        5,
    );

    assert!(!results.is_empty(), "FT8 sync returned no candidates");
    let best = &results[0];
    let decoded = Ft8Codec::decode_soft(&best.llr)
        .expect("FT8 sync+decode_soft failed (noiseless, aligned)");
    assert_eq!(payload, decoded, "FT8 sync noiseless aligned: payload mismatch");
}

#[test]
fn sync_ft8_noiseless_time_offset() {
    use orion_sdr::modulate::ft8::FT8_SAMPLES_PER_SYM;

    let base_hz = 800.0_f32;
    let offset = 3 * FT8_SAMPLES_PER_SYM;
    let (buf, payload) = make_ft8_test_buffer(offset, base_hz, 0.0);

    let results = ft8_sync(
        &buf,
        12_000.0,
        base_hz - 6.25,
        base_hz + 50.0 + 6.25,
        0,
        5,
        5,
    );

    assert!(!results.is_empty(), "FT8 sync (time offset) returned no candidates");
    let best = &results[0];
    let decoded = Ft8Codec::decode_soft(&best.llr)
        .expect("FT8 sync+decode_soft failed (noiseless, time offset)");
    assert_eq!(payload, decoded, "FT8 sync noiseless time-offset: payload mismatch");
}

#[test]
fn sync_ft8_noisy_high_snr() {
    let base_hz = 1_200.0_f32;
    let noise_power = 0.005_f32;
    let (buf, payload) = make_ft8_test_buffer(0, base_hz, noise_power);

    let results = ft8_sync(
        &buf,
        12_000.0,
        base_hz - 6.25,
        base_hz + 50.0 + 6.25,
        0,
        0,
        5,
    );

    assert!(!results.is_empty(), "FT8 sync (high SNR) returned no candidates");
    let best = &results[0];
    let decoded = Ft8Codec::decode_soft(&best.llr)
        .expect("FT8 sync+decode_soft failed (high SNR)");
    assert_eq!(payload, decoded, "FT8 sync high-SNR: payload mismatch");
}

// -- SNR sensitivity regression tests -----------------------------------------
//
// Fixed-threshold CI tests: each asserts that the full pipeline
// (codec encode -> mod -> add_awgn -> sync -> decode_soft) succeeds at a
// specific SNR level.  SNR is expressed in the WSJT-X convention: signal
// power relative to noise in a 2500 Hz reference bandwidth.
//
//   noise_power = P_sig * fs / (2500 * 10^(SNR/10))
//               = 0.5 * 12000 / (2500 * 10^(SNR/10))
//               = 2.4 / 10^(SNR/10)

const SNR_FS: f32 = 12_000.0;
const REF_BW_HZ: f32 = 2_500.0;
const SIG_POWER: f32 = 0.5;

fn snr_to_noise_power(snr_db: f32) -> f32 {
    SIG_POWER * SNR_FS / (REF_BW_HZ * 10.0_f32.powf(snr_db / 10.0))
}

/// FT8 must decode at -15 dB SNR/2500 Hz.
#[test]
fn ft8_decodes_at_minus_15db_snr_2500hz() {
    let base_hz = 1_000.0_f32;
    let noise_power = snr_to_noise_power(-15.0);
    let (buf, payload) = make_ft8_test_buffer(0, base_hz, noise_power);

    let results = ft8_sync(
        &buf, SNR_FS,
        base_hz - 6.25, base_hz + 50.0 + 6.25,
        0, 0, 5,
    );

    assert!(
        !results.is_empty(),
        "FT8 sync returned no candidates at -15 dB SNR/2500 Hz (noise_power={:.5})",
        noise_power
    );
    let decoded = Ft8Codec::decode_soft(&results[0].llr);
    assert!(
        decoded == Some(payload),
        "FT8 decode_soft failed at -15 dB SNR/2500 Hz (noise_power={:.5}, got={:?})",
        noise_power, decoded
    );
}

// -- Full-stack message roundtrip tests ---------------------------------------

#[test]
fn full_stack_ft8_type1() {
    let mut ht = CallsignHashTable::new();
    let msg = Ft8Message::Standard {
        call_to: "KD9ABC".to_string(),
        call_de: "W9XYZ".to_string(),
        extra: GridField::Grid("FN31".to_string()),
    };

    let payload = pack77(&msg, &mut ht).expect("pack77");
    let frame = Ft8Codec::encode(&payload);
    let iq = Ft8Mod::new(12_000.0, 1_000.0, 0.0, 1.0).modulate(&frame);
    let frame_out = Ft8Demod::new(12_000.0, 1_000.0)
        .demodulate(&iq)
        .expect("Ft8Demod failed");
    let bits_out = Ft8Codec::decode_hard(&frame_out).expect("decode_hard failed");
    let decoded = unpack77(&bits_out, &ht);
    assert_eq!(decoded, msg, "Full-stack FT8 Type1 roundtrip mismatch");
}

#[test]
fn full_stack_ft8_free_text() {
    let mut ht = CallsignHashTable::new();
    let msg = Ft8Message::FreeText("CQ DX".to_string());

    let payload = pack77(&msg, &mut ht).expect("pack77 free text");
    let frame = Ft8Codec::encode(&payload);
    let iq = Ft8Mod::new(12_000.0, 1_000.0, 0.0, 1.0).modulate(&frame);
    let frame_out = Ft8Demod::new(12_000.0, 1_000.0)
        .demodulate(&iq)
        .expect("Ft8Demod failed");
    let bits_out = Ft8Codec::decode_hard(&frame_out).expect("decode_hard failed");
    let decoded = unpack77(&bits_out, &ht);
    assert_eq!(decoded, msg, "Full-stack FT8 free text roundtrip mismatch");
}
