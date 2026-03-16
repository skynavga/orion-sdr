
use crate::modulate::{Ft8Mod, Ft8Frame};
use crate::modulate::ft8::{FT8_DATA_SYMS, FT8_FRAME_LEN};
use crate::demodulate::Ft8Demod;
use crate::codec::ft8::{Ft8Codec, Ft8Bits};
use std::hint::black_box;
use super::{minsps_from_env, measure_throughput};

const FS: f32 = 12_000.0;
const BASE_HZ: f32 = 1_000.0;
const REPEATS: usize = 20;

fn make_frame() -> Ft8Frame {
    let tones: [u8; FT8_DATA_SYMS] = core::array::from_fn(|i| (i % 8) as u8);
    Ft8Frame::new(tones)
}

fn make_payload() -> Ft8Bits {
    [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60]
}

#[test]
fn throughput_ft8_mod() {
    let frame = make_frame();
    let tx = Ft8Mod::new(FS, BASE_HZ, 0.0, 1.0);

    let (msps, dt) = measure_throughput(
        || {
            let iq = tx.modulate(&frame);
            black_box(iq[0].re);
            FT8_FRAME_LEN
        },
        FT8_FRAME_LEN,
        REPEATS,
    );

    println!("[FT8 mod] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.5);
    assert!(msps >= min_msps, "FT8 mod throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}

#[test]
fn throughput_ft8_demod() {
    let frame = make_frame();
    let tx = Ft8Mod::new(FS, BASE_HZ, 0.0, 1.0);
    let iq = tx.modulate(&frame);
    let rx = Ft8Demod::new(FS, BASE_HZ);

    let (msps, dt) = measure_throughput(
        || {
            let out = rx.demodulate(&iq).expect("demod failed");
            black_box(out.0[0]);
            FT8_FRAME_LEN
        },
        FT8_FRAME_LEN,
        REPEATS,
    );

    println!("[FT8 demod] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.5);
    assert!(msps >= min_msps, "FT8 demod throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}

#[test]
fn throughput_ft8_codec_encode() {
    let payload = make_payload();

    let (msps, dt) = measure_throughput(
        || {
            let frame = Ft8Codec::encode(&payload);
            black_box(frame.0[0]);
            FT8_FRAME_LEN
        },
        FT8_FRAME_LEN,
        REPEATS,
    );

    println!("[FT8 codec encode] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.01);
    assert!(msps >= min_msps, "FT8 codec encode throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}

#[test]
fn throughput_ft8_codec_decode() {
    let payload = make_payload();
    let frame = Ft8Codec::encode(&payload);

    let (msps, dt) = measure_throughput(
        || {
            let out = Ft8Codec::decode_hard(&frame).expect("decode failed");
            black_box(out[0]);
            FT8_FRAME_LEN
        },
        FT8_FRAME_LEN,
        REPEATS,
    );

    println!("[FT8 codec decode] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.01);
    assert!(msps >= min_msps, "FT8 codec decode throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}

#[test]
fn throughput_ft8_roundtrip() {
    let payload = make_payload();
    let tx = Ft8Mod::new(FS, BASE_HZ, 0.0, 1.0);
    let rx = Ft8Demod::new(FS, BASE_HZ);

    let (msps, dt) = measure_throughput(
        || {
            let frame = Ft8Codec::encode(&payload);
            let iq = tx.modulate(&frame);
            let frame_out = rx.demodulate(&iq).expect("demod failed");
            let out = Ft8Codec::decode_hard(&frame_out).expect("decode failed");
            black_box(out[0]);
            FT8_FRAME_LEN
        },
        FT8_FRAME_LEN,
        REPEATS,
    );

    println!("[FT8 roundtrip] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.5);
    assert!(msps >= min_msps, "FT8 roundtrip throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}
