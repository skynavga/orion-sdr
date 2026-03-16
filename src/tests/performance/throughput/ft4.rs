
use crate::modulate::{Ft4Mod, Ft4Frame};
use crate::modulate::ft4::{FT4_DATA_SYMS, FT4_FRAME_LEN};
use crate::demodulate::Ft4Demod;
use crate::codec::ft4::{Ft4Codec, Ft4Bits};
use std::hint::black_box;
use super::{minsps_from_env, measure_throughput};

const FS: f32 = 12_000.0;
const BASE_HZ: f32 = 1_000.0;
const REPEATS: usize = 20;

fn make_frame() -> Ft4Frame {
    let tones: [u8; FT4_DATA_SYMS] = core::array::from_fn(|i| (i % 4) as u8);
    Ft4Frame::new(tones)
}

fn make_payload() -> Ft4Bits {
    [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60]
}

#[test]
fn throughput_ft4_mod() {
    let frame = make_frame();
    let tx = Ft4Mod::new(FS, BASE_HZ, 0.0, 1.0);

    let (msps, dt) = measure_throughput(
        || {
            let iq = tx.modulate(&frame);
            black_box(iq[0].re);
            FT4_FRAME_LEN
        },
        FT4_FRAME_LEN,
        REPEATS,
    );

    println!("[FT4 mod] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.5);
    assert!(msps >= min_msps, "FT4 mod throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}

#[test]
fn throughput_ft4_demod() {
    let frame = make_frame();
    let tx = Ft4Mod::new(FS, BASE_HZ, 0.0, 1.0);
    let iq = tx.modulate(&frame);
    let rx = Ft4Demod::new(FS, BASE_HZ);

    let (msps, dt) = measure_throughput(
        || {
            let out = rx.demodulate(&iq).expect("demod failed");
            black_box(out.0[0]);
            FT4_FRAME_LEN
        },
        FT4_FRAME_LEN,
        REPEATS,
    );

    println!("[FT4 demod] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.5);
    assert!(msps >= min_msps, "FT4 demod throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}

#[test]
fn throughput_ft4_codec_encode() {
    let payload = make_payload();

    let (msps, dt) = measure_throughput(
        || {
            let frame = Ft4Codec::encode(&payload);
            black_box(frame.0[0]);
            FT4_FRAME_LEN
        },
        FT4_FRAME_LEN,
        REPEATS,
    );

    println!("[FT4 codec encode] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.01);
    assert!(msps >= min_msps, "FT4 codec encode throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}

#[test]
fn throughput_ft4_codec_decode() {
    let payload = make_payload();
    let frame = Ft4Codec::encode(&payload);

    let (msps, dt) = measure_throughput(
        || {
            let out = Ft4Codec::decode_hard(&frame).expect("decode failed");
            black_box(out[0]);
            FT4_FRAME_LEN
        },
        FT4_FRAME_LEN,
        REPEATS,
    );

    println!("[FT4 codec decode] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.01);
    assert!(msps >= min_msps, "FT4 codec decode throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}

#[test]
fn throughput_ft4_roundtrip() {
    let payload = make_payload();
    let tx = Ft4Mod::new(FS, BASE_HZ, 0.0, 1.0);
    let rx = Ft4Demod::new(FS, BASE_HZ);

    let (msps, dt) = measure_throughput(
        || {
            let frame = Ft4Codec::encode(&payload);
            let iq = tx.modulate(&frame);
            let frame_out = rx.demodulate(&iq).expect("demod failed");
            let out = Ft4Codec::decode_hard(&frame_out).expect("decode failed");
            black_box(out[0]);
            FT4_FRAME_LEN
        },
        FT4_FRAME_LEN,
        REPEATS,
    );

    println!("[FT4 roundtrip] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.5);
    assert!(msps >= min_msps, "FT4 roundtrip throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}
