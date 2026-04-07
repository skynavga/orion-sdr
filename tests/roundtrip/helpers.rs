// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use num_complex::Complex32 as C32;
use crate::common::add_awgn;

pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() { return 0.0; }
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    (sum_sq / x.len() as f32).sqrt()
}

pub fn tail<T>(x: &[T]) -> &[T] {
    &x[x.len() / 4..]
}

pub fn real_tone(fs: f32, f_hz: f32, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|k| amp * (std::f32::consts::TAU * f_hz * (k as f32) / fs).sin())
        .collect()
}

pub fn make_ft8_test_buffer(
    time_offset_samples: usize,
    base_hz: f32,
    noise_power: f32,
) -> (Vec<C32>, orion_sdr::codec::ft8::Ft8Bits) {
    use orion_sdr::codec::ft8::{Ft8Codec, Ft8Bits};
    use orion_sdr::modulate::Ft8Mod;
    use orion_sdr::modulate::ft8::FT8_FRAME_LEN;

    let payload: Ft8Bits = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60];
    let frame = Ft8Codec::encode(&payload);
    let iq_frame = Ft8Mod::new(12_000.0, base_hz, 0.0, 1.0).modulate(&frame);

    let total_len = time_offset_samples + FT8_FRAME_LEN + FT8_FRAME_LEN / 4;
    let mut buf = vec![C32::new(0.0, 0.0); total_len];

    for (i, s) in iq_frame.iter().enumerate() {
        buf[time_offset_samples + i] = *s;
    }

    if noise_power > 0.0 {
        add_awgn(&mut buf, noise_power, 0x1234_5678_ABCD_EF00);
    }

    (buf, payload)
}

pub fn make_ft4_test_buffer(
    time_offset_samples: usize,
    base_hz: f32,
    noise_power: f32,
) -> (Vec<C32>, orion_sdr::codec::ft4::Ft4Bits) {
    use orion_sdr::codec::ft4::{Ft4Codec, Ft4Bits};
    use orion_sdr::modulate::Ft4Mod;
    use orion_sdr::modulate::ft4::FT4_FRAME_LEN;

    let payload: Ft4Bits = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60];
    let frame = Ft4Codec::encode(&payload);
    let iq_frame = Ft4Mod::new(12_000.0, base_hz, 0.0, 1.0).modulate(&frame);

    let total_len = time_offset_samples + FT4_FRAME_LEN + FT4_FRAME_LEN / 4;
    let mut buf = vec![C32::new(0.0, 0.0); total_len];

    for (i, s) in iq_frame.iter().enumerate() {
        buf[time_offset_samples + i] = *s;
    }

    if noise_power > 0.0 {
        add_awgn(&mut buf, noise_power, 0x9876_5432_FEDC_BA98);
    }

    (buf, payload)
}
