// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared test helpers used by multiple test crates.

use num_complex::Complex32 as C32;

#[allow(dead_code)] // used by snr tests, not all test binaries
pub fn snr_db_at(fs: f32, f0: f32, x: &[f32]) -> f32 {
    let n = x.len().max(1);
    let proj = |f: f32| {
        let w = -std::f32::consts::TAU * f / fs;
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (k, &s) in x.iter().enumerate() {
            let t = w * (k as f32);
            re += s * t.cos();
            im += s * t.sin();
        }
        (re * re + im * im) / ((n as f32) * (n as f32))
    };
    let p_sig = proj(f0);
    let p_off = proj(f0 * 0.73) + 1e-20;
    10.0 * (p_sig / p_off).log10()
}

#[allow(dead_code)] // used by roundtrip and performance, not unit
pub fn add_awgn(iq: &mut [C32], noise_power: f32, seed: u64) {
    let mut state = seed ^ 0xDEAD_BEEF_CAFE_0000;
    let scale = (noise_power / 2.0).sqrt();

    let next_f32 = |s: &mut u64| -> f32 {
        let mut sum = 0.0f32;
        for _ in 0..12 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            sum += (*s as f32) / (u64::MAX as f32) - 0.5;
        }
        sum
    };

    for sample in iq.iter_mut() {
        let n_i = next_f32(&mut state) * scale;
        let n_q = next_f32(&mut state) * scale;
        sample.re += n_i;
        sample.im += n_q;
    }
}
