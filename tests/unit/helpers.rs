// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Tiny single-bin DFT; good enough for power at a specific frequency.
pub fn dft_power(signal: &[f32], fs: f32, f_hz: f32) -> f32 {
    let n = signal.len();
    let w = -2.0 * std::f32::consts::PI * f_hz / fs;
    let mut re = 0.0f32;
    let mut im = 0.0f32;
    for (k, &x) in signal.iter().enumerate() {
        let t = w * (k as f32);
        re += x * t.cos();
        im += x * t.sin();
    }
    (re * re + im * im) / (n as f32 * n as f32)
}
