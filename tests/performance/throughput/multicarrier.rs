// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{measure_throughput, minsps_from_env};
use num_complex::Complex32 as C32;
use orion_sdr::core::Block;
use orion_sdr::multicarrier::{CyclicPrefixInsert, CyclicPrefixRemove, FftBlock, IfftBlock};
use std::hint::black_box;

fn throughput_fft_ifft_roundtrip(n_fft: usize, label: &str) {
    let symbols = 4096usize.div_ceil(n_fft).max(1) * 16;
    let repeats = 30;

    let input: Vec<C32> = (0..n_fft)
        .map(|k| {
            let phase = std::f32::consts::TAU * 3.0 * (k as f32) / (n_fft as f32);
            C32::new(phase.cos(), phase.sin())
        })
        .collect();
    let mut freq = vec![C32::default(); n_fft];
    let mut time = vec![C32::default(); n_fft];

    let mut fft = FftBlock::new(n_fft);
    let mut ifft = IfftBlock::new(n_fft);

    let (msps, dt) = measure_throughput(
        || {
            for _ in 0..symbols {
                fft.process(&input, &mut freq);
                ifft.process(&freq, &mut time);
                black_box(time[0]);
            }
            symbols * n_fft
        },
        symbols * n_fft,
        repeats,
    );

    println!("[{}] {:.2} Msps in {:.3}s", label, msps, dt);
    let min_msps = minsps_from_env(0.2);
    assert!(
        msps >= min_msps,
        "{} throughput {:.2} Msps < min {:.2} Msps",
        label,
        msps,
        min_msps
    );
}

#[test]
fn throughput_fft_ifft_roundtrip_n64() {
    throughput_fft_ifft_roundtrip(64, "FFT/IFFT n=64");
}

#[test]
fn throughput_fft_ifft_roundtrip_n1024() {
    throughput_fft_ifft_roundtrip(1024, "FFT/IFFT n=1024");
}

#[test]
fn throughput_fft_ifft_roundtrip_n4096() {
    throughput_fft_ifft_roundtrip(4096, "FFT/IFFT n=4096");
}

#[test]
fn throughput_cyclic_prefix_roundtrip() {
    let n_fft = 1024;
    let cp_len = 128;
    let symbols = 256;
    let repeats = 30;

    let input: Vec<C32> = (0..n_fft).map(|k| C32::new(k as f32, 0.0)).collect();
    let mut with_cp = vec![C32::default(); n_fft + cp_len];
    let mut restored = vec![C32::default(); n_fft];

    let mut insert = CyclicPrefixInsert::new(n_fft, cp_len);
    let mut remove = CyclicPrefixRemove::new(n_fft, cp_len);

    let (msps, dt) = measure_throughput(
        || {
            for _ in 0..symbols {
                insert.process(&input, &mut with_cp);
                remove.process(&with_cp, &mut restored);
                black_box(restored[0]);
            }
            symbols * n_fft
        },
        symbols * n_fft,
        repeats,
    );

    println!("[CyclicPrefix] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.2);
    assert!(
        msps >= min_msps,
        "CyclicPrefix throughput {:.2} Msps < min {:.2} Msps",
        msps,
        min_msps
    );
}
