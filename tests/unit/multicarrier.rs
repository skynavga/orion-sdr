// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use num_complex::Complex32 as C32;
use orion_sdr::core::Block;
use orion_sdr::multicarrier::{CyclicPrefixInsert, CyclicPrefixRemove, FftBlock, IfftBlock};

fn tone(n: usize, cycles: f32) -> Vec<C32> {
    (0..n)
        .map(|k| {
            let phase = std::f32::consts::TAU * cycles * (k as f32) / (n as f32);
            C32::new(phase.cos(), phase.sin())
        })
        .collect()
}

#[test]
fn fft_ifft_roundtrip_identity() {
    let n_fft = 64;
    let input = tone(n_fft, 3.0);
    let mut freq = vec![C32::default(); n_fft];
    let mut time = vec![C32::default(); n_fft];

    let mut fft = FftBlock::new(n_fft);
    let mut ifft = IfftBlock::new(n_fft);

    fft.process(&input, &mut freq);
    ifft.process(&freq, &mut time);

    let eps = 1e-4f32;
    for (a, b) in input.iter().zip(time.iter()) {
        assert!(
            (a - b).norm() < eps,
            "roundtrip mismatch: {:?} vs {:?}",
            a,
            b
        );
    }
}

#[test]
fn fft_dc_bin_impulse() {
    // A constant (DC) input should produce all its energy in bin 0 under a
    // unity-gain forward FFT.
    let n_fft = 32;
    let input = vec![C32::new(1.0, 0.0); n_fft];
    let mut freq = vec![C32::default(); n_fft];

    FftBlock::new(n_fft).process(&input, &mut freq);

    let eps = 1e-3f32;
    assert!(
        (freq[0].re - n_fft as f32).abs() < eps,
        "DC bin expected {}, got {:?}",
        n_fft,
        freq[0]
    );
    for bin in &freq[1..] {
        assert!(bin.norm() < eps, "non-DC bin not silent: {:?}", bin);
    }
}

#[test]
fn fft_partial_chunk_is_noop() {
    let n_fft = 64;
    let input = tone(n_fft - 1, 1.0); // one sample short
    let mut output = vec![C32::default(); n_fft];

    let wr = FftBlock::new(n_fft).process(&input, &mut output);
    assert_eq!(wr.in_read, 0);
    assert_eq!(wr.out_written, 0);
}

#[test]
fn fft_multi_symbol_chunk() {
    // process() only consumes one n_fft-sized symbol per call, even if more
    // input/output is available.
    let n_fft = 16;
    let input = tone(n_fft * 3, 2.0);
    let mut output = vec![C32::default(); n_fft * 3];

    let wr = FftBlock::new(n_fft).process(&input, &mut output);
    assert_eq!(wr.in_read, n_fft);
    assert_eq!(wr.out_written, n_fft);
}

#[test]
fn cyclic_prefix_insert_content() {
    let n_fft = 8;
    let cp_len = 3;
    let input: Vec<C32> = (0..n_fft).map(|k| C32::new(k as f32, 0.0)).collect();
    let mut output = vec![C32::default(); n_fft + cp_len];

    let wr = CyclicPrefixInsert::new(n_fft, cp_len).process(&input, &mut output);
    assert_eq!(wr.in_read, n_fft);
    assert_eq!(wr.out_written, n_fft + cp_len);

    // CP is the last cp_len samples of the symbol, copied to the front.
    assert_eq!(output[..cp_len], input[n_fft - cp_len..]);
    assert_eq!(output[cp_len..], input[..]);
}

#[test]
fn cyclic_prefix_roundtrip() {
    let n_fft = 16;
    let cp_len = 4;
    let input: Vec<C32> = (0..n_fft)
        .map(|k| C32::new(k as f32 * 0.5, -(k as f32)))
        .collect();
    let mut with_cp = vec![C32::default(); n_fft + cp_len];
    let mut restored = vec![C32::default(); n_fft];

    CyclicPrefixInsert::new(n_fft, cp_len).process(&input, &mut with_cp);
    CyclicPrefixRemove::new(n_fft, cp_len).process(&with_cp, &mut restored);

    // Exact match expected — no float error since this is a pure copy.
    assert_eq!(restored, input);
}

#[test]
fn cyclic_prefix_zero_length_cp() {
    let n_fft = 8;
    let cp_len = 0;
    let input: Vec<C32> = (0..n_fft).map(|k| C32::new(k as f32, 0.0)).collect();
    let mut output = vec![C32::default(); n_fft];

    let wr = CyclicPrefixInsert::new(n_fft, cp_len).process(&input, &mut output);
    assert_eq!(wr.in_read, n_fft);
    assert_eq!(wr.out_written, n_fft);
    assert_eq!(output, input);
}
