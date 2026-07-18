// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/multicarrier/fft.rs
use crate::core::{Block, WorkReport};
use num_complex::Complex32 as C32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Forward FFT, unity gain: `C32 → C32`, `n_fft` samples in, `n_fft` bins out.
///
/// One `n_fft`-sized symbol per `process()` call; a partial trailing chunk
/// (fewer than `n_fft` samples) is a no-op (`WorkReport::default()`), with no
/// cross-call buffering.
#[derive(Clone)]
pub struct FftBlock {
    n_fft: usize,
    plan: Arc<dyn Fft<f32>>,
    scratch: Vec<C32>,
}

impl FftBlock {
    pub fn new(n_fft: usize) -> Self {
        let plan = FftPlanner::new().plan_fft_forward(n_fft);
        let scratch = vec![C32::default(); plan.get_inplace_scratch_len()];
        Self {
            n_fft,
            plan,
            scratch,
        }
    }

    pub fn n_fft(&self) -> usize {
        self.n_fft
    }
}

impl Block for FftBlock {
    type In = C32;
    type Out = C32;

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> WorkReport {
        let n = self.n_fft;
        if input.len() < n || output.len() < n {
            return WorkReport::default();
        }
        copy_c32(&input[..n], &mut output[..n]);
        self.plan
            .process_with_scratch(&mut output[..n], &mut self.scratch);
        WorkReport {
            in_read: n,
            out_written: n,
        }
    }
}

/// Inverse FFT with `1/N` scale folded into the output copy: `C32 → C32`,
/// `n_fft` bins in, `n_fft` samples out.
///
/// Same whole-symbol-per-call contract as `FftBlock`: a partial trailing
/// chunk is a no-op.
#[derive(Clone)]
pub struct IfftBlock {
    n_fft: usize,
    scale: f32,
    plan: Arc<dyn Fft<f32>>,
    scratch: Vec<C32>,
    time: Vec<C32>,
}

impl IfftBlock {
    pub fn new(n_fft: usize) -> Self {
        let plan = FftPlanner::new().plan_fft_inverse(n_fft);
        let scratch = vec![C32::default(); plan.get_inplace_scratch_len()];
        Self {
            n_fft,
            scale: 1.0 / n_fft as f32,
            plan,
            scratch,
            time: vec![C32::default(); n_fft],
        }
    }

    pub fn n_fft(&self) -> usize {
        self.n_fft
    }
}

impl Block for IfftBlock {
    type In = C32;
    type Out = C32;

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> WorkReport {
        let n = self.n_fft;
        if input.len() < n || output.len() < n {
            return WorkReport::default();
        }
        copy_c32(&input[..n], &mut self.time[..n]);
        self.plan
            .process_with_scratch(&mut self.time[..n], &mut self.scratch);
        let g = self.scale;
        let mut i = 0;
        let nn = n & !3;
        while i < nn {
            output[i] = self.time[i] * g;
            output[i + 1] = self.time[i + 1] * g;
            output[i + 2] = self.time[i + 2] * g;
            output[i + 3] = self.time[i + 3] * g;
            i += 4;
        }
        while i < n {
            output[i] = self.time[i] * g;
            i += 1;
        }
        WorkReport {
            in_read: n,
            out_written: n,
        }
    }
}

#[inline(always)]
fn copy_c32(input: &[C32], output: &mut [C32]) {
    let n = input.len();
    let mut i = 0;
    let nn = n & !3;
    while i < nn {
        output[i] = input[i];
        output[i + 1] = input[i + 1];
        output[i + 2] = input[i + 2];
        output[i + 3] = input[i + 3];
        i += 4;
    }
    while i < n {
        output[i] = input[i];
        i += 1;
    }
}
