// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/multicarrier/cyclic_prefix.rs
use crate::core::{Block, WorkReport};
use num_complex::Complex32 as C32;

/// Prepends a cyclic prefix to each OFDM symbol: `C32 → C32`, `n_fft` samples
/// in, `n_fft + cp_len` samples out (the last `cp_len` samples of the symbol
/// copied to the front).
///
/// One `n_fft`-sized symbol per `process()` call; a partial trailing chunk is
/// a no-op.
#[derive(Debug, Clone)]
pub struct CyclicPrefixInsert {
    n_fft: usize,
    cp_len: usize,
}

impl CyclicPrefixInsert {
    pub fn new(n_fft: usize, cp_len: usize) -> Self {
        Self { n_fft, cp_len }
    }

    pub fn symbol_len(&self) -> usize {
        self.n_fft + self.cp_len
    }
}

impl Block for CyclicPrefixInsert {
    type In = C32;
    type Out = C32;

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> WorkReport {
        let n = self.n_fft;
        let cp = self.cp_len;
        let out_len = n + cp;
        if input.len() < n || output.len() < out_len {
            return WorkReport::default();
        }
        if cp > 0 {
            output[..cp].copy_from_slice(&input[n - cp..n]);
        }
        output[cp..out_len].copy_from_slice(&input[..n]);
        WorkReport {
            in_read: n,
            out_written: out_len,
        }
    }
}

/// Removes the cyclic prefix from each OFDM symbol: `C32 → C32`,
/// `n_fft + cp_len` samples in, `n_fft` samples out.
///
/// One `(n_fft + cp_len)`-sized symbol per `process()` call; a partial
/// trailing chunk is a no-op.
#[derive(Debug, Clone)]
pub struct CyclicPrefixRemove {
    n_fft: usize,
    cp_len: usize,
}

impl CyclicPrefixRemove {
    pub fn new(n_fft: usize, cp_len: usize) -> Self {
        Self { n_fft, cp_len }
    }

    pub fn symbol_len(&self) -> usize {
        self.n_fft + self.cp_len
    }
}

impl Block for CyclicPrefixRemove {
    type In = C32;
    type Out = C32;

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> WorkReport {
        let n = self.n_fft;
        let cp = self.cp_len;
        let in_len = n + cp;
        if input.len() < in_len || output.len() < n {
            return WorkReport::default();
        }
        output[..n].copy_from_slice(&input[cp..in_len]);
        WorkReport {
            in_read: in_len,
            out_written: n,
        }
    }
}
