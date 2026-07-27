// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/fec/interleaver.rs
//
// A rectangular block interleaver: symbols are written into an R×C matrix
// row-by-row and read out column-by-column (the deinterleaver does the
// inverse). This spreads a burst of adjacent coded symbols — which a deep
// frequency-selective null zeroes out on a contiguous run of subcarriers —
// across the FEC codeword, so no single fade exceeds the code's correcting
// span.
//
// The permutation is generic over `T: Copy` because the two deinterleavers in
// a concatenated chain operate in different domains: the *inner* deinterleaver
// permutes `f32` LLRs (before the soft inner decoder), while the *outer*
// deinterleaver permutes `u8` bytes/symbols (after inner decode, before the
// algebraic outer decoder). One index-permutation implementation serves both.
//
// When the data length is not a multiple of `rows * cols`, the final partial
// row is zero-conceptually-padded: `interleave` and `deinterleave` operate on
// exactly `rows * cols` elements, and the caller is responsible for padding to
// (and trimming from) that block size — the true element count is tracked by
// the frame layer.

/// Rectangular row-in / column-out block interleaver over `rows × cols`
/// elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockInterleaver {
    rows: usize,
    cols: usize,
}

impl BlockInterleaver {
    /// Creates an interleaver over a `rows × cols` block. Both dimensions must
    /// be nonzero.
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(
            rows > 0 && cols > 0,
            "interleaver dimensions must be nonzero"
        );
        Self { rows, cols }
    }

    /// Number of elements per interleaver block (`rows * cols`).
    #[inline]
    pub fn block_len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Interleaves one block: reads `input` (filled row-major) and writes it
    /// out column-major into `output`. Both slices must be exactly
    /// [`block_len`](Self::block_len) long.
    ///
    /// Element `input[r*cols + c]` (row `r`, column `c`) is placed at
    /// `output[c*rows + r]`.
    pub fn interleave<T: Copy>(&self, input: &[T], output: &mut [T]) {
        let n = self.block_len();
        assert_eq!(input.len(), n, "interleave input must be one full block");
        assert_eq!(output.len(), n, "interleave output must be one full block");
        for r in 0..self.rows {
            for c in 0..self.cols {
                output[c * self.rows + r] = input[r * self.cols + c];
            }
        }
    }

    /// Inverse of [`interleave`](Self::interleave): reads `input` (column-major)
    /// and restores row-major order into `output`. Both slices must be exactly
    /// [`block_len`](Self::block_len) long.
    pub fn deinterleave<T: Copy>(&self, input: &[T], output: &mut [T]) {
        let n = self.block_len();
        assert_eq!(input.len(), n, "deinterleave input must be one full block");
        assert_eq!(
            output.len(),
            n,
            "deinterleave output must be one full block"
        );
        for r in 0..self.rows {
            for c in 0..self.cols {
                output[r * self.cols + c] = input[c * self.rows + r];
            }
        }
    }
}
