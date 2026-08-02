// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/fec/interleaver.rs
//
// Two interleavers share this module:
//
//   • [`BlockInterleaver`] — a stateless rectangular row-in / column-out block
//     interleaver, generic over the element type.
//   • [`ConvInterleaver`] / [`ConvDeinterleaver`] — DVB-T's Forney convolutional
//     byte interleaver, as *stateful streaming* blocks.
//
// BLOCK INTERLEAVER. Symbols are written into an R×C matrix row-by-row and read
// out column-by-column (the deinterleaver does the inverse). This spreads a
// burst of adjacent coded symbols — which a deep frequency-selective null zeroes
// out on a contiguous run of subcarriers — across the FEC codeword, so no single
// fade exceeds the code's correcting span.
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
//
// CONVOLUTIONAL (FORNEY) INTERLEAVER — DUAL-MODE STREAMING. DVB-T's outer byte
// interleaver (ETSI EN 300 744 §4.3.1): `I` branches commutated round-robin,
// branch `j` a FIFO of `j·M` bytes (branch 0 undelayed). DVB-T uses `I = 12`,
// `M = 17`; note `204 = I·M`, so one RS(204,188) codeword fills all 12 branches
// exactly once.
//
// This is inherently a *continuous stream* device: a byte entering branch `j`
// re-emerges `j·M` commutator-cycles later, so [`ConvInterleaver::feed`] carries
// the branch FIFOs across calls (`feed` is 1:1 in length — the delay is internal
// state, not a length change). The interleaver+deinterleaver pair has a total
// end-to-end delay of `I·(I-1)·M` bytes; [`ConvInterleaver::flush`] emits exactly
// that many drain bytes so the deepest branch's real data exits, and the matched
// [`ConvDeinterleaver`] reproduces the input after its own `I·(I-1)·M`-byte
// startup. `reset` clears the FIFOs and commutator.
//
// DUAL MODE. A *stream* orchestrator (the DVB-T pipeline) feeds an unbounded byte
// stream and pays the flush once at end-of-stream — this is real DVB-T, zero
// per-frame overhead. A *frame* orchestrator gets a self-contained per-unit
// permutation via `reset()` then `feed(unit)` then `flush()`; the round-trip
// delay `I·(I-1)·M` is the price of framing a stream device, and the deinterleaver
// recovers the unit at output offset `I·(I-1)·M`. Both share the identical block.

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

use std::collections::VecDeque;

/// Total end-to-end delay of a Forney interleaver + deinterleaver pair, in
/// bytes: `branches · (branches − 1) · depth`. Also the number of drain bytes
/// [`ConvInterleaver::flush`]/[`ConvDeinterleaver::flush`] emit, and the output
/// offset at which the deinterleaver reproduces the interleaver's input.
#[inline]
pub const fn conv_roundtrip_delay(branches: usize, depth: usize) -> usize {
    branches * (branches - 1) * depth
}

/// Streaming Forney convolutional byte interleaver (see the module header).
/// Branch `j` carries a `j·depth`-byte FIFO; `feed` is length-preserving and
/// stateful. DVB-T's outer interleaver is `ConvInterleaver::dvb_t()`.
#[derive(Debug, Clone)]
pub struct ConvInterleaver {
    branches: usize,
    depth: usize,
    /// One FIFO per branch; branch `j` holds `j·depth` cells (branch 0 empty).
    fifos: Vec<VecDeque<u8>>,
    /// Commutator position: the next byte enters branch `pos % branches`.
    pos: usize,
}

impl ConvInterleaver {
    /// Creates a Forney interleaver with `branches` (`I`) branches, each a
    /// multiple-of-`depth` (`M`) delay line. Both must be nonzero. DVB-T's outer
    /// interleaver is [`ConvInterleaver::dvb_t`].
    pub fn new(branches: usize, depth: usize) -> Self {
        assert!(
            branches > 0 && depth > 0,
            "convolutional interleaver dimensions must be nonzero"
        );
        Self {
            branches,
            depth,
            fifos: (0..branches)
                .map(|j| VecDeque::from(vec![0u8; j * depth]))
                .collect(),
            pos: 0,
        }
    }

    /// The DVB-T outer interleaver: `I = 12` branches, `M = 17` cells.
    pub fn dvb_t() -> Self {
        Self::new(12, 17)
    }

    pub fn branches(&self) -> usize {
        self.branches
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Round-trip (interleave + deinterleave) delay in bytes,
    /// `branches·(branches−1)·depth` — the flush length and deinterleaver
    /// recovery offset.
    pub fn roundtrip_delay(&self) -> usize {
        conv_roundtrip_delay(self.branches, self.depth)
    }

    /// Clears the delay lines and commutator, so the next `feed` starts a fresh
    /// stream (frame-orchestrator use: `reset` before each unit).
    pub fn reset(&mut self) {
        for (j, fifo) in self.fifos.iter_mut().enumerate() {
            fifo.clear();
            fifo.extend(std::iter::repeat_n(0u8, j * self.depth));
        }
        self.pos = 0;
    }

    /// Interleaves a chunk of bytes, carrying delay-line state across calls.
    /// Output length equals input length (the delay is internal state).
    pub fn feed(&mut self, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(data.len());
        for &b in data {
            let j = self.pos % self.branches;
            if j == 0 {
                out.push(b);
            } else {
                self.fifos[j].push_back(b);
                out.push(self.fifos[j].pop_front().unwrap());
            }
            self.pos += 1;
        }
        out
    }

    /// Drains the delay lines by feeding [`roundtrip_delay`](Self::roundtrip_delay)
    /// zero bytes, so every buffered real byte exits. Returns the drain output.
    /// After a `feed`+`flush`, the matched [`ConvDeinterleaver`] reproduces the
    /// fed bytes at output offset `roundtrip_delay`.
    pub fn flush(&mut self) -> Vec<u8> {
        let d = self.roundtrip_delay();
        self.feed(&vec![0u8; d])
    }
}

/// Streaming Forney convolutional byte deinterleaver — the exact inverse of
/// [`ConvInterleaver`]. Branch `j` carries a `(branches−1−j)·depth`-byte FIFO
/// (the mirror of the interleaver's `j·depth`), so each byte's total delay
/// through the pair is the constant `(branches−1)·depth` commutator-cycles.
#[derive(Debug, Clone)]
pub struct ConvDeinterleaver {
    branches: usize,
    depth: usize,
    fifos: Vec<VecDeque<u8>>,
    pos: usize,
}

impl ConvDeinterleaver {
    /// Creates the matched deinterleaver for a [`ConvInterleaver::new`]`(branches,
    /// depth)`.
    pub fn new(branches: usize, depth: usize) -> Self {
        assert!(
            branches > 0 && depth > 0,
            "convolutional interleaver dimensions must be nonzero"
        );
        Self {
            branches,
            depth,
            fifos: (0..branches)
                .map(|j| VecDeque::from(vec![0u8; (branches - 1 - j) * depth]))
                .collect(),
            pos: 0,
        }
    }

    /// The matched deinterleaver for DVB-T's outer interleaver (`I=12`, `M=17`).
    pub fn dvb_t() -> Self {
        Self::new(12, 17)
    }

    pub fn branches(&self) -> usize {
        self.branches
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Round-trip delay / recovery offset in bytes (see [`ConvInterleaver`]).
    pub fn roundtrip_delay(&self) -> usize {
        conv_roundtrip_delay(self.branches, self.depth)
    }

    /// Clears the delay lines and commutator.
    pub fn reset(&mut self) {
        for (j, fifo) in self.fifos.iter_mut().enumerate() {
            fifo.clear();
            fifo.extend(std::iter::repeat_n(
                0u8,
                (self.branches - 1 - j) * self.depth,
            ));
        }
        self.pos = 0;
    }

    /// Deinterleaves a chunk, carrying state across calls (1:1 in length).
    pub fn feed(&mut self, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(data.len());
        for &b in data {
            let j = self.pos % self.branches;
            // The mirror branch: interleaver branch j has delay j·M, so the
            // deinterleaver's branch j uses (I-1-j)·M; branch I-1 is undelayed.
            if self.branches - 1 - j == 0 {
                out.push(b);
            } else {
                self.fifos[j].push_back(b);
                out.push(self.fifos[j].pop_front().unwrap());
            }
            self.pos += 1;
        }
        out
    }

    /// Drains the delay lines (feeds `roundtrip_delay` zeros).
    pub fn flush(&mut self) -> Vec<u8> {
        let d = self.roundtrip_delay();
        self.feed(&vec![0u8; d])
    }
}
