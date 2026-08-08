// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/multicarrier/symbol_fft.rs
use super::fft::FftBlock;
use crate::core::Block;
use num_complex::Complex32 as C32;

/// RX front end for one OFDM symbol: select the `n_fft`-sample FFT window
/// within the `n_fft + cp_len` symbol, then [`FftBlock`] it — producing the
/// frequency-domain symbol (natural rustfft bin order).
///
/// This is the single shared window-select + FFT kernel every RX path runs
/// before it diverges (grid extraction, equalization, pilot installation, or
/// `|X|²` accumulation). Consolidating it here gives one place that owns the
/// `time`/`freq` scratch and — crucially — the one place the FFT-window
/// position is chosen, so the window **back-off** is set here rather than
/// replicated across every demodulator.
///
/// **Window position.** With back-off `b = 0` (the default) the window is the
/// last `n_fft` samples of the symbol, `input[cp_len .. cp_len + n_fft]` — the
/// standard CP-removal position, bit-identical to a plain
/// [`CyclicPrefixRemove`](super::CyclicPrefixRemove). A back-off `b > 0` slides
/// the window `b` samples *earlier*, into the guard interval:
/// `input[cp_len - b .. cp_len - b + n_fft]`. This is standard receiver
/// practice for multipath/pre-echo robustness (guard on both sides of the
/// useful part) and is the enabler for RX-transparent TX symbol windowing (a
/// TX taper confined to the outer guard then falls outside the window). The
/// back-off is bounded by `cp_len` and clamped to it on construction.
///
/// Input consumption is always `symbol_len() = n_fft + cp_len` regardless of
/// back-off — only the *window within* that span moves — so symbol boundaries
/// and the strided RX cursor are unaffected. A shorter input is a no-op
/// returning `None`.
#[derive(Clone)]
pub struct SymbolFft {
    n_fft: usize,
    cp_len: usize,
    /// FFT-window back-off in samples (`0..=cp_len`): how far the window is
    /// pulled earlier from the CP boundary into the guard interval.
    backoff: usize,
    fft: FftBlock,
    time: Vec<C32>,
    freq: Vec<C32>,
}

impl SymbolFft {
    pub fn new(n_fft: usize, cp_len: usize) -> Self {
        Self {
            n_fft,
            cp_len,
            backoff: 0,
            fft: FftBlock::new(n_fft),
            time: vec![C32::default(); n_fft],
            freq: vec![C32::default(); n_fft],
        }
    }

    /// Sets the FFT-window back-off (samples pulled earlier into the guard).
    /// Clamped to `cp_len` — a back-off cannot exceed the guard, since the
    /// window must still start at or after the symbol's first sample. `0`
    /// reproduces the standard CP-boundary window exactly.
    pub fn with_window_backoff(mut self, backoff: usize) -> Self {
        self.backoff = backoff.min(self.cp_len);
        self
    }

    pub fn n_fft(&self) -> usize {
        self.n_fft
    }

    /// The effective (clamped) window back-off in samples.
    pub fn window_backoff(&self) -> usize {
        self.backoff
    }

    /// Samples consumed per symbol (`n_fft + cp_len`).
    pub fn symbol_len(&self) -> usize {
        self.n_fft + self.cp_len
    }

    /// Selects the FFT window and FFTs it, returning the `n_fft`-bin
    /// frequency-domain symbol on success, or `None` if `input` is shorter than
    /// one symbol. The returned slice borrows internal scratch and is valid
    /// until the next call.
    pub fn demod_symbol(&mut self, input: &[C32]) -> Option<&[C32]> {
        let n = self.n_fft;
        if input.len() < self.symbol_len() {
            return None;
        }
        // Window start: cp_len at back-off 0, sliding earlier into the guard as
        // back-off grows. `backoff <= cp_len` is guaranteed by the setter, so
        // `start >= 0` and `start + n <= cp_len + n = symbol_len`.
        let start = self.cp_len - self.backoff;
        self.time.copy_from_slice(&input[start..start + n]);
        if self.fft.process(&self.time, &mut self.freq).out_written != n {
            return None;
        }
        Some(&self.freq)
    }
}
