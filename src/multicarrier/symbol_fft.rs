// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/multicarrier/symbol_fft.rs
use super::cyclic_prefix::CyclicPrefixRemove;
use super::fft::FftBlock;
use crate::core::Block;
use num_complex::Complex32 as C32;

/// RX front end for one OFDM symbol: [`CyclicPrefixRemove`] → [`FftBlock`],
/// producing the frequency-domain symbol (natural rustfft bin order).
///
/// This is the single shared CP-remove + FFT kernel every RX path runs before
/// it diverges (grid extraction, equalization, pilot installation, or `|X|²`
/// accumulation). Consolidating it here gives one place that owns the
/// `time`/`freq` scratch and — crucially — one place where the FFT-window
/// position is chosen, so a future window back-off is threaded in a single
/// spot rather than replicated across every demodulator.
///
/// One `symbol_len()`-sample input per [`demod_symbol`](Self::demod_symbol)
/// call; a shorter input is a no-op returning `None`, matching the underlying
/// blocks' partial-chunk contract.
#[derive(Clone)]
pub struct SymbolFft {
    n_fft: usize,
    cp_len: usize,
    cp_remove: CyclicPrefixRemove,
    fft: FftBlock,
    time: Vec<C32>,
    freq: Vec<C32>,
}

impl SymbolFft {
    pub fn new(n_fft: usize, cp_len: usize) -> Self {
        Self {
            n_fft,
            cp_len,
            cp_remove: CyclicPrefixRemove::new(n_fft, cp_len),
            fft: FftBlock::new(n_fft),
            time: vec![C32::default(); n_fft],
            freq: vec![C32::default(); n_fft],
        }
    }

    pub fn n_fft(&self) -> usize {
        self.n_fft
    }

    /// Samples consumed per symbol (`n_fft + cp_len`).
    pub fn symbol_len(&self) -> usize {
        self.n_fft + self.cp_len
    }

    /// CP-removes then FFTs one symbol from the front of `input`, returning the
    /// `n_fft`-bin frequency-domain symbol on success, or `None` if `input` is
    /// shorter than one symbol. The returned slice borrows internal scratch and
    /// is valid until the next call.
    pub fn demod_symbol(&mut self, input: &[C32]) -> Option<&[C32]> {
        let n = self.n_fft;
        if self.cp_remove.process(input, &mut self.time).out_written != n {
            return None;
        }
        if self.fft.process(&self.time, &mut self.freq).out_written != n {
            return None;
        }
        Some(&self.freq)
    }
}
