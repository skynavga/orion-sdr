// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/multicarrier/symbol_window.rs
use crate::core::{Block, WorkReport};
use num_complex::Complex32 as C32;

/// TX-side raised-cosine (Tukey) edge taper for one assembled OFDM symbol:
/// `C32 → C32`, `symbol_len` samples in and out, no length change.
///
/// Applies a cosine ramp-up over the first `roll_off` samples and a matching
/// ramp-down over the last `roll_off` samples of each `symbol_len`-sample
/// symbol, leaving the interior flat. This softens the symbol-boundary
/// discontinuity that gives plain OFDM its slowly-decaying (`~1/f`) spectral
/// skirt.
///
/// **RX transparency.** The taper is confined to the symbol's outer guard
/// samples — those a receiver discards. A receiver reading the window
/// `symbol[cp_len - b .. cp_len - b + n_fft]` (see
/// [`SymbolFft`](super::SymbolFft) with back-off `b`) never touches the first
/// `cp_len - b` or the last `b` samples, so a symmetric taper of
/// `roll_off ≤ min(cp_len - b, b)` samples per side (maximized at `b = cp_len/2`
/// ⇒ `cp_len/2`) leaves the receiver's `n_fft`-sample core bit-identical.
/// **The caller is responsible for pairing `roll_off` with a compatible RX
/// back-off**; this block only applies the taper.
///
/// Same-length and stateless: symbols abut (they are not overlap-added), so
/// each symbol is windowed independently with no cross-call buffering and the
/// downstream stream length and symbol boundaries are unchanged. `roll_off == 0`
/// is an identity pass-through.
///
/// One `symbol_len`-sample symbol per [`process`](Block::process) call; a
/// partial trailing chunk is a no-op.
#[derive(Debug, Clone)]
pub struct SymbolWindow {
    symbol_len: usize,
    roll_off: usize,
    /// Ramp-up coefficients, length `roll_off`, rising 0→1. The ramp-down is
    /// this reversed, so only one table is stored.
    ramp: Vec<f32>,
}

impl SymbolWindow {
    /// Builds a windower for `symbol_len`-sample symbols with a `roll_off`-sample
    /// cosine ramp at each edge. `roll_off` is clamped so the two ramps cannot
    /// overlap (`2 * roll_off <= symbol_len`); `roll_off == 0` is an identity.
    pub fn new(symbol_len: usize, roll_off: usize) -> Self {
        let roll_off = roll_off.min(symbol_len / 2);
        // Half-cosine rising edge: w[i] = 0.5 (1 - cos(pi (i + 0.5) / L)), which
        // is 0 at the symbol edge and 1 at the flat interior, sampled at bin
        // centers so the two edges are symmetric. Complementary across a seam:
        // ramp_up[i] and the abutting ramp_down value sum smoothly when the
        // underlying samples are cyclically equal (the CP guarantees this).
        let ramp = (0..roll_off)
            .map(|i| {
                let x = core::f32::consts::PI * (i as f32 + 0.5) / roll_off as f32;
                0.5 * (1.0 - x.cos())
            })
            .collect();
        Self {
            symbol_len,
            roll_off,
            ramp,
        }
    }

    pub fn symbol_len(&self) -> usize {
        self.symbol_len
    }

    pub fn roll_off(&self) -> usize {
        self.roll_off
    }
}

impl Block for SymbolWindow {
    type In = C32;
    type Out = C32;

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> WorkReport {
        let n = self.symbol_len;
        if input.len() < n || output.len() < n {
            return WorkReport::default();
        }
        output[..n].copy_from_slice(&input[..n]);
        let l = self.roll_off;
        for i in 0..l {
            let w = self.ramp[i];
            // Ramp up the leading edge and ramp down the trailing edge (the
            // trailing coefficient mirrors the leading one: ramp[l-1-i]).
            output[i] = scale(output[i], w);
            output[n - 1 - i] = scale(output[n - 1 - i], w);
        }
        WorkReport {
            in_read: n,
            out_written: n,
        }
    }
}

#[inline]
fn scale(c: C32, w: f32) -> C32 {
    C32::new(c.re * w, c.im * w)
}
