// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/fec/scrambler.rs
//
// An additive (synchronous) PN scrambler / energy-dispersal whitener. A
// linear-feedback shift register generates a pseudo-random bit sequence that
// is XORed with the data, one bit per data bit. Because the PN sequence is
// generated independently of the data, the descrambler is identical to the
// scrambler (XOR is self-inverse) as long as both start from the same seed —
// hence a single `scramble` method serves both directions.
//
// The LFSR polynomial, register width, and seed are all parameters, because
// they differ by standard: 802.11 uses a 7-bit register (x^7 + x^4 + 1) with a
// per-frame-random seed signaled in the header; DVB energy dispersal uses a
// 15-bit register (x^15 + x^14 + 1) with a fixed init. The `orion-sdr` default
// is a bespoke additive PN. (A self-synchronizing *multiplicative* variant,
// which feeds data back into the register and needs a distinct descrambler, is
// a later addition and not implemented here.)
//
// Convention: the register is a Fibonacci LFSR. `taps` is a bitmask over the
// register's `width` bits; the feedback bit is the XOR (parity) of the tapped
// register bits. The scrambling bit taken per step is the register's low bit
// (bit 0), and the register shifts right with the feedback bit inserted at the
// top (bit `width - 1`). Data bits are processed LSB-first within each byte,
// matching the natural bit order the frame layer packs bytes in.

/// A parameterized additive LFSR whitener (self-inverse: `scramble` both
/// scrambles and descrambles).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PnScrambler {
    /// Feedback tap mask over the low `width` bits.
    taps: u32,
    /// Register width in bits (2..=32).
    width: u32,
    /// Initial register state (must be nonzero, and fits in `width` bits).
    seed: u32,
}

impl PnScrambler {
    /// Creates a scrambler with feedback `taps` over a `width`-bit register,
    /// starting from `seed`.
    ///
    /// `width` must be in `2..=32`; `seed` must be nonzero (an all-zero
    /// register is a fixed point that never advances) and representable in
    /// `width` bits.
    pub fn new(taps: u32, width: u32, seed: u32) -> Self {
        assert!((2..=32).contains(&width), "LFSR width must be in 2..=32");
        let mask = Self::mask_for(width);
        assert!(seed != 0, "LFSR seed must be nonzero");
        assert!(seed & !mask == 0, "LFSR seed must fit in `width` bits");
        assert!(taps & !mask == 0, "LFSR taps must fit in `width` bits");
        Self { taps, width, seed }
    }

    #[inline]
    fn mask_for(width: u32) -> u32 {
        if width == 32 {
            u32::MAX
        } else {
            (1u32 << width) - 1
        }
    }

    /// XORs the PN sequence (restarted from `seed`) over `data` in place. Run
    /// it once to scramble and again — with the same parameters — to recover
    /// the original bytes.
    pub fn scramble(&self, data: &mut [u8]) {
        let mask = Self::mask_for(self.width);
        let top = self.width - 1;
        let mut reg = self.seed & mask;
        for byte in data.iter_mut() {
            let mut out = 0u8;
            for bit in 0..8 {
                // PN bit is the register's low bit.
                let pn = (reg & 1) as u8;
                out |= (((*byte >> bit) & 1) ^ pn) << bit;
                // Advance: feedback = parity of tapped bits, shift right,
                // insert feedback at the top bit.
                let fb = (reg & self.taps).count_ones() & 1;
                reg = (reg >> 1) | (fb << top);
                reg &= mask;
            }
            *byte = out;
        }
    }
}
