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
//
// TWO SHAPES. [`PnScrambler`] restarts from `seed` on every `scramble` call —
// the frame-orchestrator shape, where each frame whitens independently.
// [`PnScramblerStream`] carries the register across `feed` calls — the
// stream-orchestrator shape (continuous DVB energy dispersal), with `reset` to
// restart from `seed`. Both share the identical per-byte register advance, so
// one `feed(&whole)` equals `scramble(&whole)` and a `reset` + chunked `feed`s
// equal one `scramble` over the concatenation.

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
            *byte = scramble_byte(*byte, &mut reg, self.taps, top, mask);
        }
    }

    /// The scrambler's parameters, for constructing a matching
    /// [`PnScramblerStream`] or inspecting the LFSR.
    pub fn taps(&self) -> u32 {
        self.taps
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// A [`PnScramblerStream`] with the same taps/width/seed — the streaming
    /// counterpart of this stateless scrambler.
    pub fn into_stream(self) -> PnScramblerStream {
        PnScramblerStream::new(self.taps, self.width, self.seed)
    }
}

/// One byte of additive scrambling: XORs the PN low-bit into each of the 8 data
/// bits (LSB-first) while advancing `reg` in place. Shared by [`PnScrambler`]
/// and [`PnScramblerStream`] so the two produce bit-identical sequences.
#[inline]
fn scramble_byte(byte: u8, reg: &mut u32, taps: u32, top: u32, mask: u32) -> u8 {
    let mut out = 0u8;
    for bit in 0..8 {
        // PN bit is the register's low bit.
        let pn = (*reg & 1) as u8;
        out |= (((byte >> bit) & 1) ^ pn) << bit;
        // Advance: feedback = parity of tapped bits, shift right, insert
        // feedback at the top bit.
        let fb = (*reg & taps).count_ones() & 1;
        *reg = ((*reg >> 1) | (fb << top)) & mask;
    }
    out
}

/// Streaming additive PN scrambler: carries the LFSR register across `feed`
/// calls so a continuous byte stream is whitened as one unbroken PN sequence
/// (DVB energy dispersal runs this way). `feed` returns the scrambled bytes;
/// [`reset`](Self::reset) restarts from the seed. Self-inverse: feeding the
/// scrambled stream through a freshly-`reset` stream with the same parameters
/// recovers the original. A single `feed(&whole)` equals
/// [`PnScrambler::scramble`]`(&whole)`, and `reset` + chunked `feed`s equal one
/// `scramble` over the concatenation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PnScramblerStream {
    taps: u32,
    width: u32,
    seed: u32,
    mask: u32,
    top: u32,
    /// Live register state, carried across `feed` calls.
    reg: u32,
}

impl PnScramblerStream {
    /// Creates a streaming scrambler with feedback `taps` over a `width`-bit
    /// register, starting from `seed`. Same validity constraints as
    /// [`PnScrambler::new`].
    pub fn new(taps: u32, width: u32, seed: u32) -> Self {
        assert!((2..=32).contains(&width), "LFSR width must be in 2..=32");
        let mask = PnScrambler::mask_for(width);
        assert!(seed != 0, "LFSR seed must be nonzero");
        assert!(seed & !mask == 0, "LFSR seed must fit in `width` bits");
        assert!(taps & !mask == 0, "LFSR taps must fit in `width` bits");
        Self {
            taps,
            width,
            seed,
            mask,
            top: width - 1,
            reg: seed & mask,
        }
    }

    /// Restarts the register from `seed` (frame-orchestrator use: `reset` before
    /// each unit gives the same per-frame whitening as [`PnScrambler::scramble`]).
    pub fn reset(&mut self) {
        self.reg = self.seed & self.mask;
    }

    pub fn taps(&self) -> u32 {
        self.taps
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// Scrambles `data`, carrying register state across calls. Returns the
    /// scrambled bytes (input length preserved).
    pub fn feed(&mut self, data: &[u8]) -> Vec<u8> {
        data.iter()
            .map(|&b| scramble_byte(b, &mut self.reg, self.taps, self.top, self.mask))
            .collect()
    }

    /// Scrambles `data` in place, carrying register state across calls.
    pub fn feed_in_place(&mut self, data: &mut [u8]) {
        for byte in data.iter_mut() {
            *byte = scramble_byte(*byte, &mut self.reg, self.taps, self.top, self.mask);
        }
    }
}
