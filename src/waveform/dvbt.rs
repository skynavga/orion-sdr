// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/waveform/dvbt.rs
//
// DVB-T (and narrowband DVB-T for amateur DATV) waveform assembly. This module
// holds the DVB-T-specific parameters and coding pieces that the generic
// `fec/`/`multicarrier/`/`modulate/` primitives are configured with. It grows
// across the NB-DVB-T phases; this first piece is the energy-dispersal whitener.
//
// ENERGY DISPERSAL (ETSI EN 300 744 §4.3.1). DVB-T whitens the transport stream
// with a PRBS from the polynomial 1 + X^14 + X^15, the 15-bit register
// initialized to `100101010000000`. Two conventions distinguish it from the
// crate's generic `PnScrambler`/`PnScramblerStream` (which are LSB-first with a
// low-bit output tap):
//
//   • the PRBS output is taken from the last register stage and MSB-first per
//     byte — the first PRBS bit XORs the MSB of the first payload byte;
//   • the first 8 PRBS output bits are `0000 0011` (= 0x03), a known-answer
//     anchor from the standard.
//
// SCOPE (Phase 1): this is the *whitener algorithm* — the continuous PRBS applied
// to the payload byte stream. The full DVB-T energy dispersal also re-inits the
// PRBS every 8 transport packets and inverts the first MPEG-2 sync byte
// (0x47 -> 0xB8, sync bytes skipped-but-clocked); that 188-byte-TS-packet framing
// lands with the TS-packet payload layer in a later phase. Applied continuously,
// this whitener is self-inverse: running a byte stream through twice from the
// same init recovers it.

/// DVB-T energy-dispersal PRBS register initialization: `100101010000000`
/// (stage 1 = MSB), i.e. `0b100_1010_1000_0000` over the low 15 bits.
pub const DVBT_PRBS_INIT: u16 = 0b100_1010_1000_0000;

/// A DVB-T energy-dispersal whitener: the standard's 1 + X^14 + X^15 PRBS,
/// MSB-first, carried as streaming state across [`feed`](Self::feed) calls.
/// Self-inverse (XOR of a data-independent PRBS), so the same whitener recovers a
/// stream it scrambled once it is [`reset`](Self::reset).
///
/// This is the whitener algorithm only (see the module header): the 8-packet
/// re-init and sync-byte inversion are part of the later TS-packet framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DvbtEnergyDispersal {
    /// Live 15-bit PRBS register (low 15 bits used).
    reg: u16,
}

impl Default for DvbtEnergyDispersal {
    fn default() -> Self {
        Self::new()
    }
}

impl DvbtEnergyDispersal {
    /// A whitener starting from the standard init `100101010000000`.
    pub fn new() -> Self {
        Self {
            reg: DVBT_PRBS_INIT,
        }
    }

    /// Reloads the standard init sequence (frame/packet-group re-init).
    pub fn reset(&mut self) {
        self.reg = DVBT_PRBS_INIT;
    }

    /// Advances the PRBS one step and returns the output bit. The register holds
    /// stages `x1..x15` with `x1` at bit 14 (the init's MSB) and `x15` at bit 0.
    /// Feedback is `x15 XOR x14` (bits 0 and 1, the taps of 1 + X^14 + X^15); the
    /// register shifts toward `x15` (right) with the feedback inserted at `x1`
    /// (bit 14), and the output bit is that feedback. Verified against the
    /// standard's `0000 0011` first-8-bits known answer.
    #[inline]
    fn next_bit(&mut self) -> u8 {
        let fb = (self.reg ^ (self.reg >> 1)) & 1;
        self.reg = (self.reg >> 1) | (fb << 14);
        fb as u8
    }

    /// XORs the PRBS over `data` in place, MSB-first per byte, carrying register
    /// state across calls. Run once to scramble, and again from the same init to
    /// descramble.
    pub fn feed_in_place(&mut self, data: &mut [u8]) {
        for byte in data.iter_mut() {
            let mut out = 0u8;
            for bit in (0..8).rev() {
                let pn = self.next_bit();
                out |= ((((*byte >> bit) & 1) ^ pn) & 1) << bit;
            }
            *byte = out;
        }
    }

    /// XORs the PRBS over `data`, MSB-first, returning the whitened bytes.
    pub fn feed(&mut self, data: &[u8]) -> Vec<u8> {
        let mut out = data.to_vec();
        self.feed_in_place(&mut out);
        out
    }
}

// ── DVB-T constellation mapping (ETSI EN 300 744 §4.3.5, Figure 9a) ─────────
//
// DVB-T maps the v bits y0..y(v-1) of a symbol so that the **even** bits
// (y0,y2,y4) select the in-phase axis and the **odd** bits (y1,y3,y5) select the
// quadrature axis (v = 2 for QPSK, 4 for 16-QAM, 6 for 64-QAM; non-hierarchical
// uses the α = 1 uniform constellation). Each axis is a Gray map with the
// lowest-index bit as the MSB. The per-axis index→level tables below are read
// directly off Figure 9a (V1.6.2) — the axis MSB is effectively a sign bit, and
// the levels are the odd integers ±1, ±3, … , ±(M−1) with `M = 2^(v/2)` levels
// per axis.
//
// This differs from `modulate::qam`'s generic mapper in both the axis-bit
// assignment (DVB-T interleaves even/odd bits across the axes; the generic
// mapper groups the first half of the bits on I and the second half on Q) and
// the exact Gray labelling, so it is a distinct DVB-T-specific mapping selected
// for DVB-T configs. Amplitudes are normalized to unit average symbol energy
// (matching `modulate::qam::axis_scale`), so the constellation power is the same
// as the generic mapper's.
//
// LLR/soft demap is not provided here: the frame layer's soft demapper consumes
// the generic mapper's convention, so DVB-T soft-decision RX is wired when the
// DVB-T config path is assembled; this module supplies the mapping tables and a
// hard nearest-point demap for round-trip/known-answer verification.

/// Per-axis DVB-T level tables (index = the axis's bits, lowest-y-index as MSB),
/// from Figure 9a. Unnormalized odd-integer levels.
const DVBT_AXIS_QPSK: [i32; 2] = [1, -1];
const DVBT_AXIS_16QAM: [i32; 4] = [3, 1, -3, -1];
const DVBT_AXIS_64QAM: [i32; 8] = [7, 5, 1, 3, -7, -5, -1, -3];

/// Returns the unnormalized per-axis level table for `v` bits per symbol
/// (`v ∈ {2, 4, 6}` → QPSK/16-QAM/64-QAM), or `None` for an unsupported order.
fn dvbt_axis_table(v: usize) -> Option<&'static [i32]> {
    match v {
        2 => Some(&DVBT_AXIS_QPSK),
        4 => Some(&DVBT_AXIS_16QAM),
        6 => Some(&DVBT_AXIS_64QAM),
        _ => None,
    }
}

/// Packs the axis bits (MSB-first) into a table index.
#[inline]
fn axis_index(bits: &[u8]) -> usize {
    bits.iter()
        .fold(0usize, |acc, &b| (acc << 1) | (b & 1) as usize)
}

/// Maps one DVB-T symbol's `v` bits `y0..y(v-1)` to a normalized constellation
/// point. Even bits (`y0,y2,…`) form the I axis, odd bits (`y1,y3,…`) the Q axis,
/// each Gray-mapped per Figure 9a. Returns `None` if `bits.len()` is not a
/// supported `v ∈ {2,4,6}`.
pub fn dvbt_map_symbol(bits: &[u8]) -> Option<num_complex::Complex32> {
    let v = bits.len();
    let table = dvbt_axis_table(v)?;
    let scale = crate::modulate::qam::axis_scale(v);
    // De-interleave: even indices → I, odd indices → Q (both MSB-first in y-order).
    let i_bits: Vec<u8> = bits.iter().step_by(2).copied().collect();
    let q_bits: Vec<u8> = bits.iter().skip(1).step_by(2).copied().collect();
    let i = table[axis_index(&i_bits)] as f32 * scale;
    let q = table[axis_index(&q_bits)] as f32 * scale;
    Some(num_complex::Complex32::new(i, q))
}

/// Hard nearest-point demap: inverse of [`dvbt_map_symbol`] for `v` bits. Finds
/// the nearest per-axis level and recovers the `v` bits `y0..y(v-1)`. Returns
/// `None` for an unsupported order.
pub fn dvbt_demap_symbol(sym: num_complex::Complex32, v: usize) -> Option<Vec<u8>> {
    let table = dvbt_axis_table(v)?;
    let scale = crate::modulate::qam::axis_scale(v);
    let k = v / 2; // bits per axis
    // Nearest table index on one axis for a received (normalized) coordinate.
    let nearest = |coord: f32| -> usize {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (idx, &lvl) in table.iter().enumerate() {
            let d = (coord - lvl as f32 * scale).abs();
            if d < best_d {
                best_d = d;
                best = idx;
            }
        }
        best
    };
    let i_idx = nearest(sym.re);
    let q_idx = nearest(sym.im);
    // Unpack each axis index back to k MSB-first bits, then re-interleave.
    let unpack = |idx: usize| -> Vec<u8> { (0..k).rev().map(|b| ((idx >> b) & 1) as u8).collect() };
    let ib = unpack(i_idx);
    let qb = unpack(q_idx);
    let mut out = vec![0u8; v];
    for j in 0..k {
        out[2 * j] = ib[j]; // even → I
        out[2 * j + 1] = qb[j]; // odd → Q
    }
    Some(out)
}
