// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/fec/conv.rs
//
// Punctured convolutional coding for the inner FEC stage. Two mother codes are
// supported, selected by [`ConvCode`]:
//
//   • [`ConvCode::K5`] — the crate's original rate-1/2, constraint-length-5 code
//     (generators G0 = 0o25, G1 = 0o23), built on `codec::conv_encode`. This is
//     the default and is byte-identical to the pre-K7 behavior.
//   • [`ConvCode::DvbK7`] — DVB-T's rate-1/2, constraint-length-7 inner code
//     (generators G0 = 0o171, G1 = 0o133, ETSI EN 300 744 §4.3.3). Needed for a
//     conformant DVB-T payload.
//
// On top of the mother code this module adds:
//
//   • zero-tail termination — K-1 zero bits appended before encoding so the
//     trellis ends in the all-zero state, giving a clean per-frame block Viterbi
//     traceback (the streaming PSK31 decoder is fixed-lag; a frame code wants
//     block termination). K-1 = 4 for K5, 6 for K7.
//   • puncturing — deleting coded bits per a fixed per-rate matrix to raise the
//     rate from 1/2 to 2/3, 3/4, 5/6, or 7/8. The decoder reinserts an erasure
//     (LLR = 0, i.e. "no information") at each punctured position before the
//     Viterbi ACS. The puncture patterns are shared by both mother codes (the
//     standard DVB/802.11 patterns derived from a rate-1/2 code).
//
// The decoder is a soft-input (LLR-domain) Viterbi: the existing
// `codec::psk31::viterbi_decode` metric is hardwired to the DQPSK constellation
// and cannot consume `OfdmSoftDemod`'s per-bit LLRs. Here the branch metric is
// the LLR-correlation `Σ (1 - 2·c) · llr` over the branch's coded bits (LLR
// convention: positive ⇒ bit 0), maximized along the surviving path. It is
// generic over the mother code via a small [`ConvCode`] descriptor; the K5 path
// stays bit-identical to the original hand-rolled implementation.

use crate::codec::conv_encode;

/// Selects the convolutional mother code. Both are rate-1/2, zero-tail
/// terminated, and share the puncture matrices in [`PunctureRate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvCode {
    /// The crate's original constraint-length-5 code, G0 = 0o25, G1 = 0o23.
    /// Also the code PSK31 uses (`codec::conv_encode`). Default for backward
    /// compatibility.
    #[default]
    K5,
    /// DVB-T's constraint-length-7 code, G0 = 0o171, G1 = 0o133
    /// (ETSI EN 300 744 §4.3.3). The conformant DVB-T inner code.
    DvbK7,
}

impl ConvCode {
    /// Constraint length K (register width is K − 1).
    #[inline]
    pub const fn constraint_length(self) -> usize {
        match self {
            ConvCode::K5 => 5,
            ConvCode::DvbK7 => 7,
        }
    }

    /// Register width in bits, K − 1 (also the number of zero tail bits).
    #[inline]
    pub const fn reg_bits(self) -> usize {
        self.constraint_length() - 1
    }

    /// Number of trellis states, `2^(K−1)`.
    #[inline]
    pub const fn num_states(self) -> usize {
        1usize << self.reg_bits()
    }

    /// Number of zero tail bits appended to terminate the trellis, `K − 1`.
    #[inline]
    pub const fn tail_bits(self) -> usize {
        self.reg_bits()
    }

    /// Generator taps (G0, G1) as bit masks over a `K`-bit window whose low
    /// `K−1` bits are the register and whose top bit is the current input.
    ///
    /// K5: G0 = 0o25 = 0b10101, G1 = 0o23 = 0b10011 (matches
    /// `codec::conv_encode`). K7: G0 = 0o171 = 0b1111001, G1 = 0o133 =
    /// 0b1011011 (DVB-T). The MSB of each generator is the input tap and the
    /// LSB the oldest register bit, so the window packs `(input << (K-1)) |
    /// register`.
    #[inline]
    const fn generators(self) -> (u16, u16) {
        match self {
            ConvCode::K5 => (0b10101, 0b10011),
            ConvCode::DvbK7 => (0b1111001, 0b1011011),
        }
    }
}

/// Convolutional puncturing rate (numerator/denominator of the code rate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PunctureRate {
    /// The unpunctured mother code.
    R1_2,
    R2_3,
    R3_4,
    R5_6,
    R7_8,
}

impl PunctureRate {
    /// The puncturing matrix, as two rows (one per generator output G0/G1) of a
    /// fixed period. A `1` keeps the coded bit; a `0` deletes it. Standard
    /// DVB/802.11-style patterns derived from a rate-1/2 mother code.
    fn matrix(self) -> (&'static [u8], &'static [u8]) {
        match self {
            PunctureRate::R1_2 => (&[1], &[1]),
            PunctureRate::R2_3 => (&[1, 1], &[1, 0]),
            PunctureRate::R3_4 => (&[1, 1, 0], &[1, 0, 1]),
            PunctureRate::R5_6 => (&[1, 1, 0, 1, 0], &[1, 0, 1, 0, 1]),
            PunctureRate::R7_8 => (&[1, 1, 1, 1, 0, 1, 0], &[1, 0, 0, 0, 1, 0, 1]),
        }
    }

    /// Puncture period (columns in the matrix).
    fn period(self) -> usize {
        self.matrix().0.len()
    }

    /// Coded bits kept per period (matrix ones).
    fn kept_per_period(self) -> usize {
        let (g0, g1) = self.matrix();
        g0.iter().chain(g1.iter()).filter(|&&x| x == 1).count()
    }
}

/// Number of tail bits appended to terminate the K = 5 trellis. Retained for
/// the K5 public API; K7 uses [`ConvCode::tail_bits`].
pub const TAIL_BITS: usize = 4;

/// Coded bits `(c0, c1)` emitted when register state `s` takes input bit `b`
/// under mother code `code`.
#[inline]
fn branch_bits(code: ConvCode, s: u16, b: u8) -> (u8, u8) {
    let (g0, g1) = code.generators();
    let window = (((b & 1) as u16) << code.reg_bits()) | (s & reg_mask(code));
    (parity(window & g0), parity(window & g1))
}

/// Next register state after `s` receives input `b`: shift right, insert `b` at
/// the top register bit.
#[inline]
fn next_state(code: ConvCode, s: u16, b: u8) -> u16 {
    (s >> 1) | (((b & 1) as u16) << (code.reg_bits() - 1))
}

/// Low `reg_bits` mask for the register.
#[inline]
fn reg_mask(code: ConvCode) -> u16 {
    (1u16 << code.reg_bits()) - 1
}

#[inline]
fn parity(x: u16) -> u8 {
    (x.count_ones() & 1) as u8
}

/// Systematic-free rate-1/2 encode of `bits` (already tail-padded) under `code`,
/// returning the interleaved `[g0_0, g1_0, g0_1, g1_1, …]` mother-code output.
/// For [`ConvCode::K5`] this defers to `codec::conv_encode` so the output stays
/// bit-identical to the original path.
fn conv_encode_code(code: ConvCode, bits: &[u8]) -> Vec<u8> {
    if code == ConvCode::K5 {
        return conv_encode(bits);
    }
    let mut out = Vec::with_capacity(bits.len() * 2);
    let mut state: u16 = 0;
    for &b in bits {
        let (c0, c1) = branch_bits(code, state, b);
        out.push(c0);
        out.push(c1);
        state = next_state(code, state, b);
    }
    out
}

/// Encodes `info_bits` with the K5 mother code (zero-tail, punctured). Kept for
/// backward compatibility; equivalent to
/// [`conv_encode_punctured_with`]`(ConvCode::K5, …)`.
pub fn conv_encode_punctured(info_bits: &[u8], rate: PunctureRate) -> Vec<u8> {
    conv_encode_punctured_with(ConvCode::K5, info_bits, rate)
}

/// Encodes `info_bits` with mother code `code`, zero-tail termination, and the
/// given puncture rate, returning the coded (punctured) bit stream.
///
/// Layout before puncturing: `encode([info | (K−1) zero tail bits])`, an
/// interleaved `[g0_0, g1_0, g0_1, g1_1, …]` of length `2·(info + K − 1)`.
/// Puncturing then deletes bits per the rate matrix.
pub fn conv_encode_punctured_with(code: ConvCode, info_bits: &[u8], rate: PunctureRate) -> Vec<u8> {
    let mut padded = info_bits.to_vec();
    padded.extend(std::iter::repeat_n(0u8, code.tail_bits()));
    let coded = conv_encode_code(code, &padded);
    puncture(&coded, rate)
}

/// Deletes coded bits per the rate's puncture matrix. `coded` is the
/// interleaved `[g0, g1, g0, g1, …]` mother-code output.
fn puncture(coded: &[u8], rate: PunctureRate) -> Vec<u8> {
    if rate == PunctureRate::R1_2 {
        return coded.to_vec();
    }
    let (g0, g1) = rate.matrix();
    let period = rate.period();
    let mut out = Vec::with_capacity(coded.len());
    // Each trellis step contributes a (g0, g1) pair; step t uses matrix column
    // t % period.
    let n_steps = coded.len() / 2;
    for t in 0..n_steps {
        let col = t % period;
        if g0[col] == 1 {
            out.push(coded[t * 2]);
        }
        if g1[col] == 1 {
            out.push(coded[t * 2 + 1]);
        }
    }
    out
}

/// Number of coded bits [`conv_encode_punctured`] produces for `info_bits`
/// information bits at `rate` under the K5 code.
pub fn punctured_coded_len(info_bits: usize, rate: PunctureRate) -> usize {
    punctured_coded_len_with(ConvCode::K5, info_bits, rate)
}

/// Number of coded bits [`conv_encode_punctured_with`] produces for `info_bits`
/// information bits at `rate` under mother code `code` (deterministic; used by
/// the frame layer's size bookkeeping).
pub fn punctured_coded_len_with(code: ConvCode, info_bits: usize, rate: PunctureRate) -> usize {
    let n_steps = info_bits + code.tail_bits(); // mother code emits 2 bits/step
    if rate == PunctureRate::R1_2 {
        return n_steps * 2;
    }
    let period = rate.period();
    let full_periods = n_steps / period;
    let rem = n_steps % period;
    let (g0, g1) = rate.matrix();
    let mut len = full_periods * rate.kept_per_period();
    for col in 0..rem {
        len += (g0[col] + g1[col]) as usize;
    }
    len
}

/// Soft-decision Viterbi decode of a K5, punctured, zero-tail-terminated
/// stream. Kept for backward compatibility; equivalent to
/// [`viterbi_decode_soft_with`]`(ConvCode::K5, …)`.
pub fn viterbi_decode_soft(coded_llrs: &[f32], info_bits: usize, rate: PunctureRate) -> Vec<u8> {
    viterbi_decode_soft_with(ConvCode::K5, coded_llrs, info_bits, rate)
}

/// Soft-decision Viterbi decode of a punctured, zero-tail-terminated stream
/// under mother code `code`.
///
/// `coded_llrs` are the received per-coded-bit LLRs (positive ⇒ bit 0), in the
/// punctured order. `info_bits` is the number of information bits to recover
/// (the tail bits are decoded but dropped). Punctured positions are treated as
/// erasures (LLR 0). Returns the `info_bits` recovered information bits.
pub fn viterbi_decode_soft_with(
    code: ConvCode,
    coded_llrs: &[f32],
    info_bits: usize,
    rate: PunctureRate,
) -> Vec<u8> {
    let n_steps = info_bits + code.tail_bits();
    let num_states = code.num_states();

    // Depuncture: rebuild the full 2-per-step LLR stream, inserting 0.0 at
    // deleted positions.
    let mut full = vec![0.0f32; n_steps * 2];
    if rate == PunctureRate::R1_2 {
        let n = coded_llrs.len().min(full.len());
        full[..n].copy_from_slice(&coded_llrs[..n]);
    } else {
        let (g0, g1) = rate.matrix();
        let period = rate.period();
        let mut src = 0usize;
        for t in 0..n_steps {
            let col = t % period;
            if g0[col] == 1 {
                if src < coded_llrs.len() {
                    full[t * 2] = coded_llrs[src];
                }
                src += 1;
            }
            if g1[col] == 1 {
                if src < coded_llrs.len() {
                    full[t * 2 + 1] = coded_llrs[src];
                }
                src += 1;
            }
        }
    }

    // Forward ACS. Metrics are correlations to be MAXIMIZED: for a branch with
    // coded bits (c0, c1), the contribution is `(1-2c0)·llr0 + (1-2c1)·llr1`
    // (a positive llr favors bit 0, so `(1-2·0)=+1` rewards agreement).
    let neg_inf = f32::MIN / 2.0;
    let mut pm = vec![neg_inf; num_states];
    pm[0] = 0.0; // known start state
    let mut prev_state_table: Vec<Vec<u16>> = vec![vec![0u16; num_states]; n_steps];
    let top_bit = code.reg_bits() - 1;

    let mut new_pm = vec![neg_inf; num_states];
    for t in 0..n_steps {
        let l0 = full[t * 2];
        let l1 = full[t * 2 + 1];
        new_pm.iter_mut().for_each(|m| *m = neg_inf);
        for (prev, &pm_prev) in pm.iter().enumerate() {
            if pm_prev <= neg_inf {
                continue;
            }
            for &bit in &[0u8, 1u8] {
                let (c0, c1) = branch_bits(code, prev as u16, bit);
                let corr = (1.0 - 2.0 * c0 as f32) * l0 + (1.0 - 2.0 * c1 as f32) * l1;
                let ns = next_state(code, prev as u16, bit) as usize;
                let cand = pm_prev + corr;
                if cand > new_pm[ns] {
                    new_pm[ns] = cand;
                    prev_state_table[t][ns] = prev as u16;
                }
            }
        }
        std::mem::swap(&mut pm, &mut new_pm);
    }

    // With zero-tail termination the ending state is 0.
    let mut state = 0usize;
    let mut bits = vec![0u8; n_steps];
    for t in (0..n_steps).rev() {
        let prev = prev_state_table[t][state] as usize;
        // The input bit driving prev → state is the top register bit of `state`
        // (next_state inserts b at bit `reg_bits-1`).
        bits[t] = ((state >> top_bit) & 1) as u8;
        state = prev;
    }

    bits.truncate(info_bits);
    bits
}
