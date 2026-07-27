// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/fec/conv.rs
//
// Punctured convolutional coding for the inner FEC stage, built on the crate's
// existing rate-1/2, constraint-length-5 mother code (`codec::conv_encode`,
// generators G0 = 0o25, G1 = 0o23). Two pieces the base code lacks are added
// here:
//
//   • zero-tail termination — K-1 = 4 zero bits appended before encoding so the
//     trellis ends in the all-zero state, giving a clean per-frame block
//     Viterbi traceback (the streaming PSK31 decoder is fixed-lag; a frame code
//     wants block termination);
//   • puncturing — deleting coded bits per a fixed per-rate matrix to raise the
//     rate from 1/2 to 2/3, 3/4, 5/6, or 7/8. The decoder reinserts an erasure
//     (LLR = 0, i.e. "no information") at each punctured position before the
//     Viterbi ACS.
//
// The decoder is a NEW soft-input (LLR-domain) Viterbi: the existing
// `codec::psk31::viterbi_decode` metric is hardwired to the DQPSK constellation
// and cannot consume `OfdmSoftDemod`'s per-bit LLRs. Here the branch metric is
// the LLR-correlation `Σ (1 - 2·c) · llr` over the branch's coded bits (LLR
// convention: positive ⇒ bit 0), maximized along the surviving path.

use crate::codec::conv_encode;

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
    /// DVB/802.11-style patterns derived from the rate-1/2 mother code.
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

/// Number of tail bits appended to terminate the K = 5 trellis.
pub const TAIL_BITS: usize = 4;

const NUM_STATES: usize = 16;

/// Coded bits `(c0, c1)` emitted when state `s` (4-bit register) takes input
/// bit `b`. Matches `codec::conv_encode`'s G0 = 0b10101, G1 = 0b10011.
#[inline]
fn branch_bits(s: u8, b: u8) -> (u8, u8) {
    let window = ((b & 1) << 4) | (s & 0x0F);
    (parity(window & 0b10101), parity(window & 0b10011))
}

/// Next 4-bit register state after state `s` receives input `b`.
#[inline]
fn next_state(s: u8, b: u8) -> u8 {
    (s >> 1) | ((b & 1) << 3)
}

#[inline]
fn parity(x: u8) -> u8 {
    let x = x ^ (x >> 4);
    let x = x ^ (x >> 2);
    (x ^ (x >> 1)) & 1
}

/// Encodes `info_bits` with zero-tail termination and the given puncture rate,
/// returning the coded (punctured) bit stream.
///
/// Layout before puncturing: `conv_encode([info | 4 zero tail bits])`, an
/// interleaved `[g0_0, g1_0, g0_1, g1_1, …]` of length `2·(info+4)`. Puncturing
/// then deletes bits per the rate matrix.
pub fn conv_encode_punctured(info_bits: &[u8], rate: PunctureRate) -> Vec<u8> {
    let mut padded = info_bits.to_vec();
    padded.extend(std::iter::repeat_n(0u8, TAIL_BITS));
    let coded = conv_encode(&padded); // [g0_0, g1_0, g0_1, g1_1, …]
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

/// Number of coded bits `conv_encode_punctured` produces for `info_bits`
/// information bits at `rate` (deterministic; used by the frame layer's size
/// bookkeeping).
pub fn punctured_coded_len(info_bits: usize, rate: PunctureRate) -> usize {
    let n_steps = info_bits + TAIL_BITS; // mother code emits 2 bits/step
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

/// Soft-decision Viterbi decode of a punctured, zero-tail-terminated stream.
///
/// `coded_llrs` are the received per-coded-bit LLRs (positive ⇒ bit 0), in the
/// punctured order. `info_bits` is the number of information bits to recover
/// (the tail bits are decoded but dropped). Punctured positions are treated as
/// erasures (LLR 0). Returns the `info_bits` recovered information bits.
pub fn viterbi_decode_soft(coded_llrs: &[f32], info_bits: usize, rate: PunctureRate) -> Vec<u8> {
    let n_steps = info_bits + TAIL_BITS;

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
    let mut pm = [neg_inf; NUM_STATES];
    pm[0] = 0.0; // known start state
    let mut prev_state_table: Vec<[u8; NUM_STATES]> = vec![[0u8; NUM_STATES]; n_steps];

    for t in 0..n_steps {
        let l0 = full[t * 2];
        let l1 = full[t * 2 + 1];
        let mut new_pm = [neg_inf; NUM_STATES];
        for (prev, &pm_prev) in pm.iter().enumerate().take(NUM_STATES) {
            if pm_prev <= neg_inf {
                continue;
            }
            for &bit in &[0u8, 1u8] {
                let (c0, c1) = branch_bits(prev as u8, bit);
                let corr = (1.0 - 2.0 * c0 as f32) * l0 + (1.0 - 2.0 * c1 as f32) * l1;
                let ns = next_state(prev as u8, bit) as usize;
                let cand = pm_prev + corr;
                if cand > new_pm[ns] {
                    new_pm[ns] = cand;
                    prev_state_table[t][ns] = prev as u8;
                }
            }
        }
        pm = new_pm;
    }

    // With zero-tail termination the ending state is 0.
    let mut state = 0usize;
    let mut bits = vec![0u8; n_steps];
    for t in (0..n_steps).rev() {
        let prev = prev_state_table[t][state] as usize;
        // The input bit driving prev → state is the top bit of `state`
        // (next_state inserts b at bit 3).
        bits[t] = ((state >> 3) & 1) as u8;
        state = prev;
    }

    bits.truncate(info_bits);
    bits
}
