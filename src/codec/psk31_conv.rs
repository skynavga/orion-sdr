// src/codec/psk31_conv.rs
//
// Rate-1/2, constraint-length-5 convolutional encoder and Viterbi decoder
// for QPSK31.
//
// Generator polynomials (octal): G0 = 25, G1 = 23.
//   G0 = 0b10101 — taps at bits {0, 2, 4} of the 5-bit shift register
//   G1 = 0b10011 — taps at bits {0, 1, 4}
//
// Shift register convention:
//   sr is a 4-bit register holding the K-1 = 4 past input bits.
//   For each new input bit b, the 5-bit encoder window is:
//     window = (b << 4) | sr      (b is the newest bit, sr[3] is oldest)
//   After encoding:
//     sr = (sr >> 1) | (b << 3)   (shift right, insert b at MSB of 4-bit sr)
//
// Interleaving: for each input bit the encoder emits [g0_bit, g1_bit].
// Output length = 2 × input.len().
//
// Viterbi decoder:
//   16 states (2^(K-1) = 2^4 = 16).
//   `decisions[t][s]` stores the previous state that reached state s at time t
//   with the minimum accumulated metric.
//   Traceback follows the chain of previous states from the best final state.

// ── Encoder ───────────────────────────────────────────────────────────────────

/// Parity of all set bits in `x` — returns 0 or 1.
#[inline(always)]
fn parity(x: u8) -> u8 {
    let x = x ^ (x >> 4);
    let x = x ^ (x >> 2);
    (x ^ (x >> 1)) & 1
}

/// Rate-1/2, K=5 convolutional encoder.
///
/// G0 = 0b10101 (octal 25), G1 = 0b10011 (octal 23).
///
/// Input: slice of bits (each byte = 0 or 1).
/// Output: interleaved coded bits `[g0_0, g1_0, g0_1, g1_1, …]`; length = 2 × input.len().
pub fn conv_encode(bits: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bits.len() * 2);
    let mut sr: u8 = 0;
    for &b in bits {
        let window = ((b & 1) << 4) | (sr & 0x0F);
        out.push(parity(window & 0b10101)); // G0
        out.push(parity(window & 0b10011)); // G1
        sr = (sr >> 1) | ((b & 1) << 3);
    }
    out
}

// ── Viterbi decoder ───────────────────────────────────────────────────────────

const NUM_STATES: usize = 16;

/// Compute the two coded bits produced when state `s` receives input `b`.
#[inline]
fn branch_bits(s: u8, b: u8) -> (u8, u8) {
    let window = ((b & 1) << 4) | (s & 0x0F);
    (parity(window & 0b10101), parity(window & 0b10011))
}

/// Next state when current state is `s` and input `b` is received.
#[inline]
fn next_state(s: u8, b: u8) -> u8 {
    (s >> 1) | ((b & 1) << 3)
}

// DQPSK expected phasors indexed by dibit = c0*2 + c1.
// Matches the QPSK31_PHASE_STEP table in src/modulate/psk31.rs:
//   dibit 0 (c0=0,c1=0): step = (+1,  0)
//   dibit 1 (c0=0,c1=1): step = ( 0, -1)
//   dibit 2 (c0=1,c1=0): step = ( 0, +1)
//   dibit 3 (c0=1,c1=1): step = (-1,  0)
const DQPSK_EXP: [(f32, f32); 4] = [
    ( 1.0,  0.0), // dibit 0
    ( 0.0, -1.0), // dibit 1
    ( 0.0,  1.0), // dibit 2
    (-1.0,  0.0), // dibit 3
];

/// Soft Viterbi decoder for the rate-1/2, K=5 code.
///
/// `soft` is an interleaved slice `[re_0, im_0, re_1, im_1, …]` of DQPSK
/// differential-detection outputs.  The branch metric uses the actual DQPSK
/// constellation phasors as expected values, matching the modulator's
/// `QPSK31_PHASE_STEP` table.
///
/// Returns a decoded bit slice of length `soft.len() / 2`.
pub fn viterbi_decode(soft: &[f32]) -> Vec<u8> {
    let n_syms = soft.len() / 2;
    if n_syms == 0 {
        return Vec::new();
    }

    // Path metrics — lower is better (minimising Euclidean distance).
    let inf = f32::MAX / 2.0;
    let mut pm = [inf; NUM_STATES];
    pm[0] = 0.0;

    // decisions[t][s] = previous state that led to state `s` at time `t`.
    // Stored as u8 (state index 0–15).
    let mut prev_state_table: Vec<[u8; NUM_STATES]> = vec![[0u8; NUM_STATES]; n_syms];

    for t in 0..n_syms {
        let s0 = soft[t * 2];
        let s1 = soft[t * 2 + 1];
        let mut new_pm = [inf; NUM_STATES];

        for prev in 0..NUM_STATES {
            if pm[prev] >= inf { continue; }
            for &bit in &[0u8, 1u8] {
                let (c0, c1) = branch_bits(prev as u8, bit);
                // Expected soft values are the DQPSK phasor for this dibit,
                // not ±1 on both axes — the DQPSK constellation places all
                // energy on a single axis per symbol.
                let dibit = (c0 & 1) * 2 + (c1 & 1);
                let (exp0, exp1) = DQPSK_EXP[dibit as usize];
                let bm = (s0 - exp0) * (s0 - exp0) + (s1 - exp1) * (s1 - exp1);
                let ns = next_state(prev as u8, bit) as usize;
                let cand = pm[prev] + bm;
                if cand < new_pm[ns] {
                    new_pm[ns] = cand;
                    prev_state_table[t][ns] = prev as u8;
                }
            }
        }
        pm = new_pm;
    }

    // Best final state.
    let mut state = pm
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Traceback: recover the sequence of decoded bits.
    let mut bits_out = vec![0u8; n_syms];
    for t in (0..n_syms).rev() {
        let prev = prev_state_table[t][state] as usize;
        // The input bit that caused the transition prev → state:
        // next_state(prev, b) == state
        // => (prev >> 1) | (b << 3) == state
        // => b = (state >> 3) & 1  (MSB of state came from input bit b)
        let b = (state >> 3) as u8 & 1;
        bits_out[t] = b;
        state = prev;
    }

    bits_out
}

/// Hard-decision Viterbi decoder (for testing with noiseless hard bits).
///
/// Input: interleaved coded bits `[c0_0, c1_0, c0_1, c1_1, …]`.
/// Converts each (c0, c1) pair to the corresponding DQPSK phasor before
/// calling the soft decoder, matching the branch metric convention.
pub fn viterbi_decode_hard(bits: &[u8]) -> Vec<u8> {
    let n_syms = bits.len() / 2;
    let mut soft = Vec::with_capacity(n_syms * 2);
    for i in 0..n_syms {
        let c0 = bits[i * 2] & 1;
        let c1 = bits[i * 2 + 1] & 1;
        let dibit = c0 * 2 + c1;
        let (re, im) = DQPSK_EXP[dibit as usize];
        soft.push(re);
        soft.push(im);
    }
    viterbi_decode(&soft)
}
