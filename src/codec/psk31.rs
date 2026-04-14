// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/codec/psk31.rs
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
pub const DQPSK_EXP: [(f32, f32); 4] = [
    (1.0, 0.0),  // dibit 0
    (0.0, -1.0), // dibit 1
    (0.0, 1.0),  // dibit 2
    (-1.0, 0.0), // dibit 3
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

        for (prev, &pm_prev) in pm.iter().enumerate().take(NUM_STATES) {
            if pm_prev >= inf {
                continue;
            }
            for &bit in &[0u8, 1u8] {
                let (c0, c1) = branch_bits(prev as u8, bit);
                // Expected soft values are the DQPSK phasor for this dibit,
                // not ±1 on both axes — the DQPSK constellation places all
                // energy on a single axis per symbol.
                let dibit = (c0 & 1) * 2 + (c1 & 1);
                let (exp0, exp1) = DQPSK_EXP[dibit as usize];
                let bm = (s0 - exp0) * (s0 - exp0) + (s1 - exp1) * (s1 - exp1);
                let ns = next_state(prev as u8, bit) as usize;
                let cand = pm_prev + bm;
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

/// Coherent soft Viterbi decoder for the rate-1/2, K=5 code.
///
/// `soft` is an interleaved slice `[re_0, im_0, re_1, im_1, …]` of phase-
/// corrected absolute symbol estimates from the DFM.  Each state tracks a
/// hypothesised absolute phasor; the branch metric is the squared Euclidean
/// distance between the received symbol and the hypothesis, eliminating the
/// ~3 dB noise-product penalty of differential detection.
///
/// `phase_steps` is the DQPSK phase-step table indexed by dibit = c0*2+c1,
/// matching the modulator's `QPSK31_PHASE_STEP`.
///
/// Returns a decoded bit slice of length `soft.len() / 2`.
pub fn viterbi_decode_coherent(soft: &[f32], phase_steps: &[(f32, f32); 4]) -> Vec<u8> {
    let n_syms = soft.len() / 2;
    if n_syms == 0 {
        return Vec::new();
    }

    let inf = f32::MAX / 2.0;
    let mut pm = [inf; NUM_STATES];
    pm[0] = 0.0;
    // Hypothesised absolute phasor per state.  Initial phasor (1,0) matches
    // Qpsk31Mod starting phase.
    let mut hyp = [(1.0f32, 0.0f32); NUM_STATES];

    let mut prev_state_table: Vec<[u8; NUM_STATES]> = vec![[0u8; NUM_STATES]; n_syms];
    // Store hypothesised phasor that won each transition, for propagation.
    let mut hyp_table: Vec<[(f32, f32); NUM_STATES]> = vec![[(0.0, 0.0); NUM_STATES]; n_syms];

    for t in 0..n_syms {
        let s_re = soft[t * 2];
        let s_im = soft[t * 2 + 1];
        let mut new_pm = [inf; NUM_STATES];
        let mut new_hyp = [(0.0f32, 0.0f32); NUM_STATES];

        for prev in 0..NUM_STATES {
            if pm[prev] >= inf {
                continue;
            }
            let (h_re, h_im) = hyp[prev];
            for &bit in &[0u8, 1u8] {
                let (c0, c1) = branch_bits(prev as u8, bit);
                let dibit = (c0 & 1) * 2 + (c1 & 1);
                let (step_re, step_im) = phase_steps[dibit as usize];
                // Hypothesised phasor after this transition: h * step.
                let nh_re = h_re * step_re - h_im * step_im;
                let nh_im = h_im * step_re + h_re * step_im;
                // Coherent branch metric: |sym_c - hyp|².
                let bm = (s_re - nh_re) * (s_re - nh_re) + (s_im - nh_im) * (s_im - nh_im);
                let ns = next_state(prev as u8, bit) as usize;
                let cand = pm[prev] + bm;
                if cand < new_pm[ns] {
                    new_pm[ns] = cand;
                    new_hyp[ns] = (nh_re, nh_im);
                    prev_state_table[t][ns] = prev as u8;
                    hyp_table[t][ns] = (nh_re, nh_im);
                }
            }
        }
        pm = new_pm;
        hyp = new_hyp;
    }

    // Best final state.
    let mut state = pm
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Traceback.
    let mut bits_out = vec![0u8; n_syms];
    for t in (0..n_syms).rev() {
        let prev = prev_state_table[t][state] as usize;
        let b = (state >> 3) as u8 & 1;
        bits_out[t] = b;
        state = prev;
    }

    bits_out
}

// ── Streaming coherent Viterbi decoder ────────────────────────────────────────

/// Fixed-lag sliding-window Viterbi decoder for coherent QPSK31.
///
/// Processes one QPSK symbol at a time and emits decoded bits with a fixed
/// latency of `TRACEBACK_DEPTH` symbols.  Uses the same coherent branch
/// metric as `viterbi_decode_coherent` (hypothesised absolute phasor per
/// trellis state).
///
/// Based on the fldigi approach: after each ACS step, if enough history has
/// accumulated, traceback from the best-metric state and emit the oldest
/// undecoded bit.
pub struct StreamingViterbi {
    pm: [f32; NUM_STATES],
    history: Vec<[u8; NUM_STATES]>, // circular buffer of prev_state
    ptr: usize,                     // write pointer into circular buffer
    count: usize,                   // total symbols processed
    phase_steps: [(f32, f32); 4],
}

/// Traceback depth.  Textbook = 5×(K-1) = 20 for rate-1/2 K=5.
/// Use 32 for extra convergence margin with differential detection.
const TRACEBACK_DEPTH: usize = 32;
/// Circular buffer size (must be > TRACEBACK_DEPTH).
const PATHMEM: usize = 128;

impl StreamingViterbi {
    /// Create a new streaming Viterbi decoder.
    /// `phase_steps` is the DQPSK phase-step table (same as for batch decoder).
    pub fn new(phase_steps: &[(f32, f32); 4]) -> Self {
        let inf = f32::MAX / 2.0;
        let mut pm = [inf; NUM_STATES];
        pm[0] = 0.0;
        Self {
            pm,
            history: vec![[0u8; NUM_STATES]; PATHMEM],
            ptr: 0,
            count: 0,
            phase_steps: *phase_steps,
        }
    }

    /// Feed one QPSK symbol.
    ///
    /// `s_re, s_im` are the DQPSK differential-detection outputs (or
    /// phase-corrected absolute phasors from the coherent demod).  The branch
    /// metric compares the received symbol against the expected DQPSK step
    /// phasors — this is a non-coherent metric that doesn't require tracking
    /// absolute phase, making it robust to PLL lock-in transients.
    pub fn feed_symbol(&mut self, s_re: f32, s_im: f32) -> Option<u8> {
        let inf = f32::MAX / 2.0;
        let mut new_pm = [inf; NUM_STATES];

        // ACS step — non-coherent DQPSK branch metric.
        for prev in 0..NUM_STATES {
            if self.pm[prev] >= inf {
                continue;
            }
            for &bit in &[0u8, 1u8] {
                let (c0, c1) = branch_bits(prev as u8, bit);
                let dibit = (c0 & 1) * 2 + (c1 & 1);
                let (exp_re, exp_im) = self.phase_steps[dibit as usize];
                let bm = (s_re - exp_re) * (s_re - exp_re) + (s_im - exp_im) * (s_im - exp_im);
                let ns = next_state(prev as u8, bit) as usize;
                let cand = self.pm[prev] + bm;
                if cand < new_pm[ns] {
                    new_pm[ns] = cand;
                    self.history[self.ptr][ns] = prev as u8;
                }
            }
        }
        self.pm = new_pm;

        // Periodic metric normalisation to prevent f32 overflow.
        if self.count % 256 == 255 {
            let min_pm = self
                .pm
                .iter()
                .copied()
                .filter(|&v| v < inf)
                .fold(inf, f32::min);
            if min_pm > 0.0 {
                for p in &mut self.pm {
                    if *p < inf {
                        *p -= min_pm;
                    }
                }
            }
        }

        self.ptr = (self.ptr + 1) % PATHMEM;
        self.count += 1;

        // Not enough history for traceback yet.
        if self.count <= TRACEBACK_DEPTH {
            return None;
        }

        // Fixed-lag traceback: find best state now, trace back TRACEBACK_DEPTH
        // steps, emit the decoded bit at the traceback endpoint.
        let mut state = self
            .pm
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut p = (self.ptr + PATHMEM - 1) % PATHMEM; // current position
        for _ in 0..TRACEBACK_DEPTH {
            state = self.history[p][state] as usize;
            p = (p + PATHMEM - 1) % PATHMEM;
        }

        // The decoded bit is the MSB of the state at the traceback endpoint
        // (same as batch: b = (state >> 3) & 1).
        Some(((state >> 3) & 1) as u8)
    }

    /// Flush remaining bits after the last symbol.  Returns up to
    /// `TRACEBACK_DEPTH` final decoded bits by tracing back from the best
    /// state at progressively shorter depths.
    pub fn flush(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        // Feed TRACEBACK_DEPTH zero-energy symbols to push out the tail.
        for _ in 0..TRACEBACK_DEPTH {
            if let Some(b) = self.feed_symbol(0.0, 0.0) {
                out.push(b);
            }
        }
        out
    }
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

// ── Streaming PSK31 decode pipeline ──────────────────────────────────────────

use crate::Block;
use crate::codec::varicode::VaricodeDecoder;
use crate::demodulate::psk31::{Bpsk31Decider, Bpsk31Demod, Qpsk31Demod};
use num_complex::Complex32 as C32;

/// Persistent streaming PSK31 decode state.
///
/// Wires together the demod → decider/viterbi → varicode pipeline and
/// tracks how far into the IQ buffer has been processed.
///
/// BPSK31: fully incremental — `Bpsk31Decider` produces hard bits instantly,
/// which are pushed through the `VaricodeDecoder` character by character.
///
/// QPSK31: `Qpsk31Demod` produces differential soft symbols; each symbol is
/// fed through `StreamingViterbi` which emits decoded bits with a fixed
/// latency of 32 symbols, then through the `VaricodeDecoder`.
pub enum Psk31Stream {
    Bpsk {
        demod: Bpsk31Demod,
        decider: Bpsk31Decider,
        vdec: VaricodeDecoder,
        fed_up_to: usize,
    },
    Qpsk {
        demod: Qpsk31Demod,
        viterbi: StreamingViterbi,
        vdec: VaricodeDecoder,
        fed_up_to: usize,
    },
}

impl Psk31Stream {
    /// Create a new BPSK31 streaming decoder.
    pub fn new_bpsk(fs: f32, carrier_hz: f32, gain: f32) -> Self {
        Psk31Stream::Bpsk {
            demod: Bpsk31Demod::new(fs, carrier_hz, gain),
            decider: Bpsk31Decider::new(),
            vdec: VaricodeDecoder::new(),
            fed_up_to: 0,
        }
    }

    /// Create a new QPSK31 streaming decoder.
    pub fn new_qpsk(fs: f32, carrier_hz: f32, gain: f32) -> Self {
        Psk31Stream::Qpsk {
            demod: Qpsk31Demod::new(fs, carrier_hz, gain),
            viterbi: StreamingViterbi::new(&DQPSK_EXP),
            vdec: VaricodeDecoder::new(),
            fed_up_to: 0,
        }
    }

    /// Number of IQ samples already processed.
    pub fn fed_up_to(&self) -> usize {
        match self {
            Psk31Stream::Bpsk { fed_up_to, .. } => *fed_up_to,
            Psk31Stream::Qpsk { fed_up_to, .. } => *fed_up_to,
        }
    }

    /// Update the processed-sample counter (e.g. after buffer truncation).
    pub fn set_fed_up_to(&mut self, v: usize) {
        match self {
            Psk31Stream::Bpsk { fed_up_to, .. } => *fed_up_to = v,
            Psk31Stream::Qpsk { fed_up_to, .. } => *fed_up_to = v,
        }
    }

    /// Feed new IQ samples through the demod chain.
    /// Returns any newly decoded printable ASCII characters.
    pub fn feed(&mut self, iq: &[C32]) -> String {
        if iq.is_empty() {
            return String::new();
        }

        match self {
            Psk31Stream::Bpsk {
                demod,
                decider,
                vdec,
                ..
            } => {
                let max_syms = iq.len() / 32 + 4;
                let mut soft = vec![0.0_f32; max_syms];
                let wr = demod.process(iq, &mut soft);
                soft.truncate(wr.out_written);

                let mut bits = vec![0_u8; soft.len()];
                let dr = decider.process(&soft, &mut bits);
                bits.truncate(dr.out_written);

                let mut text = String::new();
                for &b in &bits {
                    vdec.push_bit(b);
                    while let Some(ch) = vdec.pop_char() {
                        if (0x20..0x7f).contains(&ch) {
                            text.push(ch as char);
                        }
                    }
                }
                text
            }
            Psk31Stream::Qpsk {
                demod,
                viterbi,
                vdec,
                ..
            } => {
                let max_soft = iq.len() / 32 + 8;
                let mut soft = vec![0.0_f32; max_soft];
                let wr = demod.process(iq, &mut soft);
                soft.truncate(wr.out_written);

                let mut text = String::new();
                let n_syms = soft.len() / 2;
                for i in 0..n_syms {
                    let d_re = soft[i * 2];
                    let d_im = soft[i * 2 + 1];
                    // Skip near-zero symbols (silence/startup).
                    if d_re * d_re + d_im * d_im < 0.01 {
                        continue;
                    }

                    if let Some(b) = viterbi.feed_symbol(d_re, d_im) {
                        vdec.push_bit(b);
                        while let Some(ch) = vdec.pop_char() {
                            if (0x20..0x7f).contains(&ch) {
                                text.push(ch as char);
                            }
                        }
                    }
                }
                text
            }
        }
    }

    /// Flush the decoder to emit trailing characters.
    pub fn flush(&mut self) -> String {
        match self {
            Psk31Stream::Bpsk { vdec, .. } => {
                vdec.push_bit(0);
                vdec.push_bit(0);
                let mut text = String::new();
                while let Some(ch) = vdec.pop_char() {
                    if (0x20..0x7f).contains(&ch) {
                        text.push(ch as char);
                    }
                }
                text
            }
            Psk31Stream::Qpsk { viterbi, vdec, .. } => {
                let mut text = String::new();
                for b in viterbi.flush() {
                    vdec.push_bit(b);
                    while let Some(ch) = vdec.pop_char() {
                        if (0x20..0x7f).contains(&ch) {
                            text.push(ch as char);
                        }
                    }
                }
                vdec.push_bit(0);
                vdec.push_bit(0);
                while let Some(ch) = vdec.pop_char() {
                    if (0x20..0x7f).contains(&ch) {
                        text.push(ch as char);
                    }
                }
                text
            }
        }
    }
}
