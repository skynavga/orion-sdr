// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/fec/ldpc_codes.rs
//
// A small, self-contained family of binary LDPC codes for the inner stage of
// the concatenated COFDM FEC. Unlike the FT8 LDPC in `codec/ldpc.rs` — whose
// parity/generator tables are hardcoded to the single (174,91) code — these
// codes are *parameterized*: the sparse parity-check matrix H is generated
// deterministically by code at construction, and both encoder and decoder are
// driven from that runtime H.
//
// Construction (systematic, lower-triangular parity — an "IRA"/staircase
// style):
//
//   H = [ A | T ]
//
// where the message occupies the first K columns (block A, sparse and
// deterministic) and the M = N − K parity columns form a lower-bidiagonal
// "staircase" T:
//
//   T[i][i] = 1, T[i][i-1] = 1  (i > 0)
//
// This makes the code systematic and gives an O(M) direct encoder: parity bit
// p_i = (row-i parity of A·message) XOR p_(i-1), so no Gaussian elimination is
// needed and a valid systematic generator always exists. The A block is filled
// with a fixed per-column weight at deterministic (seeded) row positions,
// yielding a regular column weight in the message part — a genuine, decodable
// LDPC structure.
//
// The decoder is the standard sum-product / belief-propagation algorithm,
// reusing the fast tanh/atanh rational approximations and best-snapshot
// tracking from `codec/ldpc.rs`, but driven from this code's sparse adjacency
// (check→bit and bit→check incidence lists) built once from H, rather than the
// FT8 hardcoded NM/MN tables.
//
// LLR convention: positive ⇒ bit more likely 0 (matches `OfdmSoftDemod` and
// `codec::ldpc::ldpc_decode_soft`).

/// Selects one of the fixed-family LDPC code points. Each maps to a
/// deterministic (N, K) with a constructed sparse parity-check matrix.
///
/// The block lengths/rates here are `orion-sdr`'s own constructive codes (not a
/// transcribed standard); see the plan's follow-on note for named-standard
/// code points and runtime matrix ingestion, which are additive extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdpcCode {
    /// Rate 1/2: N = 512, K = 256.
    N512R12,
    /// Rate 2/3: N = 576, K = 384.
    N576R23,
    /// Rate 3/4: N = 512, K = 384.
    N512R34,
}

impl LdpcCode {
    /// Codeword length in bits.
    pub fn n(self) -> usize {
        match self {
            LdpcCode::N512R12 => 512,
            LdpcCode::N576R23 => 576,
            LdpcCode::N512R34 => 512,
        }
    }

    /// Information length in bits.
    pub fn k(self) -> usize {
        match self {
            LdpcCode::N512R12 => 256,
            LdpcCode::N576R23 => 384,
            LdpcCode::N512R34 => 384,
        }
    }

    /// Number of parity bits (`N − K`).
    pub fn m(self) -> usize {
        self.n() - self.k()
    }

    /// Column weight of the message part of H (rows tapped per message column).
    fn col_weight(self) -> usize {
        3
    }
}

/// A constructed LDPC code: sparse parity-check incidence plus the dimensions
/// needed to encode and decode.
#[derive(Debug, Clone)]
pub struct Ldpc {
    code: LdpcCode,
    n: usize,
    k: usize,
    m: usize,
    /// For each of the K message columns, the list of parity-check rows it
    /// participates in (into the A block). Length K.
    msg_col_rows: Vec<Vec<usize>>,
    /// check → bit incidence over the full N columns (message A-block bits plus
    /// the two staircase parity bits per row). Length M.
    check_bits: Vec<Vec<usize>>,
    /// bit → check incidence over all M rows. Length N.
    bit_checks: Vec<Vec<usize>>,
}

impl Ldpc {
    /// Builds the code selected by `code`.
    pub fn new(code: LdpcCode) -> Self {
        let n = code.n();
        let k = code.k();
        let m = code.m();
        assert!(m >= 1 && k >= 1 && n == k + m);

        // Deterministic sparse A block: each message column taps `col_weight`
        // distinct parity rows. To keep belief-propagation well-behaved we
        // enforce two properties as the block is filled:
        //   • row-degree balance — prefer the least-loaded rows, so no check
        //     node is over-connected;
        //   • no A-block 4-cycles — reject any row that would make two message
        //     columns share the same *pair* of rows, the dominant cause of
        //     sum-product oscillation.
        // Note this eliminates 4-cycles *within the A block only*. The fixed
        // staircase column p_{i-1} occupies rows {i-1, i}, so an A-column that
        // taps both of those rows still forms a message↔staircase 4-cycle; the
        // assembled H therefore has girth 4, not 6 (a modest error-floor cost,
        // not a correctness issue — the codes show a clean FER waterfall). The
        // guard runs before the staircase edges exist and does not see them.
        // A fixed xorshift only breaks ties, so the same code is reproduced
        // identically on TX and RX with no stored table.
        let cw = code.col_weight();
        let mut msg_col_rows: Vec<Vec<usize>> = Vec::with_capacity(k);
        let mut row_load = vec![0usize; m];
        // Set of unordered row-pairs already used by some column (4-cycle guard).
        let mut used_pairs: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();
        let mut state: u64 = code_seed(code);
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for _col in 0..k {
            let mut rows: Vec<usize> = Vec::with_capacity(cw);
            while rows.len() < cw {
                // Rank candidate rows by current load (ascending), tie-broken by
                // a rotating pseudo-random offset for spread; pick the first
                // that keeps this column distinct and forms no 4-cycle with the
                // rows already chosen for it.
                let offset = (next() % m as u64) as usize;
                let mut best: Option<usize> = None;
                let mut best_load = usize::MAX;
                for step in 0..m {
                    let r = (offset + step) % m;
                    if rows.contains(&r) {
                        continue;
                    }
                    // Would adding r create a 4-cycle with any already-chosen row?
                    let makes_cycle = rows
                        .iter()
                        .any(|&q| used_pairs.contains(&ordered_pair(q, r)));
                    if makes_cycle {
                        continue;
                    }
                    if row_load[r] < best_load {
                        best_load = row_load[r];
                        best = Some(r);
                    }
                }
                match best {
                    Some(r) => rows.push(r),
                    // No cycle-free row available (dense corner) — relax the
                    // girth constraint for this last pick rather than loop
                    // forever, keeping the column weight exact.
                    None => {
                        let r = (0..m)
                            .map(|s| (offset + s) % m)
                            .find(|r| !rows.contains(r))
                            .expect("m > col_weight guarantees a free row");
                        rows.push(r);
                    }
                }
            }
            // Register the new row-pairs and loads.
            for i in 0..rows.len() {
                row_load[rows[i]] += 1;
                for j in (i + 1)..rows.len() {
                    used_pairs.insert(ordered_pair(rows[i], rows[j]));
                }
            }
            rows.sort_unstable();
            msg_col_rows.push(rows);
        }

        // Build the full check→bit and bit→check incidence. Column layout:
        //   [0 .. K)        message bits (A block)
        //   [K .. K+M)      parity bits p_0 .. p_(M-1) (staircase T)
        let mut check_bits: Vec<Vec<usize>> = vec![Vec::new(); m];
        let mut bit_checks: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (col, rows) in msg_col_rows.iter().enumerate() {
            for &r in rows {
                check_bits[r].push(col);
                bit_checks[col].push(r);
            }
        }
        // Staircase parity part: row i touches parity col (K+i), and (K+i-1) for
        // i>0. `i` indexes `check_bits` and derives the parity column K+i into
        // `bit_checks` — a cross-index that an iterator rewrite can't express.
        #[allow(clippy::needless_range_loop)]
        for i in 0..m {
            let pcol = k + i;
            check_bits[i].push(pcol);
            bit_checks[pcol].push(i);
            if i > 0 {
                let prev = k + i - 1;
                check_bits[i].push(prev);
                bit_checks[prev].push(i);
            }
        }

        Self {
            code,
            n,
            k,
            m,
            msg_col_rows,
            check_bits,
            bit_checks,
        }
    }

    pub fn code(&self) -> LdpcCode {
        self.code
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn m(&self) -> usize {
        self.m
    }

    /// Systematically encodes `message` (`K` bits, values in {0,1}) into an
    /// `N`-bit codeword `[message | parity]`.
    ///
    /// Direct staircase encoding: for each parity row i, `p_i = s_i XOR
    /// p_(i-1)`, where `s_i` is the parity of the A·message dot-product for row
    /// i (`p_-1 = 0`).
    pub fn encode(&self, message: &[u8]) -> Vec<u8> {
        assert_eq!(message.len(), self.k, "LDPC message must be exactly K bits");
        let mut cw = vec![0u8; self.n];
        cw[..self.k].copy_from_slice(message);

        // Row sums s_i = XOR of message bits tapped into row i (A block only).
        let mut s = vec![0u8; self.m];
        for (col, rows) in self.msg_col_rows.iter().enumerate() {
            let bit = message[col] & 1;
            if bit != 0 {
                for &r in rows {
                    s[r] ^= 1;
                }
            }
        }

        // Staircase back-substitution.
        let mut prev = 0u8;
        for i in 0..self.m {
            let p = s[i] ^ prev;
            cw[self.k + i] = p;
            prev = p;
        }
        cw
    }

    /// Hard-decision syndrome weight: number of unsatisfied parity checks for
    /// `hard` (0 ⇒ valid codeword). `hard` is `N` bits.
    pub fn syndrome_weight(&self, hard: &[u8]) -> usize {
        let mut unsat = 0;
        for bits in &self.check_bits {
            let mut x = 0u8;
            for &b in bits {
                x ^= hard[b] & 1;
            }
            if x != 0 {
                unsat += 1;
            }
        }
        unsat
    }

    /// Soft-decision sum-product decoding.
    ///
    /// `llr` — `N` channel LLRs (positive ⇒ bit more likely 0).
    /// `max_iter` — maximum belief-propagation iterations.
    /// Returns the recovered `K`-bit message and the residual unsatisfied-check
    /// count (0 ⇒ a valid codeword was reached).
    pub fn decode_soft(&self, llr: &[f32], max_iter: usize) -> (Vec<u8>, usize) {
        assert_eq!(llr.len(), self.n, "LDPC LLR slice must be N long");

        let mut hard = vec![0u8; self.n];
        for (h, &l) in hard.iter_mut().zip(llr) {
            *h = u8::from(l <= 0.0);
        }
        let init_unsat = self.syndrome_weight(&hard);
        if init_unsat == 0 {
            return (hard[..self.k].to_vec(), 0);
        }

        // Edge messages, keyed by (check, bit) via the incidence lists. We store
        // per-check variable→check messages `m[check][idx]` and check→variable
        // messages `e[check][idx]`, where `idx` indexes into check_bits[check].
        let mut msg: Vec<Vec<f32>> = self
            .check_bits
            .iter()
            .map(|bits| bits.iter().map(|&b| llr[b]).collect())
            .collect();
        let mut ext: Vec<Vec<f32>> = self.check_bits.iter().map(|b| vec![0.0; b.len()]).collect();

        let mut min_unsat = init_unsat;
        let mut best = hard.clone();

        for _iter in 0..max_iter {
            // Check-node update (tanh product rule):
            //   ext = 2·atanh(∏_{other bits} tanh(msg/2)).
            // Written without the `tanh(-msg/2)` / `-2·atanh` double-negation
            // form some fixed-degree decoders use: that form's sign is only
            // correct when every check has the same degree parity, whereas this
            // code's checks have mixed degrees (4 and 5).
            for (c, bits) in self.check_bits.iter().enumerate() {
                let deg = bits.len();
                // `i1`/`i2` index the parallel per-edge `msg`/`ext` arrays; the
                // leave-one-out product needs both indices, so this stays a
                // range loop (same pattern as `codec::ldpc`'s BP decoder).
                #[allow(clippy::needless_range_loop)]
                for i1 in 0..deg {
                    let mut prod = 1.0f32;
                    for (i2, _) in bits.iter().enumerate() {
                        if i2 != i1 {
                            prod *= fast_tanh(msg[c][i2] / 2.0);
                        }
                    }
                    // Clamp before `fast_atanh`: `fast_tanh` can overshoot
                    // slightly above 1.0 near its cutoff, so a high-degree
                    // product could exceed 1.0 and cross `fast_atanh`'s pole
                    // (~1.1035), injecting a huge wrong-signed message. The true
                    // tanh product is always within [-1, 1], so this clamp only
                    // removes the approximation's overshoot — harmless for the
                    // current codes (max product ~1.07 < pole) and a hard
                    // safety guard for any denser code added later.
                    ext[c][i1] = 2.0 * fast_atanh(prod.clamp(-1.0, 1.0));
                }
            }

            // Variable-node hard decision from channel LLR + all incoming ext.
            for (bit, checks) in self.bit_checks.iter().enumerate() {
                let mut l = llr[bit];
                for &c in checks {
                    let idx = self.check_bits[c].iter().position(|&b| b == bit).unwrap();
                    l += ext[c][idx];
                }
                hard[bit] = u8::from(l <= 0.0);
            }

            let unsat = self.syndrome_weight(&hard);
            if unsat < min_unsat {
                min_unsat = unsat;
                best.copy_from_slice(&hard);
                if unsat == 0 {
                    break;
                }
            }

            // Variable→check update: message on edge (c, bit) excludes c's own
            // extrinsic contribution.
            for (bit, checks) in self.bit_checks.iter().enumerate() {
                let total: f32 = llr[bit]
                    + checks
                        .iter()
                        .map(|&c| {
                            let idx = self.check_bits[c].iter().position(|&b| b == bit).unwrap();
                            ext[c][idx]
                        })
                        .sum::<f32>();
                for &c in checks {
                    let idx = self.check_bits[c].iter().position(|&b| b == bit).unwrap();
                    msg[c][idx] = total - ext[c][idx];
                }
            }
        }

        (best[..self.k].to_vec(), min_unsat)
    }
}

/// Orders a row pair so `(a, b)` and `(b, a)` hash identically.
#[inline]
fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Fixed xorshift seed per code point, so TX and RX build an identical H
/// without a stored table.
#[inline]
fn code_seed(code: LdpcCode) -> u64 {
    match code {
        LdpcCode::N512R12 => 0x4C44_5043_3531_3200,
        LdpcCode::N576R23 => 0x4C44_5043_3531_3201,
        LdpcCode::N512R34 => 0x4C44_5043_3531_3202,
    }
}

#[inline]
fn fast_tanh(x: f32) -> f32 {
    if x < -4.97 {
        return -1.0;
    }
    if x > 4.97 {
        return 1.0;
    }
    let x2 = x * x;
    let a = x * (945.0 + x2 * (105.0 + x2));
    let b = 945.0 + x2 * (420.0 + x2 * 15.0);
    a / b
}

#[inline]
fn fast_atanh(x: f32) -> f32 {
    let x2 = x * x;
    let a = x * (945.0 + x2 * (-735.0 + x2 * 64.0));
    let b = 945.0 + x2 * (-1050.0 + x2 * 225.0);
    a / b
}
