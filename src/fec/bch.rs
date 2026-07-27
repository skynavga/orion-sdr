// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/fec/bch.rs
//
// Binary BCH code over GF(2^8), used as the outer code of the concatenated
// COFDM FEC. A primitive binary BCH code has natural length n0 = 255 (= 2^8 −
// 1); it is defined by the design distance, i.e. the guaranteed number `t` of
// correctable bit errors. The generator polynomial g(x) is the least common
// multiple of the minimal polynomials of the consecutive roots α^1, α^2, …,
// α^(2t), where α is the primitive element of GF(2^8). Its degree equals the
// number of parity bits n0 − k0.
//
// The code can be *shortened* to any n ≤ n0 by prepending (n0 − n) implicit
// zero information bits that are never transmitted; this is how a byte-oriented
// frame chooses a codeword length that fits its payload without needing a
// distinct generator. Shortening preserves the error-correcting power `t`.
//
// Encoding is systematic: parity = message·x^(n−k) mod g(x); the codeword is
// [message | parity]. Decoding is the classic syndrome → Berlekamp–Massey →
// Chien-search hard-decision pipeline over GF(2^8).
//
// Bit order: information and codeword bits are `u8` values in {0,1}, MSB-first
// in time — index 0 is the highest-degree term. This matches the frame layer's
// bit packing.

use super::gf::Gf256;

/// Errors constructing or decoding a [`Bch`] code.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BchError {
    #[error("design t={0} is too large for GF(2^8) BCH (parity would exceed the block)")]
    DesignTooLarge(usize),
    #[error("shortened length n={0} is out of range 1..=255 or leaves no room for parity")]
    BadLength(usize),
    #[error("codeword has {0} residual errors after decoding (uncorrectable)")]
    Uncorrectable(usize),
}

/// A binary BCH code over GF(2^8), correcting up to `t` bit errors, optionally
/// shortened to length `n`.
#[derive(Debug, Clone)]
pub struct Bch {
    gf: Gf256,
    /// Codeword length (≤ 255; equals 255 when unshortened).
    n: usize,
    /// Information-bit length `k = n − parity_bits`.
    k: usize,
    /// Guaranteed correctable errors.
    t: usize,
    /// Generator polynomial coefficients over GF(2), MSB-first (index 0 is the
    /// highest-degree term). Length = parity_bits + 1.
    gen_poly: Vec<u8>,
}

impl Bch {
    /// Constructs a primitive (unshortened, n = 255) binary BCH code correcting
    /// up to `t` errors.
    pub fn new(t: usize) -> Result<Self, BchError> {
        Self::shortened(255, t)
    }

    /// Constructs a BCH code correcting up to `t` errors, shortened to codeword
    /// length `n` (1..=255). The parity length is fixed by `t` (independent of
    /// shortening); `k = n − parity_bits`.
    pub fn shortened(n: usize, t: usize) -> Result<Self, BchError> {
        if n == 0 || n > 255 {
            return Err(BchError::BadLength(n));
        }
        let gf = Gf256::new();
        let gen_poly = build_generator(&gf, t)?;
        let parity_bits = gen_poly.len() - 1;
        if parity_bits >= n {
            return Err(BchError::BadLength(n));
        }
        let k = n - parity_bits;
        Ok(Self {
            gf,
            n,
            k,
            t,
            gen_poly,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn t(&self) -> usize {
        self.t
    }

    /// Number of parity bits (`n − k`).
    pub fn parity_bits(&self) -> usize {
        self.gen_poly.len() - 1
    }

    /// Systematically encodes `message` (`k` bits, values in {0,1}, MSB-first)
    /// into an `n`-bit codeword `[message | parity]`.
    pub fn encode(&self, message: &[u8]) -> Vec<u8> {
        assert_eq!(message.len(), self.k, "BCH message must be exactly k bits");
        let parity_bits = self.parity_bits();

        // Compute remainder of message·x^(parity_bits) divided by g(x) over
        // GF(2), via a shift-register LFSR clocked MSB-first.
        let mut reg = vec![0u8; parity_bits];
        for &bit in message {
            let feedback = bit ^ reg[0];
            // Shift register left by one.
            for i in 0..parity_bits - 1 {
                reg[i] = reg[i + 1] ^ (self.gen_poly[i + 1] & feedback);
            }
            reg[parity_bits - 1] = self.gen_poly[parity_bits] & feedback;
        }

        let mut codeword = Vec::with_capacity(self.n);
        codeword.extend_from_slice(message);
        codeword.extend_from_slice(&reg);
        codeword
    }

    /// Decodes an `n`-bit received word, correcting up to `t` bit errors, and
    /// returns the recovered `k`-bit message. Returns [`BchError::Uncorrectable`]
    /// if more than `t` errors are present (detected as a failed final syndrome
    /// or an inconsistent error locator).
    pub fn decode(&self, received: &[u8]) -> Result<Vec<u8>, BchError> {
        assert_eq!(received.len(), self.n, "BCH word must be exactly n bits");
        let gf = &self.gf;

        // The received word occupies the *high* end of the full length-255
        // code; shortening prepends (255 − n) zero positions. Map a received
        // index `p` (0-based, MSB-first) to the GF power of its locator:
        // position p corresponds to codeword bit of degree (255 − 1 − shift −
        // p) in the unshortened code. We track locators directly as α^exp.
        let shift = 255 - self.n;

        // Syndromes S_j = r(α^j) for j = 1..=2t, where the received bit at
        // index p (MSB-first) has degree d = (n − 1 − p) + 0 in the *shortened*
        // frame; in the full frame its exponent offset is `shift` higher. Since
        // shortening only zeroes leading positions, evaluating r(α^j) over the
        // present bits with degree d works directly (absent positions are 0).
        let two_t = 2 * self.t;
        let mut syndromes = vec![0u8; two_t + 1];
        let mut has_error = false;
        for (j, syn) in syndromes.iter_mut().enumerate().take(two_t + 1).skip(1) {
            let mut acc = 0u8;
            for (p, &bit) in received.iter().enumerate() {
                if bit != 0 {
                    let degree = self.n - 1 - p + shift;
                    acc = gf.add(acc, gf.pow(gf.exp_of(j), degree));
                }
            }
            *syn = acc;
            if acc != 0 {
                has_error = true;
            }
        }

        if !has_error {
            return Ok(received[..self.k].to_vec());
        }

        // Berlekamp–Massey to find the error-locator polynomial σ(x).
        let sigma = berlekamp_massey(gf, &syndromes, self.t);

        // Chien search: the roots of σ are the inverse error locators. For each
        // candidate degree d in 0..255, test σ(α^-d) == 0 → error at that
        // degree. Convert degree back to a received index (if present).
        let mut corrected = received.to_vec();
        let mut n_found = 0usize;
        for d in 0..255usize {
            // Evaluate σ(α^-d).
            let x = gf.exp_of((255 - (d % 255)) % 255); // α^-d
            let mut val = 0u8;
            for (i, &c) in sigma.iter().enumerate() {
                if c != 0 {
                    val = gf.add(val, gf.mul(c, gf.pow(x, i)));
                }
            }
            if val == 0 {
                // Error at degree d. Map to received index p: degree = n-1-p+shift.
                if d >= shift && d <= self.n - 1 + shift {
                    let p = self.n - 1 + shift - d;
                    if p < self.n {
                        corrected[p] ^= 1;
                        n_found += 1;
                    }
                }
            }
        }

        // Verify: recompute syndrome-1 style check by re-evaluating one
        // syndrome on the corrected word. A clean correction zeroes all
        // syndromes; if the locator degree exceeded t or roots were spurious,
        // residual errors remain.
        let residual = self.residual_errors(&corrected);
        if residual != 0 || n_found > self.t {
            return Err(BchError::Uncorrectable(residual.max(n_found)));
        }

        Ok(corrected[..self.k].to_vec())
    }

    /// Number of nonzero syndromes for `word` (0 ⇒ a valid codeword).
    fn residual_errors(&self, word: &[u8]) -> usize {
        let gf = &self.gf;
        let shift = 255 - self.n;
        let mut count = 0;
        for j in 1..=2 * self.t {
            let mut acc = 0u8;
            for (p, &bit) in word.iter().enumerate() {
                if bit != 0 {
                    let degree = self.n - 1 - p + shift;
                    acc = gf.add(acc, gf.pow(gf.exp_of(j), degree));
                }
            }
            if acc != 0 {
                count += 1;
            }
        }
        count
    }
}

/// Builds the BCH generator polynomial for design error-correction `t` over
/// GF(2^8): g(x) = lcm of the minimal polynomials of α^1..α^(2t). Returned
/// MSB-first (index 0 = highest degree) with GF(2) {0,1} coefficients.
fn build_generator(gf: &Gf256, t: usize) -> Result<Vec<u8>, BchError> {
    if t == 0 {
        return Err(BchError::DesignTooLarge(0));
    }

    // Collect the distinct minimal polynomials over the roots α^1..α^(2t).
    // Each root's conjugacy class {α^(i·2^s)} shares one minimal polynomial;
    // track which powers 1..=2t we've already covered.
    let mut covered = vec![false; 2 * t + 1];
    // g starts as the constant polynomial 1 (low-degree-first for easy multiply).
    let mut g_lo = vec![1u8]; // index 0 = constant term

    for i in 1..=2 * t {
        if covered[i] {
            continue;
        }
        // Build the minimal polynomial of α^i from its conjugates
        // α^i, α^(2i), α^(4i), … (mod 255), as a product Π (x − α^(root)).
        let mut roots = Vec::new();
        let mut e = i % 255;
        loop {
            if e >= 1 && e <= 2 * t {
                covered[e] = true;
            }
            roots.push(e);
            e = (e * 2) % 255;
            if e == i % 255 {
                break;
            }
        }
        // min_poly = Π (x − α^root), low-degree-first over GF(2^8); the result
        // has GF(2) coefficients (guaranteed for a conjugacy-class product).
        let mut min_poly = vec![1u8];
        for &r in &roots {
            let alpha_r = gf.exp_of(r);
            min_poly = poly_mul_linear(gf, &min_poly, alpha_r);
        }
        // Multiply into g (both low-degree-first, GF(2) coefficients).
        g_lo = poly_mul(gf, &g_lo, &min_poly);
    }

    if g_lo.len() > 255 {
        return Err(BchError::DesignTooLarge(t));
    }

    // Convert to MSB-first {0,1}. Coefficients are already in GF(2).
    let mut g: Vec<u8> = g_lo.iter().rev().map(|&c| c & 1).collect();
    // Drop any leading zeros (shouldn't be any: top coeff is 1).
    while g.len() > 1 && g[0] == 0 {
        g.remove(0);
    }
    Ok(g)
}

/// Multiply a low-degree-first GF(2^8) polynomial by the linear factor
/// `(x − alpha)` = `(x + alpha)` (subtraction is XOR).
fn poly_mul_linear(gf: &Gf256, p: &[u8], alpha: u8) -> Vec<u8> {
    let mut out = vec![0u8; p.len() + 1];
    for (i, &c) in p.iter().enumerate() {
        // c·x·... shifts up by one.
        out[i + 1] = gf.add(out[i + 1], c);
        // c·alpha stays.
        out[i] = gf.add(out[i], gf.mul(c, alpha));
    }
    out
}

/// Multiply two low-degree-first GF(2^8) polynomials.
fn poly_mul(gf: &Gf256, a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            if bj != 0 {
                out[i + j] = gf.add(out[i + j], gf.mul(ai, bj));
            }
        }
    }
    out
}

/// Berlekamp–Massey over GF(2^8): returns the error-locator polynomial σ(x)
/// (low-degree-first, σ[0] = 1) from syndromes `s[1..=2t]`.
fn berlekamp_massey(gf: &Gf256, s: &[u8], t: usize) -> Vec<u8> {
    let mut sigma = vec![1u8]; // current locator, low-degree-first
    let mut b = vec![1u8]; // last locator before the last length change
    let mut l = 0usize; // current register length
    let mut m = 1usize; // steps since last length change

    for n in 1..=2 * t {
        // Discrepancy δ = s_n + Σ σ_i · s_(n−i).
        let mut delta = s[n];
        for i in 1..=l {
            if i < sigma.len() {
                delta = gf.add(delta, gf.mul(sigma[i], s[n - i]));
            }
        }

        if delta == 0 {
            m += 1;
        } else if 2 * l < n {
            // Length change (Berlekamp–Massey's 2L ≤ n−1 condition).
            let t_sigma = sigma.clone();
            // sigma = sigma − δ · x^m · b.
            let coef = delta; // over GF(2^8) we scale b by δ (b already scaled by 1/prev_delta)
            apply_correction(gf, &mut sigma, &b, coef, m);
            l = n - l;
            b = t_sigma;
            // Rescale b by 1/δ so the next correction is normalized.
            let inv = gf.inv(delta);
            for c in b.iter_mut() {
                *c = gf.mul(*c, inv);
            }
            m = 1;
        } else {
            apply_correction(gf, &mut sigma, &b, delta, m);
            m += 1;
        }
    }

    sigma
}

/// σ ← σ + coef · x^shift · b  (over GF(2^8), low-degree-first).
fn apply_correction(gf: &Gf256, sigma: &mut Vec<u8>, b: &[u8], coef: u8, shift: usize) {
    let needed = b.len() + shift;
    if sigma.len() < needed {
        sigma.resize(needed, 0);
    }
    for (i, &bi) in b.iter().enumerate() {
        if bi != 0 {
            sigma[i + shift] = gf.add(sigma[i + shift], gf.mul(coef, bi));
        }
    }
}
