// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/fec/reed_solomon.rs
//
// Reed–Solomon block code over GF(2^8), the byte-domain outer code of the
// concatenated COFDM FEC. Symbols are bytes; a code with `n_parity = 2t`
// parity bytes corrects up to `t` symbol errors. The canonical instance is
// DVB-T's RS(204,188), a shortened RS(255,239) with t = 8.
//
// Unlike the binary BCH code in `bch.rs` — where a located error is always a
// bit flip (magnitude 1) — Reed–Solomon errors have arbitrary byte magnitudes,
// so decoding needs Forney's algorithm to compute error *values* in addition
// to their locations:
//
//   syndromes → Berlekamp–Massey (error-locator σ) → Chien search (locations)
//             → Forney (magnitudes via the error-evaluator Ω)
//
// Conventions: first consecutive root FCR = 0, so the generator polynomial is
// g(x) = Π_{i=0}^{2t-1} (x − α^i). Codeword byte order is MSB-first in time:
// index 0 is the highest-degree symbol. Shortening prepends implicit zero
// information symbols that are never transmitted.

use super::gf::Gf256;

/// Errors constructing or decoding a Reed–Solomon code.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RsError {
    #[error("code length n={0} out of range 1..=255 or too short for {1} parity symbols")]
    BadLength(usize, usize),
    #[error("codeword is uncorrectable ({0} residual symbol errors)")]
    Uncorrectable(usize),
}

/// A Reed–Solomon code over GF(2^8): `n` codeword bytes, `k` information bytes,
/// `2t = n − k` parity bytes correcting up to `t` symbol errors.
#[derive(Debug, Clone)]
pub struct ReedSolomon {
    gf: &'static Gf256,
    n: usize,
    k: usize,
    n_parity: usize,
    /// Generator polynomial, low-degree-first, GF(2^8) coefficients.
    gen_poly: Vec<u8>,
}

impl ReedSolomon {
    /// Constructs an RS code with `n` codeword bytes and `n_parity = 2t` parity
    /// bytes (so `k = n − n_parity`, correcting `t = n_parity/2` errors). `n`
    /// must be ≤ 255 and leave room for the parity.
    pub fn new(n: usize, n_parity: usize) -> Result<Self, RsError> {
        if n == 0 || n > 255 || n_parity >= n {
            return Err(RsError::BadLength(n, n_parity));
        }
        let gf = Gf256::shared();
        let gen_poly = build_generator(gf, n_parity);
        Ok(Self {
            gf,
            n,
            k: n - n_parity,
            n_parity,
            gen_poly,
        })
    }

    /// The DVB-T outer code: RS(204, 188), t = 8 (shortened RS(255, 239)).
    pub fn dvb() -> Self {
        Self::new(204, 16).expect("valid DVB RS(204,188)")
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// Number of correctable symbol errors, `t = n_parity / 2`.
    pub fn t(&self) -> usize {
        self.n_parity / 2
    }

    pub fn parity_bytes(&self) -> usize {
        self.n_parity
    }

    /// Systematically encodes `message` (`k` bytes) into an `n`-byte codeword
    /// `[message | parity]`.
    pub fn encode(&self, message: &[u8]) -> Vec<u8> {
        assert_eq!(message.len(), self.k, "RS message must be exactly k bytes");
        let gf = self.gf;
        // parity = remainder of message·x^{n_parity} / g(x), via LFSR.
        let mut reg = vec![0u8; self.n_parity];
        for &m in message {
            let feedback = gf.add(m, reg[0]);
            for i in 0..self.n_parity - 1 {
                // reg shifts left; add feedback·gen[deg = n_parity-1-i].
                reg[i] = gf.add(
                    reg[i + 1],
                    gf.mul(feedback, self.gen_poly[self.n_parity - 1 - i]),
                );
            }
            reg[self.n_parity - 1] = gf.mul(feedback, self.gen_poly[0]);
        }
        let mut cw = Vec::with_capacity(self.n);
        cw.extend_from_slice(message);
        cw.extend_from_slice(&reg);
        cw
    }

    /// Decodes an `n`-byte received word, correcting up to `t` symbol errors,
    /// and returns the recovered `k`-byte message.
    pub fn decode(&self, received: &[u8]) -> Result<Vec<u8>, RsError> {
        assert_eq!(received.len(), self.n, "RS word must be exactly n bytes");
        let gf = self.gf;
        let shift = 255 - self.n; // shortening: leading zero positions

        // Position p (0-based, MSB-first) has code degree d = (n-1-p) + shift.
        let degree = |p: usize| self.n - 1 - p + shift;

        // Syndromes S_j = r(α^j) for j = 0..2t-1 (FCR = 0).
        let mut syndromes = vec![0u8; self.n_parity];
        let mut nonzero = false;
        for (j, syn) in syndromes.iter_mut().enumerate() {
            let mut acc = 0u8;
            for (p, &b) in received.iter().enumerate() {
                if b != 0 {
                    acc = gf.add(acc, gf.mul(b, gf.pow(gf.exp_of(j), degree(p))));
                }
            }
            *syn = acc;
            if acc != 0 {
                nonzero = true;
            }
        }
        if !nonzero {
            return Ok(received[..self.k].to_vec());
        }

        // Berlekamp–Massey → error-locator σ(x) (low-degree-first, σ[0] = 1).
        let sigma = berlekamp_massey(gf, &syndromes, self.t());

        // Chien search: roots of σ are α^{-i} for error positions of code
        // degree i. Collect the error degrees.
        let mut error_degrees = Vec::new();
        for i in 0..255usize {
            let x = gf.exp_of((255 - (i % 255)) % 255); // α^{-i}
            let mut val = 0u8;
            for (deg, &c) in sigma.iter().enumerate() {
                if c != 0 {
                    val = gf.add(val, gf.mul(c, gf.pow(x, deg)));
                }
            }
            if val == 0 {
                error_degrees.push(i);
            }
        }

        // Number of roots must equal the locator degree, else uncorrectable.
        let sigma_deg = sigma.iter().rposition(|&c| c != 0).unwrap_or(0);
        if error_degrees.len() != sigma_deg || sigma_deg > self.t() {
            return Err(RsError::Uncorrectable(sigma_deg));
        }

        // Error evaluator Ω(x) = [S(x)·σ(x)] mod x^{2t}, with S(x) the syndrome
        // polynomial Σ S_j x^j.
        let omega = error_evaluator(gf, &syndromes, &sigma, self.n_parity);
        // σ'(x) — formal derivative (only odd-degree terms survive over GF(2)).
        let sigma_deriv = formal_derivative(&sigma);

        // Forney: magnitude at error position with code degree i is
        //   e = X · Ω(X^{-1}) / σ'(X^{-1}),  X = α^i,  with FCR = 0
        // (the general FCR-b factor X^{1-b} reduces to X for b = 0).
        let mut corrected = received.to_vec();
        for &i in &error_degrees {
            let x = gf.exp_of(i % 255); // α^i
            let x_inv = gf.inv(x);
            let omega_val = poly_eval(gf, &omega, x_inv);
            let deriv_val = poly_eval(gf, &sigma_deriv, x_inv);
            if deriv_val == 0 {
                return Err(RsError::Uncorrectable(error_degrees.len()));
            }
            let magnitude = gf.mul(x, gf.div(omega_val, deriv_val));
            // Map code degree i back to a received index p (if present).
            if i >= shift && i <= self.n - 1 + shift {
                let p = self.n - 1 + shift - i;
                if p < self.n {
                    corrected[p] = gf.add(corrected[p], magnitude);
                }
            }
        }

        // Verify: a clean correction zeroes all syndromes.
        if self.residual_errors(&corrected) != 0 {
            return Err(RsError::Uncorrectable(error_degrees.len()));
        }
        Ok(corrected[..self.k].to_vec())
    }

    fn residual_errors(&self, word: &[u8]) -> usize {
        let gf = self.gf;
        let shift = 255 - self.n;
        let mut count = 0;
        for j in 0..self.n_parity {
            let mut acc = 0u8;
            for (p, &b) in word.iter().enumerate() {
                if b != 0 {
                    let deg = self.n - 1 - p + shift;
                    acc = gf.add(acc, gf.mul(b, gf.pow(gf.exp_of(j), deg)));
                }
            }
            if acc != 0 {
                count += 1;
            }
        }
        count
    }
}

/// g(x) = Π_{i=0}^{n_parity-1} (x − α^i), low-degree-first.
fn build_generator(gf: &Gf256, n_parity: usize) -> Vec<u8> {
    let mut g = vec![1u8];
    for i in 0..n_parity {
        let root = gf.exp_of(i);
        g = poly_mul_linear(gf, &g, root);
    }
    g
}

/// Multiply low-degree-first polynomial `p` by `(x − alpha)` over GF(2^8).
fn poly_mul_linear(gf: &Gf256, p: &[u8], alpha: u8) -> Vec<u8> {
    let mut out = vec![0u8; p.len() + 1];
    for (i, &c) in p.iter().enumerate() {
        out[i + 1] = gf.add(out[i + 1], c); // c·x
        out[i] = gf.add(out[i], gf.mul(c, alpha)); // c·alpha
    }
    out
}

/// Evaluate a low-degree-first polynomial at `x` (Horner).
fn poly_eval(gf: &Gf256, p: &[u8], x: u8) -> u8 {
    let mut acc = 0u8;
    for &c in p.iter().rev() {
        acc = gf.add(gf.mul(acc, x), c);
    }
    acc
}

/// Berlekamp–Massey over GF(2^8): error-locator σ(x) from syndromes
/// `s[0..2t-1]` (FCR = 0). Low-degree-first, σ[0] = 1.
fn berlekamp_massey(gf: &Gf256, s: &[u8], t: usize) -> Vec<u8> {
    let mut sigma = vec![1u8];
    let mut b = vec![1u8];
    let mut l = 0usize;
    let mut m = 1usize;

    for n in 0..2 * t {
        // Discrepancy δ = s_n + Σ_{i=1}^{L} σ_i · s_{n-i}.
        let mut delta = s[n];
        for i in 1..=l {
            if i < sigma.len() {
                delta = gf.add(delta, gf.mul(sigma[i], s[n - i]));
            }
        }
        if delta == 0 {
            m += 1;
        } else if 2 * l <= n {
            let t_sigma = sigma.clone();
            apply_correction(gf, &mut sigma, &b, delta, m);
            l = n + 1 - l;
            b = t_sigma;
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

/// σ ← σ + coef · x^shift · b (low-degree-first, over GF(2^8)).
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

/// Error-evaluator Ω(x) = [S(x)·σ(x)] mod x^{n_parity}, where the syndrome
/// polynomial is S(x) = Σ s_j x^j.
fn error_evaluator(gf: &Gf256, s: &[u8], sigma: &[u8], n_parity: usize) -> Vec<u8> {
    let mut omega = vec![0u8; n_parity];
    for (i, &si) in s.iter().enumerate() {
        if si == 0 {
            continue;
        }
        for (j, &sj) in sigma.iter().enumerate() {
            if sj != 0 && i + j < n_parity {
                omega[i + j] = gf.add(omega[i + j], gf.mul(si, sj));
            }
        }
    }
    omega
}

/// Formal derivative over GF(2): only odd-degree terms survive (coefficient of
/// x^{k} moves to x^{k-1} for odd k; even-degree terms vanish since 2·c = 0).
fn formal_derivative(p: &[u8]) -> Vec<u8> {
    if p.len() <= 1 {
        return vec![0u8];
    }
    let mut d = vec![0u8; p.len() - 1];
    for (k, &c) in p.iter().enumerate().skip(1) {
        if k % 2 == 1 {
            d[k - 1] = c;
        }
    }
    d
}
