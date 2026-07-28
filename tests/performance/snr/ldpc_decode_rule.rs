// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// §1.4 coding-gain axis: post-decode BER vs. channel SNR for the LDPC
// `DecodeRule` variants (sum-product vs. min-sum vs. scaled min-sum). The
// companion speed axis is `throughput::fec`'s `throughput_ldpc_decode_rules`.
// Together they quantify the min-sum trade so R8b can decide whether to keep the
// scaffold as an opt-in mode. On-air decode is unaffected — this only measures.
//
// Channel model: standard BPSK-over-AWGN directly on the codeword, isolating the
// decoder's own coding gain from the OFDM PHY. Coded bit b → symbol x = 1−2b
// (±1); received y = x + n, n ~ N(0, σ²); channel LLR L = 2y/σ² (positive ⇒ bit
// 0, matching the decoder's convention). Es/N0 in dB = −10·log10(2σ²) for unit-
// energy antipodal symbols. The coding-gain loss of a rule is read off as the dB
// shift of its BER waterfall relative to sum-product.
//
// Feature-gated (`--features throughput`); always passes — a measurement run.

use orion_sdr::fec::{DecodeRule, Ldpc, LdpcCode};

const TRIALS: usize = 200;
const MAX_ITER: usize = 50;

// Es/N0 points (dB) over the steep part of the waterfall, where the min-sum
// coding-gain loss is resolvable. Below ~ −3 dB all rules floor near the
// uncoded rate; above ~ +2 dB all reach BER 0 for these short codes — the
// discriminating region is the fractional-dB slope between.
const ESNO_DB: &[f32] = &[
    -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0,
];

const RULES: &[(DecodeRule, &str)] = &[
    (DecodeRule::SumProduct, "sum-product"),
    (DecodeRule::MinSum, "min-sum"),
    (DecodeRule::ScaledMinSum(0.75), "scaled-min-sum(0.75)"),
];

/// Deterministic Gaussian sample via the 12-uniform-sum approximation (the same
/// shape `tests/common::add_awgn` uses), variance 1 before scaling.
fn next_gaussian(s: &mut u64) -> f32 {
    let mut sum = 0.0f32;
    for _ in 0..12 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        sum += (*s as f32) / (u64::MAX as f32) - 0.5;
    }
    sum // sum of 12 U(−0.5,0.5) ≈ N(0, 1)
}

/// Mean post-decode information-bit error rate for `code` decoded with `rule` at
/// the given Es/N0, over `TRIALS` random codewords through a BPSK-AWGN channel.
fn mean_ber(code: LdpcCode, rule: DecodeRule, esno_db: f32, seed_base: u64) -> f32 {
    let ldpc = Ldpc::new(code);
    let k = ldpc.k();
    let n = ldpc.n();

    // Es/N0 = 1/(2σ²)  ⇒  σ² = 1/(2·10^(esno/10)).
    let esno = 10f32.powf(esno_db / 10.0);
    let sigma2 = 1.0 / (2.0 * esno);
    let sigma = sigma2.sqrt();

    let mut total_err = 0usize;
    let mut total_bits = 0usize;

    for trial in 0..TRIALS {
        // Distinct, deterministic seed per (trial, Es/N0 point). Scale the dB by
        // 10 first so fractional points (−2.5 vs −2.0) don't collide.
        let esno_key = (esno_db * 10.0) as i64 as u64;
        let mut rng = seed_base
            .wrapping_add(trial as u64)
            .wrapping_add(esno_key.wrapping_mul(0x9E37))
            | 1;

        // Random message → codeword.
        let msg: Vec<u8> = (0..k)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (rng & 1) as u8
            })
            .collect();
        let cw = ldpc.encode(&msg);

        // BPSK over AWGN → channel LLRs. x = 1−2b; y = x + σ·g; L = 2y/σ².
        let llr: Vec<f32> = (0..n)
            .map(|i| {
                let x = 1.0 - 2.0 * cw[i] as f32;
                let y = x + sigma * next_gaussian(&mut rng);
                2.0 * y / sigma2
            })
            .collect();

        let (decoded, _unsat) = ldpc.decode_soft_with(&llr, MAX_ITER, rule);
        total_err += decoded
            .iter()
            .zip(msg.iter())
            .filter(|(a, b)| a != b)
            .count();
        total_bits += k;
    }

    total_err as f32 / total_bits as f32
}

fn run_sweep(code: LdpcCode, code_name: &str, seed_base: u64) {
    println!("\n[LDPC {code_name} decode-rule BER sweep, {TRIALS} trials/point, BPSK-AWGN]");
    print!("{:>10}", "Es/N0(dB)");
    for (_, rule_name) in RULES {
        print!(" {rule_name:>22}");
    }
    println!();
    println!("{}", "-".repeat(10 + RULES.len() * 23));

    for &esno_db in ESNO_DB {
        print!("{esno_db:>10.1}");
        for (rule, _) in RULES {
            let ber = mean_ber(code, *rule, esno_db, seed_base);
            print!(" {ber:>22.6}");
        }
        println!();
    }
    println!(
        "\nRead the coding-gain loss of a min-sum rule as the horizontal (dB) shift\n\
         of its waterfall relative to the sum-product column at matched BER."
    );
    // Always passes — measurement run.
}

#[test]
fn snr_sweep_ldpc_decode_rule_r12() {
    run_sweep(LdpcCode::N512R12, "N512R12", 0x1D9C_0000_0000_0000);
}

#[test]
fn snr_sweep_ldpc_decode_rule_r34() {
    run_sweep(LdpcCode::N512R34, "N512R34", 0x1D9C_0000_0000_0001);
}
