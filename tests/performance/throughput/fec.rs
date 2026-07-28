// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// Throughput benchmarks for the COFDM FEC / interleave / scrambler blocks.
//
// These measure the channel-coding kernels in isolation (per §1 of the
// ofdm-optimize-cofdm plan): each block's forward (Tx) and inverse (Rx)
// direction separately (§1.1), the paired forward→inverse roundtrips with a
// baked-in correctness assertion (§1.2), and the per-frame code-object
// construction cost the caching work targets (§1.3).
//
// Msps convention: unlike the sample-domain modem benchmarks, "samples" here is
// the number of **information bits** processed per pass (before coding on the Tx
// side, after decoding on the Rx side). This keeps the figure comparable across
// code rates — a rate-1/2 and a rate-3/4 code doing the same info-bit work
// report on the same axis regardless of how many coded bits they emit.
//
// Fixtures build their code object ONCE, outside the measured closure (we are
// measuring the kernel, not construction — §1.3 measures construction on
// purpose). Decode fixtures feed a realistic error load so the decoder does
// representative work (BP iterates, BM/Chien run) rather than the degenerate
// zero-error fast path.

use super::{measure_throughput, minsps_from_env};
use num_complex::Complex32 as C32;
use orion_sdr::demodulate::demodulate_frame;
use orion_sdr::fec::{
    Bch, BlockInterleaver, DecodeRule, FrameMetadata, FramePacket, InnerFec, InterleaverKind, Ldpc,
    LdpcCode, OuterFec, PnScrambler, PunctureRate, ReedSolomon, conv_encode_punctured,
    punctured_coded_len, viterbi_decode_soft,
};
use orion_sdr::modulate::ofdm_frame::interleave_bits;
use orion_sdr::modulate::{
    CodecCache, ConstellationOrder, Mcs, McsTable, OfdmConfig, OfdmFrameMod,
};
use orion_sdr::multicarrier::CarrierPlan;
use orion_sdr::sync::OfdmPreamble;
use std::hint::black_box;

// Small deterministic xorshift for reproducible messages / error patterns
// (mirrors the one in tests/unit/fec.rs so fixtures are reproducible).
fn xorshift(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

fn random_bits(n: usize, seed: u64) -> Vec<u8> {
    let mut r = xorshift(seed);
    (0..n).map(|_| (r() & 1) as u8).collect()
}

fn random_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut r = xorshift(seed);
    (0..n).map(|_| (r() & 0xFF) as u8).collect()
}

/// Maps hard bits to strong LLRs (bit 0 → +mag, bit 1 → −mag), the +⇒bit-0
/// convention `OfdmSoftDemod`/`Ldpc`/`viterbi_decode_soft` all use.
fn bits_to_llrs(bits: &[u8], mag: f32) -> Vec<f32> {
    bits.iter()
        .map(|&b| if b == 0 { mag } else { -mag })
        .collect()
}

fn report_and_assert(label: &str, msps: f32, dt: f64, floor: f32) {
    println!("[{}] {:.2} Msps in {:.3}s", label, msps, dt);
    let min_msps = minsps_from_env(floor);
    assert!(
        msps >= min_msps,
        "{} throughput {:.2} Msps < min {:.2} Msps",
        label,
        msps,
        min_msps
    );
}

const REPEATS: usize = 200;

// ── §1.1 LDPC (per code point) ──────────────────────────────────────────────

const LDPC_CODES: [(LdpcCode, &str); 3] = [
    (LdpcCode::N512R12, "N512R12"),
    (LdpcCode::N576R23, "N576R23"),
    (LdpcCode::N512R34, "N512R34"),
];

fn throughput_ldpc_encode(code: LdpcCode, name: &str) {
    let ldpc = Ldpc::new(code);
    let msg = random_bits(ldpc.k(), 0xE0C0 ^ code.n() as u64);
    let info_bits = ldpc.k();

    let (msps, dt) = measure_throughput(
        || {
            let cw = ldpc.encode(&msg);
            black_box(cw[0]);
            info_bits
        },
        info_bits,
        REPEATS,
    );
    report_and_assert(&format!("LDPC-Encode {name}"), msps, dt, 0.2);
}

fn throughput_ldpc_decode(code: LdpcCode, name: &str) {
    let ldpc = Ldpc::new(code);
    let msg = random_bits(ldpc.k(), 0xDEC0 ^ code.n() as u64);
    let cw = ldpc.encode(&msg);
    // A modest soft-error load so belief propagation runs a realistic number of
    // iterations (not the zero-error early exit, not the max-iter no-converge
    // ceiling).
    let mut llr = bits_to_llrs(&cw, 6.0);
    let mut e = xorshift(0xE770 ^ code.n() as u64);
    for _ in 0..4 {
        let pos = (e() as usize) % ldpc.n();
        llr[pos] = -llr[pos] * 0.5;
    }
    let info_bits = ldpc.k();

    let (msps, dt) = measure_throughput(
        || {
            let (decoded, _unsat) = ldpc.decode_soft(&llr, 50);
            black_box(decoded[0]);
            info_bits
        },
        info_bits,
        REPEATS,
    );
    report_and_assert(&format!("LDPC-Decode {name}"), msps, dt, 0.05);
}

#[test]
fn throughput_ldpc_encode_all() {
    for (code, name) in LDPC_CODES {
        throughput_ldpc_encode(code, name);
    }
}

#[test]
fn throughput_ldpc_decode_all() {
    for (code, name) in LDPC_CODES {
        throughput_ldpc_decode(code, name);
    }
}

// §1.4 speed axis: decode throughput per `DecodeRule`, so the min-sum
// investigation can weigh its speed gain against the coding-gain loss measured
// by the `snr::ldpc_decode_rule` sweep. Same error-injected fixture as
// `throughput_ldpc_decode` above.
#[test]
fn throughput_ldpc_decode_rules() {
    let rules = [
        (DecodeRule::SumProduct, "sum-product"),
        (DecodeRule::MinSum, "min-sum"),
        (DecodeRule::ScaledMinSum(0.75), "scaled-min-sum(0.75)"),
    ];
    for (code, name) in LDPC_CODES {
        let ldpc = Ldpc::new(code);
        let msg = random_bits(ldpc.k(), 0xDEC0 ^ code.n() as u64);
        let cw = ldpc.encode(&msg);
        let mut llr = bits_to_llrs(&cw, 6.0);
        let mut e = xorshift(0xE770 ^ code.n() as u64);
        for _ in 0..4 {
            let pos = (e() as usize) % ldpc.n();
            llr[pos] = -llr[pos] * 0.5;
        }
        let info_bits = ldpc.k();
        for (rule, rule_name) in rules {
            let (msps, dt) = measure_throughput(
                || {
                    let (decoded, _unsat) = ldpc.decode_soft_with(&llr, 50, rule);
                    black_box(decoded[0]);
                    info_bits
                },
                info_bits,
                REPEATS,
            );
            report_and_assert(&format!("LDPC-Decode {name} [{rule_name}]"), msps, dt, 0.05);
        }
    }
}

// ── §1.1 Convolutional (per puncture rate) ──────────────────────────────────

const PUNCTURE_RATES: [(PunctureRate, &str); 5] = [
    (PunctureRate::R1_2, "R1_2"),
    (PunctureRate::R2_3, "R2_3"),
    (PunctureRate::R3_4, "R3_4"),
    (PunctureRate::R5_6, "R5_6"),
    (PunctureRate::R7_8, "R7_8"),
];

const CONV_INFO_BITS: usize = 512;

fn throughput_conv_encode(rate: PunctureRate, name: &str) {
    let info = random_bits(CONV_INFO_BITS, 0xC0DE ^ name.len() as u64);
    let (msps, dt) = measure_throughput(
        || {
            let coded = conv_encode_punctured(&info, rate);
            black_box(coded[0]);
            CONV_INFO_BITS
        },
        CONV_INFO_BITS,
        REPEATS,
    );
    report_and_assert(&format!("Conv-Encode {name}"), msps, dt, 0.2);
}

fn throughput_conv_decode(rate: PunctureRate, name: &str) {
    let info = random_bits(CONV_INFO_BITS, 0xC0DE ^ name.len() as u64);
    let coded = conv_encode_punctured(&info, rate);
    let llrs = bits_to_llrs(&coded, 4.0);
    let (msps, dt) = measure_throughput(
        || {
            let decoded = viterbi_decode_soft(&llrs, CONV_INFO_BITS, rate);
            black_box(decoded[0]);
            CONV_INFO_BITS
        },
        CONV_INFO_BITS,
        REPEATS,
    );
    report_and_assert(&format!("Conv-Decode {name}"), msps, dt, 0.05);
}

#[test]
fn throughput_conv_encode_all() {
    for (rate, name) in PUNCTURE_RATES {
        throughput_conv_encode(rate, name);
    }
}

#[test]
fn throughput_conv_decode_all() {
    for (rate, name) in PUNCTURE_RATES {
        throughput_conv_decode(rate, name);
    }
}

// ── §1.1 BCH (outer, hard-decision) ─────────────────────────────────────────

const BCH_T: usize = 8;

fn throughput_bch_encode() {
    let code = Bch::new(BCH_T).expect("valid BCH t");
    let msg = random_bits(code.k(), 0xBC40);
    let info_bits = code.k();
    let (msps, dt) = measure_throughput(
        || {
            let cw = code.encode(&msg);
            black_box(cw[0]);
            info_bits
        },
        info_bits,
        REPEATS,
    );
    report_and_assert("BCH-Encode t=8", msps, dt, 0.2);
}

fn throughput_bch_decode() {
    let code = Bch::new(BCH_T).expect("valid BCH t");
    let msg = random_bits(code.k(), 0xBC40);
    let mut received = code.encode(&msg);
    // Inject exactly t bit errors — the worst-case BM/Chien workload.
    let mut e = xorshift(0xBCE7);
    for _ in 0..BCH_T {
        let pos = (e() as usize) % code.n();
        received[pos] ^= 1;
    }
    let info_bits = code.k();
    let (msps, dt) = measure_throughput(
        || {
            let decoded = code.decode(&received).expect("t errors are correctable");
            black_box(decoded[0]);
            info_bits
        },
        info_bits,
        REPEATS,
    );
    report_and_assert("BCH-Decode t=8", msps, dt, 0.05);
}

#[test]
fn throughput_bch_encode_test() {
    throughput_bch_encode();
}

#[test]
fn throughput_bch_decode_test() {
    throughput_bch_decode();
}

// ── §1.1 Reed–Solomon (outer, byte-domain) ──────────────────────────────────

fn throughput_rs_encode(rs: &ReedSolomon, label: &str) {
    let msg = random_bytes(rs.k(), 0x2504);
    let info_bits = rs.k() * 8;
    let (msps, dt) = measure_throughput(
        || {
            let cw = rs.encode(&msg);
            black_box(cw[0]);
            info_bits
        },
        info_bits,
        REPEATS,
    );
    report_and_assert(&format!("RS-Encode {label}"), msps, dt, 0.2);
}

fn throughput_rs_decode(rs: &ReedSolomon, label: &str) {
    let msg = random_bytes(rs.k(), 0x2504);
    let mut received = rs.encode(&msg);
    // Inject exactly t symbol errors (arbitrary nonzero magnitude) — worst case.
    let mut e = xorshift(0x25E7);
    for _ in 0..rs.t() {
        let pos = (e() as usize) % rs.n();
        let mag = ((e() & 0xFF) as u8) | 1; // nonzero error value
        received[pos] ^= mag;
    }
    let info_bits = rs.k() * 8;
    let (msps, dt) = measure_throughput(
        || {
            let decoded = rs.decode(&received).expect("t errors are correctable");
            black_box(decoded[0]);
            info_bits
        },
        info_bits,
        REPEATS,
    );
    report_and_assert(&format!("RS-Decode {label}"), msps, dt, 0.05);
}

#[test]
fn throughput_rs_encode_all() {
    throughput_rs_encode(&ReedSolomon::dvb(), "204,188");
    // A shortened variant (the Conv+RS concatenation's RS(60,52), t=4).
    throughput_rs_encode(&ReedSolomon::new(60, 8).expect("valid RS"), "60,52");
}

#[test]
fn throughput_rs_decode_all() {
    throughput_rs_decode(&ReedSolomon::dvb(), "204,188");
    throughput_rs_decode(&ReedSolomon::new(60, 8).expect("valid RS"), "60,52");
}

// ── §1.1 Interleaver (both domains) ─────────────────────────────────────────

// A representative on-air block size (rows × cols).
const IL_ROWS: usize = 32;
const IL_COLS: usize = 32;

fn throughput_interleaver_u8(inverse: bool) {
    let il = BlockInterleaver::new(IL_ROWS, IL_COLS);
    let n = il.block_len();
    let input: Vec<u8> = random_bits(n, 0x171E);
    let mut output = vec![0u8; n];
    let label = if inverse {
        "Interleaver-Deinterleave u8"
    } else {
        "Interleaver-Interleave u8"
    };
    let (msps, dt) = measure_throughput(
        || {
            if inverse {
                il.deinterleave(&input, &mut output);
            } else {
                il.interleave(&input, &mut output);
            }
            black_box(output[0]);
            n
        },
        n,
        REPEATS,
    );
    report_and_assert(label, msps, dt, 1.0);
}

fn throughput_interleaver_f32(inverse: bool) {
    let il = BlockInterleaver::new(IL_ROWS, IL_COLS);
    let n = il.block_len();
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let mut output = vec![0.0f32; n];
    let label = if inverse {
        "Interleaver-Deinterleave f32"
    } else {
        "Interleaver-Interleave f32"
    };
    let (msps, dt) = measure_throughput(
        || {
            if inverse {
                il.deinterleave(&input, &mut output);
            } else {
                il.interleave(&input, &mut output);
            }
            black_box(output[0]);
            n
        },
        n,
        REPEATS,
    );
    report_and_assert(label, msps, dt, 1.0);
}

#[test]
fn throughput_interleaver_all() {
    throughput_interleaver_u8(false);
    throughput_interleaver_u8(true);
    throughput_interleaver_f32(false);
    throughput_interleaver_f32(true);
}

// The chain-driver `interleave_bits` (fragments into blocks, pads, permutes) —
// the path R7 hoists per-chunk allocations out of. Unlike the kernel benchmarks
// above, this exercises the multi-chunk driver a configured link runs; it is the
// regression guard for that hoist (the default MCS uses no interleaver, so the
// frame-chain benchmark never reaches this path).
#[test]
fn throughput_interleave_bits_chain() {
    let il = InterleaverKind::Block {
        rows: IL_ROWS,
        cols: IL_COLS,
    };
    // Several blocks' worth of coded bits, with a partial final block to exercise
    // padding.
    let n_bits = IL_ROWS * IL_COLS * 8 + 37;
    let bits = random_bits(n_bits, 0x171B);
    let (msps, dt) = measure_throughput(
        || {
            let out = interleave_bits(il, &bits);
            black_box(out[0]);
            n_bits
        },
        n_bits,
        REPEATS,
    );
    report_and_assert("Interleave-Bits chain (8+ blocks)", msps, dt, 1.0);
}

// ── §1.1 Scrambler (self-inverse; single direction, three widths) ───────────

// A representative coded-frame byte length.
const SCRAMBLER_BYTES: usize = 512;

fn throughput_scrambler(sc: PnScrambler, label: &str) {
    let mut data = random_bytes(SCRAMBLER_BYTES, 0x5C4A);
    let n_bits = SCRAMBLER_BYTES * 8;
    let (msps, dt) = measure_throughput(
        || {
            sc.scramble(&mut data);
            black_box(data[0]);
            n_bits
        },
        n_bits,
        REPEATS,
    );
    report_and_assert(&format!("Scrambler-{label}"), msps, dt, 1.0);
}

#[test]
fn throughput_scrambler_all() {
    // The three (taps, width, seed) triples the unit tests exercise.
    throughput_scrambler(PnScrambler::new(0b1001, 7, 0x7F), "w7");
    throughput_scrambler(
        PnScrambler::new(0b11 << 13, 15, 0b100_1010_1000_0000),
        "w15",
    );
    throughput_scrambler(PnScrambler::new(0x8020_0003, 32, 0x1234_5678), "w32");
}

// ── §1.2 Paired roundtrips (forward → inverse), correctness asserted ────────

fn roundtrip_ldpc(code: LdpcCode, name: &str) {
    let ldpc = Ldpc::new(code);
    let msg = random_bits(ldpc.k(), 0x2D7 ^ code.n() as u64);
    let info_bits = ldpc.k();
    let (msps, dt) = measure_throughput(
        || {
            let cw = ldpc.encode(&msg);
            let llr = bits_to_llrs(&cw, 8.0);
            let (decoded, unsat) = ldpc.decode_soft(&llr, 50);
            assert_eq!(unsat, 0, "LDPC {name}: clean roundtrip must converge");
            assert_eq!(decoded, msg, "LDPC {name}: message recovered");
            black_box(decoded[0]);
            info_bits
        },
        info_bits,
        REPEATS,
    );
    report_and_assert(&format!("Roundtrip-LDPC {name}"), msps, dt, 0.05);
}

fn roundtrip_conv(rate: PunctureRate, name: &str) {
    let info = random_bits(CONV_INFO_BITS, 0x2C07 ^ name.len() as u64);
    let (msps, dt) = measure_throughput(
        || {
            let coded = conv_encode_punctured(&info, rate);
            debug_assert_eq!(coded.len(), punctured_coded_len(info.len(), rate));
            let llrs = bits_to_llrs(&coded, 4.0);
            let decoded = viterbi_decode_soft(&llrs, info.len(), rate);
            assert_eq!(decoded, info, "Conv {name}: info recovered");
            black_box(decoded[0]);
            CONV_INFO_BITS
        },
        CONV_INFO_BITS,
        REPEATS,
    );
    report_and_assert(&format!("Roundtrip-Conv {name}"), msps, dt, 0.05);
}

fn roundtrip_bch() {
    let code = Bch::new(BCH_T).expect("valid BCH t");
    let msg = random_bits(code.k(), 0x2BC4);
    let info_bits = code.k();
    let mut e = xorshift(0x2BE7);
    let (msps, dt) = measure_throughput(
        || {
            let mut received = code.encode(&msg);
            for _ in 0..BCH_T {
                let pos = (e() as usize) % code.n();
                received[pos] ^= 1;
            }
            let decoded = code.decode(&received).expect("t errors correctable");
            assert_eq!(decoded, msg, "BCH: message recovered through t errors");
            black_box(decoded[0]);
            info_bits
        },
        info_bits,
        REPEATS,
    );
    report_and_assert("Roundtrip-BCH t=8", msps, dt, 0.05);
}

fn roundtrip_rs(rs: &ReedSolomon, label: &str) {
    let msg = random_bytes(rs.k(), 0x2725);
    let info_bits = rs.k() * 8;
    let mut e = xorshift(0x27E7);
    let (msps, dt) = measure_throughput(
        || {
            let mut received = rs.encode(&msg);
            for _ in 0..rs.t() {
                let pos = (e() as usize) % rs.n();
                let mag = ((e() & 0xFF) as u8) | 1;
                received[pos] ^= mag;
            }
            let decoded = rs.decode(&received).expect("t errors correctable");
            assert_eq!(
                decoded, msg,
                "RS {label}: message recovered through t errors"
            );
            black_box(decoded[0]);
            info_bits
        },
        info_bits,
        REPEATS,
    );
    report_and_assert(&format!("Roundtrip-RS {label}"), msps, dt, 0.05);
}

fn roundtrip_interleaver_u8() {
    let il = BlockInterleaver::new(IL_ROWS, IL_COLS);
    let n = il.block_len();
    let input: Vec<u8> = random_bits(n, 0x2171);
    let mut mid = vec![0u8; n];
    let mut back = vec![0u8; n];
    let (msps, dt) = measure_throughput(
        || {
            il.interleave(&input, &mut mid);
            il.deinterleave(&mid, &mut back);
            assert_eq!(back, input, "interleaver u8 roundtrip is identity");
            black_box(back[0]);
            n
        },
        n,
        REPEATS,
    );
    report_and_assert("Roundtrip-Interleaver u8", msps, dt, 1.0);
}

fn roundtrip_interleaver_f32() {
    let il = BlockInterleaver::new(IL_ROWS, IL_COLS);
    let n = il.block_len();
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 2.0).collect();
    let mut mid = vec![0.0f32; n];
    let mut back = vec![0.0f32; n];
    let (msps, dt) = measure_throughput(
        || {
            il.interleave(&input, &mut mid);
            il.deinterleave(&mid, &mut back);
            assert_eq!(back, input, "interleaver f32 roundtrip is identity");
            black_box(back[0]);
            n
        },
        n,
        REPEATS,
    );
    report_and_assert("Roundtrip-Interleaver f32", msps, dt, 1.0);
}

fn roundtrip_scrambler() {
    let sc = PnScrambler::new(0x8020_0003, 32, 0x1234_5678);
    let original = random_bytes(SCRAMBLER_BYTES, 0x25C4);
    let mut data = original.clone();
    let n_bits = SCRAMBLER_BYTES * 8;
    let (msps, dt) = measure_throughput(
        || {
            sc.scramble(&mut data); // scramble
            sc.scramble(&mut data); // descramble (self-inverse)
            assert_eq!(data, original, "scrambler roundtrip is identity");
            black_box(data[0]);
            n_bits
        },
        n_bits,
        REPEATS,
    );
    report_and_assert("Roundtrip-Scrambler w32", msps, dt, 1.0);
}

#[test]
fn roundtrip_ldpc_all() {
    for (code, name) in LDPC_CODES {
        roundtrip_ldpc(code, name);
    }
}

#[test]
fn roundtrip_conv_all() {
    for (rate, name) in PUNCTURE_RATES {
        roundtrip_conv(rate, name);
    }
}

#[test]
fn roundtrip_bch_test() {
    roundtrip_bch();
}

#[test]
fn roundtrip_rs_all() {
    roundtrip_rs(&ReedSolomon::dvb(), "204,188");
    roundtrip_rs(&ReedSolomon::new(60, 8).expect("valid RS"), "60,52");
}

#[test]
fn roundtrip_interleaver_all() {
    roundtrip_interleaver_u8();
    roundtrip_interleaver_f32();
}

#[test]
fn roundtrip_scrambler_test() {
    roundtrip_scrambler();
}

// ── §1.3 Construction-cost microbenchmarks (Finding A baseline) ─────────────
//
// These measure the per-frame code-object construction the caching work (R2/R3)
// targets. They print but assert only a loose floor: the whole point of the
// caching work is to drive these DOWN, so a hard ceiling would be brittle. The
// "Msps" here is a nominal 1-unit-per-construction rate (constructions/µs·1e6);
// it is a relative before/after handle, not a sample-domain figure.

fn measure_construction(label: &str, mut build: impl FnMut()) {
    // One "sample" per construction; report constructions-per-second (as Msps)
    // so the caching commits can show the before/after delta directly.
    let repeats = 2000;
    let (rate, dt) = measure_throughput(
        || {
            build();
            1
        },
        1,
        repeats,
    );
    // rate is in "constructions per µs × 1e6" i.e. Mconstructions/s — tiny for
    // the heavy codes, which is exactly the signal.
    println!(
        "[Construct {}] {:.4} M-constructs/s in {:.3}s",
        label, rate, dt
    );
}

#[test]
fn construction_cost_ldpc() {
    for (code, name) in LDPC_CODES {
        measure_construction(&format!("Ldpc::new {name}"), || {
            black_box(Ldpc::new(black_box(code)));
        });
    }
}

#[test]
fn construction_cost_bch() {
    measure_construction("Bch::new t=8", || {
        black_box(Bch::new(black_box(8)).unwrap());
    });
}

#[test]
fn construction_cost_rs() {
    measure_construction("ReedSolomon::dvb", || {
        black_box(ReedSolomon::dvb());
    });
}

/// Per-frame amortization handle: constructing the LDPC code fresh N times (the
/// current per-frame behavior) vs. building it once and reusing it. R3's caching
/// commit makes the real chain behave like the "reused" line; this benchmark
/// quantifies the gap it closes.
#[test]
fn construction_cost_ldpc_per_frame_vs_reused() {
    let n_frames = 64;
    let code = LdpcCode::N512R12;

    // "Per-frame": rebuild the code object every iteration (status quo).
    let msg = random_bits(Ldpc::new(code).k(), 0x0F);
    let (fresh, dt1) = measure_throughput(
        || {
            for _ in 0..n_frames {
                let ldpc = Ldpc::new(code);
                let cw = ldpc.encode(&msg);
                black_box(cw[0]);
            }
            n_frames
        },
        n_frames,
        50,
    );
    println!(
        "[Construct per-frame LDPC×{}] {:.4} Mframes/s in {:.3}s",
        n_frames, fresh, dt1
    );

    // "Reused": build once, encode N times (the post-R3 amortized behavior).
    let ldpc = Ldpc::new(code);
    let (reused, dt2) = measure_throughput(
        || {
            for _ in 0..n_frames {
                let cw = ldpc.encode(&msg);
                black_box(cw[0]);
            }
            n_frames
        },
        n_frames,
        50,
    );
    println!(
        "[Construct reused LDPC×{}]    {:.4} Mframes/s in {:.3}s",
        n_frames, reused, dt2
    );
    println!(
        "[Construct LDPC reuse speedup] {:.1}×",
        reused / fresh.max(f32::MIN_POSITIVE)
    );
    // Measurement run — always passes.
}

// ── §1.2 Full COFDM frame chain (the per-link path the CodecCache lives on) ──
//
// Measures the real `OfdmFrameMod::modulate_frame` and batch `demodulate_frame`
// over many frames on ONE mod instance — the path where the per-instance
// CodecCache amortizes code construction across frames (each FEC code is built
// once, not per frame). "Msps" is total frame IQ samples / wall time, matching
// the "COFDM frame throughput" table in docs/performance.md. Both configs from
// that table are covered: LDPC(n512r12)+BCH(t=8) and Convolutional r1/2 +
// RS(60,52). Correctness is asserted each pass.

const FRAME_N: usize = 64; // n_fft
const FRAME_CP: usize = 8;

fn frame_config() -> OfdmConfig {
    let half = (FRAME_N / 2) as i32;
    let data: Vec<i32> = (1..half).chain(-(half - 1)..0).collect();
    let plan = CarrierPlan::new(FRAME_N, FRAME_CP).with_data_carriers(data);
    OfdmConfig::new(plan, 48_000.0, 0.0, 1.0, ConstellationOrder::Bpsk)
}

fn frame_preamble(cfg: &OfdmConfig) -> OfdmPreamble {
    OfdmPreamble::new(4, 16)
        .with_training_symbol(cfg.carrier_plan.n_fft(), cfg.carrier_plan.cp_len())
}

/// Measures mod (`modulate_frame`, cache warm across frames) and batch demod
/// (`demodulate_frame` with a persistent cache) for one concatenation, printing
/// both Msps. `mcs_index` selects the entry from `table` to exercise.
fn frame_chain(table: McsTable, mcs_index: u8, label: &str) {
    let cfg = frame_config();
    let pre = frame_preamble(&cfg);
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let payload = random_bytes(96, 0xF4A3);

    let frame = FramePacket::new(FrameMetadata::new(0x2468, mcs_index), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let frame_samples = iq.len();
    let body: Vec<C32> = iq[pre.total_len()..].to_vec();

    let repeats = 200;

    // Modulate path: one instance, many frames — cross-frame cache warm after the
    // first. The frame and seed are black-boxed each pass so the constant-input
    // modulation can't be hoisted, and the whole output is consumed.
    let mut seed = 0u32;
    let (mod_msps, mod_dt) = measure_throughput(
        || {
            seed = seed.wrapping_add(1);
            let iq = modu.modulate_frame(black_box(&frame), black_box(seed));
            let mut acc = 0.0f32;
            for s in &iq {
                acc += s.re;
            }
            black_box(acc);
            frame_samples
        },
        frame_samples,
        repeats,
    );
    println!("[Frame-Chain-Mod {label}] {mod_msps:.4} Msps in {mod_dt:.3}s");

    // Demodulate path: batch entry point with a caller-owned cache reused across
    // all passes, so decode throughput is measured on the SAME warm-cache footing
    // as the modulate path (not dominated by per-call code construction). Asserts
    // correctness each pass.
    let demod_cache = CodecCache::new();
    let (demod_msps, demod_dt) = measure_throughput(
        || {
            let got = demodulate_frame(&cfg, &table, &body, Some(&demod_cache)).expect("decode");
            assert_eq!(got.payload, payload, "frame chain: payload recovered");
            black_box(got.payload[0]);
            frame_samples
        },
        frame_samples,
        repeats,
    );
    println!("[Frame-Chain-Demod {label}] {demod_msps:.4} Msps in {demod_dt:.3}s");
    // Measurement run for the doc table; floor guards gross regressions only.
    let floor = minsps_from_env(0.05);
    assert!(mod_msps >= floor && demod_msps >= floor);
}

#[test]
fn throughput_frame_chain_ldpc_bch() {
    // The default ladder is LDPC(n512r12)+BCH(t=8); mcs 1 is the QPSK payload.
    frame_chain(McsTable::default_ladder(), 1, "LDPC+BCH");
}

#[test]
fn throughput_frame_chain_conv_rs() {
    // The DVB-style concatenation: punctured convolutional r1/2 + RS(60,52),
    // QPSK payload — the second row of the COFDM frame-throughput table.
    let table = McsTable::new(vec![Mcs::new(
        ConstellationOrder::Qpsk,
        InnerFec::Convolutional {
            rate: PunctureRate::R1_2,
        },
        OuterFec::ReedSolomon { n: 60, n_parity: 8 },
    )]);
    frame_chain(table, 0, "Conv+RS");
}
