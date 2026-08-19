// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use orion_sdr::codec::{crc16, crc32};
use orion_sdr::fec::{
    Bch, BchError, BlockInterleaver, ConvCode, ConvDeinterleaver, ConvInterleaver, DecodeRule,
    Gf256, Ldpc, LdpcCode, PnScrambler, PnScramblerStream, PunctureRate, ReedSolomon, RsError,
    conv_encode_punctured, conv_encode_punctured_with, conv_roundtrip_delay, punctured_coded_len,
    punctured_coded_len_with, viterbi_decode_soft, viterbi_decode_soft_with,
};

// Small deterministic xorshift for reproducible test messages/errors.
fn xorshift(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

// ── GF(2^8) arithmetic ─────────────────────────────────────────────────────

#[test]
fn gf256_add_is_xor_and_self_inverse() {
    let gf = Gf256::new();
    for a in 0u8..=255 {
        for b in [0u8, 1, 7, 42, 128, 255] {
            assert_eq!(gf.add(a, b), a ^ b);
            // Adding twice cancels.
            assert_eq!(gf.add(gf.add(a, b), b), a);
        }
    }
}

#[test]
fn gf256_mul_identity_and_zero() {
    let gf = Gf256::new();
    for a in 0u8..=255 {
        assert_eq!(gf.mul(a, 1), a, "1 is the multiplicative identity");
        assert_eq!(gf.mul(a, 0), 0, "0 is absorbing");
        assert_eq!(gf.mul(0, a), 0, "0 is absorbing");
    }
}

#[test]
fn gf256_mul_is_commutative_and_associative() {
    let gf = Gf256::new();
    let sample = [1u8, 2, 3, 5, 17, 99, 200, 255];
    for &a in &sample {
        for &b in &sample {
            assert_eq!(gf.mul(a, b), gf.mul(b, a));
            for &c in &sample {
                assert_eq!(gf.mul(gf.mul(a, b), c), gf.mul(a, gf.mul(b, c)));
            }
        }
    }
}

#[test]
fn gf256_inverse_and_division() {
    let gf = Gf256::new();
    for a in 1u8..=255 {
        let inv = gf.inv(a);
        assert_eq!(gf.mul(a, inv), 1, "a * a^-1 == 1");
        // Division is multiplication by the inverse.
        for b in 1u8..=255 {
            assert_eq!(gf.div(a, b), gf.mul(a, gf.inv(b)));
            // (a / b) * b == a
            assert_eq!(gf.mul(gf.div(a, b), b), a);
        }
    }
}

#[test]
fn gf256_pow_matches_repeated_mul() {
    let gf = Gf256::new();
    for a in [1u8, 2, 3, 10, 100, 255] {
        let mut acc = 1u8;
        for n in 0..20usize {
            assert_eq!(gf.pow(a, n), acc, "pow({a}, {n})");
            acc = gf.mul(acc, a);
        }
    }
}

#[test]
fn gf256_exp_log_round_trip() {
    let gf = Gf256::new();
    // g^i cycles with period 255; exp_of/log_of invert each other on nonzero.
    for a in 1u8..=255 {
        let l = gf.log_of(a) as usize;
        assert_eq!(gf.exp_of(l), a);
    }
    // The generator has full order 255.
    assert_eq!(gf.exp_of(0), 1);
    assert_eq!(gf.exp_of(255), 1, "g^255 == 1 (order 255)");
}

// ── Block interleaver ──────────────────────────────────────────────────────

fn interleave_round_trip<T: Copy + PartialEq + std::fmt::Debug>(
    il: &BlockInterleaver,
    data: &[T],
    fill: T,
) {
    let n = il.block_len();
    // Pad the data up to a full block (the frame layer does this in practice).
    let mut block: Vec<T> = data.to_vec();
    block.resize(n, fill);

    let mut interleaved = vec![fill; n];
    il.interleave(&block, &mut interleaved);

    let mut restored = vec![fill; n];
    il.deinterleave(&interleaved, &mut restored);

    assert_eq!(restored, block, "deinterleave∘interleave must be identity");
}

#[test]
fn interleaver_round_trip_bytes_square() {
    let il = BlockInterleaver::new(4, 4);
    let data: Vec<u8> = (0..16).collect();
    interleave_round_trip(&il, &data, 0u8);
}

#[test]
fn interleaver_round_trip_bytes_non_square() {
    let il = BlockInterleaver::new(3, 7);
    let data: Vec<u8> = (0..21).map(|i| (i * 3 + 1) as u8).collect();
    interleave_round_trip(&il, &data, 0u8);
}

#[test]
fn interleaver_round_trip_bytes_padded() {
    // 5×5 = 25 slots, only 19 real elements → final partial row padded.
    let il = BlockInterleaver::new(5, 5);
    let data: Vec<u8> = (0..19).map(|i| (200 - i) as u8).collect();
    interleave_round_trip(&il, &data, 0u8);
}

#[test]
fn interleaver_round_trip_llrs() {
    // The inner deinterleaver runs in the f32 (LLR) domain.
    let il = BlockInterleaver::new(6, 4);
    let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.5 - 5.0).collect();
    interleave_round_trip(&il, &data, 0.0f32);
}

#[test]
fn interleaver_actually_permutes_across_rows() {
    // A burst confined to one row of the input must be spread across columns
    // (i.e. non-adjacent) in the interleaved output — the whole point.
    let il = BlockInterleaver::new(4, 4);
    // Row 0 = ones, everything else zero.
    let mut block = vec![0u8; 16];
    for slot in block.iter_mut().take(4) {
        *slot = 1;
    }
    let mut out = vec![0u8; 16];
    il.interleave(&block, &mut out);
    // The four 1s should land at output indices 0, 4, 8, 12 (one per column
    // read-out), i.e. spread by `rows` apart — never adjacent.
    let ones: Vec<usize> = out
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v == 1)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(ones, vec![0, 4, 8, 12]);
}

// ── Forney convolutional interleaver (streaming) ───────────────────────────

#[test]
fn conv_interleaver_dvb_t_dimensions() {
    let ci = ConvInterleaver::dvb_t();
    assert_eq!((ci.branches(), ci.depth()), (12, 17));
    assert_eq!(204, ci.branches() * ci.depth()); // one RS(204,188) codeword
    assert_eq!(ci.roundtrip_delay(), conv_roundtrip_delay(12, 17));
    assert_eq!(ci.roundtrip_delay(), 12 * 11 * 17);
}

/// Frame-mode round-trip: reset, feed the payload, flush; then deinterleave the
/// whole stream and recover the payload at output offset `roundtrip_delay`.
fn conv_frame_round_trip(branches: usize, depth: usize, payload: &[u8]) {
    let d = conv_roundtrip_delay(branches, depth);
    let mut il = ConvInterleaver::new(branches, depth);
    let mut interleaved = il.feed(payload);
    interleaved.extend_from_slice(&il.flush());
    assert_eq!(interleaved.len(), payload.len() + d);

    let mut di = ConvDeinterleaver::new(branches, depth);
    let deint = di.feed(&interleaved);
    assert_eq!(&deint[d..d + payload.len()], payload, "frame-mode recovery");
}

#[test]
fn conv_interleaver_frame_round_trip_dvb_t() {
    for &n in &[204usize, 408, 2040] {
        let mut r = xorshift(0xF0 ^ n as u64);
        let payload: Vec<u8> = (0..n).map(|_| (r() & 0xff) as u8).collect();
        conv_frame_round_trip(12, 17, &payload);
    }
}

#[test]
fn conv_interleaver_frame_round_trip_custom_dims() {
    let payload: Vec<u8> = (0..96).map(|i| (i * 5 + 1) as u8).collect();
    conv_frame_round_trip(4, 3, &payload);
}

#[test]
fn conv_interleaver_stream_equals_chunked() {
    // Stream mode: feeding a byte stream in one call vs. across arbitrary chunk
    // boundaries must produce identical interleaver output (state carries).
    let mut r = xorshift(0x5EED);
    let data: Vec<u8> = (0..500).map(|_| (r() & 0xff) as u8).collect();

    let mut one_shot = ConvInterleaver::dvb_t();
    let full = one_shot.feed(&data);

    let mut chunked = ConvInterleaver::dvb_t();
    let mut acc = Vec::new();
    for chunk in data.chunks(37) {
        acc.extend_from_slice(&chunked.feed(chunk));
    }
    assert_eq!(full, acc, "streaming feed is chunk-boundary-invariant");
}

#[test]
fn conv_interleaver_reset_restarts() {
    let data: Vec<u8> = (1..=204).collect();
    let mut ci = ConvInterleaver::dvb_t();
    let first = ci.feed(&data);
    ci.reset();
    let second = ci.feed(&data);
    assert_eq!(
        first, second,
        "reset returns the interleaver to its initial state"
    );
}

#[test]
fn conv_interleaver_spreads_a_burst() {
    // A contiguous input burst must be dispersed across the interleaver depth.
    let d = conv_roundtrip_delay(12, 17);
    let mut data = vec![0u8; 204];
    for slot in data.iter_mut().skip(24).take(12) {
        *slot = 1; // a 12-byte burst, one per branch
    }
    let mut il = ConvInterleaver::dvb_t();
    let mut out = il.feed(&data);
    out.extend_from_slice(&il.flush());
    let ones: Vec<usize> = out
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v == 1)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(ones.len(), 12, "burst bytes preserved");
    let span = ones.last().unwrap() - ones.first().unwrap();
    assert!(
        span > 12,
        "burst must spread beyond its original 12-byte span, got {span}"
    );
    let _ = d;
}

// ── PN scrambler ───────────────────────────────────────────────────────────

// A few representative additive LFSR parameterizations.
fn scramblers() -> Vec<PnScrambler> {
    vec![
        // 802.11-style: x^7 + x^4 + 1, 7-bit register.
        PnScrambler::new(0b1001, 7, 0x7F),
        // DVB energy-dispersal-style: x^15 + x^14 + 1, 15-bit register.
        PnScrambler::new(0b11 << 13, 15, 0b100_1010_1000_0000),
        // A wide 32-bit register.
        PnScrambler::new(0x8020_0003, 32, 0x1234_5678),
    ]
}

#[test]
fn scrambler_is_self_inverse() {
    let original: Vec<u8> = (0..64).map(|i| (i * 7 + 3) as u8).collect();
    for sc in scramblers() {
        let mut data = original.clone();
        sc.scramble(&mut data);
        assert_ne!(data, original, "scrambling must change the data");
        sc.scramble(&mut data);
        assert_eq!(data, original, "scramble∘scramble must be identity");
    }
}

#[test]
fn scrambler_breaks_all_zero_run() {
    // An all-zero payload must come out non-trivially whitened (no long runs
    // of a constant), which is the reason a whitener exists.
    for sc in scramblers() {
        let mut data = vec![0u8; 32];
        sc.scramble(&mut data);
        let nonzero = data.iter().filter(|&&b| b != 0).count();
        assert!(
            nonzero > data.len() / 2,
            "whitened all-zero input should be mostly nonzero, got {nonzero} nonzero bytes"
        );
    }
}

#[test]
fn scrambler_deterministic() {
    let sc = PnScrambler::new(0b1001, 7, 0x7F);
    let mut a = vec![0xAAu8; 40];
    let mut b = vec![0xAAu8; 40];
    sc.scramble(&mut a);
    sc.scramble(&mut b);
    assert_eq!(a, b, "same seed/params → same PN sequence");
}

// ── Streaming PN scrambler ─────────────────────────────────────────────────

#[test]
fn scrambler_stream_matches_oneshot() {
    // A single stream `feed` over the whole buffer must equal the stateless
    // `scramble` over the same buffer (identical PN advance).
    let original: Vec<u8> = (0..128).map(|i| (i * 13 + 5) as u8).collect();
    for sc in scramblers() {
        let mut oneshot = original.clone();
        sc.scramble(&mut oneshot);

        let streamed = sc.into_stream().feed(&original);
        assert_eq!(streamed, oneshot, "stream feed == stateless scramble");
    }
}

#[test]
fn scrambler_stream_carries_across_chunks() {
    // reset + chunked feeds must equal one scramble over the concatenation —
    // the register state carries across `feed` calls.
    let original: Vec<u8> = (0..200).map(|i| (i * 37 + 11) as u8).collect();
    let sc = PnScrambler::new(0b11 << 13, 15, 0b100_1010_1000_0000); // DVB LFSR
    let mut oneshot = original.clone();
    sc.scramble(&mut oneshot);

    let mut stream = sc.into_stream();
    let mut acc = Vec::new();
    for chunk in original.chunks(23) {
        acc.extend_from_slice(&stream.feed(chunk));
    }
    assert_eq!(acc, oneshot, "chunked stream feeds == one-shot scramble");
}

#[test]
fn scrambler_stream_self_inverse_continuous() {
    // Feeding a continuous stream through a scrambler, then through a freshly
    // reset descrambler with the same params, recovers the original — even when
    // fed in different chunkings on each side.
    let original: Vec<u8> = (0..300).map(|i| (i * 7 + 1) as u8).collect();
    let mut enc = PnScramblerStream::new(0b11 << 13, 15, 0b100_1010_1000_0000);
    let scrambled: Vec<u8> = original.chunks(50).flat_map(|c| enc.feed(c)).collect();
    assert_ne!(scrambled, original);

    let mut dec = PnScramblerStream::new(0b11 << 13, 15, 0b100_1010_1000_0000);
    let recovered: Vec<u8> = scrambled.chunks(17).flat_map(|c| dec.feed(c)).collect();
    assert_eq!(
        recovered, original,
        "continuous descramble recovers the stream"
    );
}

#[test]
fn scrambler_stream_reset_restarts() {
    let mut stream = PnScramblerStream::new(0b1001, 7, 0x7F);
    let data: Vec<u8> = (0..40).map(|i| (i * 3) as u8).collect();
    let first = stream.feed(&data);
    stream.reset();
    let second = stream.feed(&data);
    assert_eq!(first, second, "reset returns the stream to its seed state");
}

#[test]
fn scrambler_stream_feed_in_place_matches_feed() {
    let data: Vec<u8> = (0..64).map(|i| (i * 5 + 2) as u8).collect();
    let mut a = PnScramblerStream::new(0x8020_0003, 32, 0x1234_5678);
    let out = a.feed(&data);
    let mut b = PnScramblerStream::new(0x8020_0003, 32, 0x1234_5678);
    let mut in_place = data.clone();
    b.feed_in_place(&mut in_place);
    assert_eq!(out, in_place, "feed and feed_in_place agree");
}

// ── Generic CRCs ───────────────────────────────────────────────────────────

#[test]
fn crc_known_answer_vectors() {
    // The canonical check string "123456789".
    assert_eq!(crc16(b"123456789"), 0x29B1, "CRC-16/CCITT-FALSE");
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926, "CRC-32/ISO-HDLC");
}

#[test]
fn crc_detects_single_bit_flip() {
    let msg = b"orion-sdr COFDM frame payload";
    let good16 = crc16(msg);
    let good32 = crc32(msg);

    for bit in 0..(msg.len() * 8) {
        let mut corrupted = msg.to_vec();
        corrupted[bit / 8] ^= 1 << (bit % 8);
        assert_ne!(crc16(&corrupted), good16, "CRC-16 must catch bit {bit}");
        assert_ne!(crc32(&corrupted), good32, "CRC-32 must catch bit {bit}");
    }
}

#[test]
fn crc_empty_input() {
    // Well-defined on empty input (the init/final-xor values).
    assert_eq!(crc16(b""), 0xFFFF);
    assert_eq!(crc32(b""), 0x0000_0000);
}

// ── BCH(n, k, t) ───────────────────────────────────────────────────────────

#[test]
fn bch_dimensions_are_consistent() {
    for t in [1usize, 2, 3, 8] {
        let code = Bch::new(t).unwrap();
        assert_eq!(code.n(), 255);
        assert_eq!(code.t(), t);
        assert_eq!(code.n() - code.k(), code.parity_bits());
        assert!(code.k() < code.n());
    }
}

#[test]
fn bch_encode_is_systematic() {
    let code = Bch::new(3).unwrap();
    let mut r = xorshift(0x5151);
    let msg: Vec<u8> = (0..code.k()).map(|_| (r() & 1) as u8).collect();
    let cw = code.encode(&msg);
    assert_eq!(cw.len(), code.n());
    assert_eq!(
        &cw[..code.k()],
        &msg[..],
        "message is the systematic prefix"
    );
}

#[test]
fn bch_corrects_up_to_t_errors() {
    for t in [1usize, 2, 3, 8] {
        let code = Bch::new(t).unwrap();
        let mut r = xorshift(0xB0B0 ^ t as u64);
        let msg: Vec<u8> = (0..code.k()).map(|_| (r() & 1) as u8).collect();
        let cw = code.encode(&msg);

        // Exactly t spread-out bit errors must be corrected.
        let mut rx = cw.clone();
        let mut e = xorshift(0xE7E7 ^ t as u64);
        let mut flipped = std::collections::HashSet::new();
        while flipped.len() < t {
            let pos = (e() as usize) % code.n();
            flipped.insert(pos);
        }
        for &pos in &flipped {
            rx[pos] ^= 1;
        }
        assert_eq!(
            code.decode(&rx).unwrap(),
            msg,
            "t={t}: {t} errors must be corrected"
        );
    }
}

#[test]
fn bch_zero_errors_round_trips() {
    let code = Bch::new(2).unwrap();
    let msg: Vec<u8> = (0..code.k()).map(|i| (i % 3 == 0) as u8).collect();
    let cw = code.encode(&msg);
    assert_eq!(code.decode(&cw).unwrap(), msg);
}

#[test]
fn bch_shortened_corrects() {
    let code = Bch::shortened(128, 3).unwrap();
    assert_eq!(code.n(), 128);
    let mut r = xorshift(0x1234);
    let msg: Vec<u8> = (0..code.k()).map(|_| (r() & 1) as u8).collect();
    let cw = code.encode(&msg);
    let mut rx = cw.clone();
    for &pos in &[2usize, 61, 120] {
        rx[pos] ^= 1;
    }
    assert_eq!(code.decode(&rx).unwrap(), msg);
}

#[test]
fn bch_flags_uncorrectable_beyond_t() {
    // With t+several errors the decoder must not silently return a wrong
    // message: it either errors or (rarely) miscorrects, but never returns the
    // original message. We assert it does not return `msg`.
    let code = Bch::new(2).unwrap();
    let mut r = xorshift(0x9A9A);
    let msg: Vec<u8> = (0..code.k()).map(|_| (r() & 1) as u8).collect();
    let cw = code.encode(&msg);
    let mut rx = cw.clone();
    // 6 errors, well beyond t=2.
    for &pos in &[1usize, 40, 80, 120, 160, 200] {
        rx[pos] ^= 1;
    }
    match code.decode(&rx) {
        Err(BchError::Uncorrectable(_)) => {}
        Ok(decoded) => assert_ne!(decoded, msg, "must not silently recover the original"),
        Err(other) => panic!("unexpected error {other:?}"),
    }
}

// ── LDPC (fixed family) ────────────────────────────────────────────────────

const LDPC_CODES: [LdpcCode; 3] = [LdpcCode::N512R12, LdpcCode::N576R23, LdpcCode::N512R34];

#[test]
fn ldpc_encode_produces_valid_codeword() {
    for code in LDPC_CODES {
        let ldpc = Ldpc::new(code);
        assert_eq!(ldpc.n(), code.n());
        assert_eq!(ldpc.k(), code.k());
        let mut r = xorshift(0xADD1 ^ code.n() as u64);
        let msg: Vec<u8> = (0..ldpc.k()).map(|_| (r() & 1) as u8).collect();
        let cw = ldpc.encode(&msg);
        assert_eq!(cw.len(), ldpc.n());
        assert_eq!(&cw[..ldpc.k()], &msg[..], "systematic prefix");
        assert_eq!(
            ldpc.syndrome_weight(&cw),
            0,
            "encoded word must satisfy every parity check"
        );
    }
}

#[test]
fn ldpc_clean_llrs_decode_exactly() {
    for code in LDPC_CODES {
        let ldpc = Ldpc::new(code);
        let mut r = xorshift(0xC1EA ^ code.n() as u64);
        let msg: Vec<u8> = (0..ldpc.k()).map(|_| (r() & 1) as u8).collect();
        let cw = ldpc.encode(&msg);
        let llr: Vec<f32> = cw
            .iter()
            .map(|&b| if b == 0 { 8.0 } else { -8.0 })
            .collect();
        let (decoded, unsat) = ldpc.decode_soft(&llr, 50);
        assert_eq!(unsat, 0);
        assert_eq!(decoded, msg);
    }
}

#[test]
fn ldpc_corrects_bit_errors_from_soft_llrs() {
    // Present strong LLRs with a modest number of sign errors (weak-wrong
    // values); belief propagation must converge to the transmitted message.
    for (code, n_err) in [(LdpcCode::N512R12, 6usize), (LdpcCode::N512R34, 3)] {
        let ldpc = Ldpc::new(code);
        let mut r = xorshift(0x50F7 ^ code.n() as u64);
        let msg: Vec<u8> = (0..ldpc.k()).map(|_| (r() & 1) as u8).collect();
        let cw = ldpc.encode(&msg);

        let mut llr: Vec<f32> = cw
            .iter()
            .map(|&b| if b == 0 { 6.0 } else { -6.0 })
            .collect();
        let mut e = xorshift(0xE770 ^ code.n() as u64);
        for _ in 0..n_err {
            let pos = (e() as usize) % ldpc.n();
            // Flip to a weak wrong-sign LLR (a soft error, not a hard erasure).
            llr[pos] = -llr[pos] * 0.5;
        }

        let (decoded, unsat) = ldpc.decode_soft(&llr, 50);
        assert_eq!(
            unsat, 0,
            "{code:?}: BP should converge with {n_err} soft errors"
        );
        assert_eq!(decoded, msg, "{code:?}: message recovered");
    }
}

#[test]
fn ldpc_decode_rule_default_matches_sum_product() {
    // `decode_soft` must be exactly `decode_soft_with(.., SumProduct)`.
    let ldpc = Ldpc::new(LdpcCode::N512R12);
    let mut r = xorshift(0x5A11);
    let msg: Vec<u8> = (0..ldpc.k()).map(|_| (r() & 1) as u8).collect();
    let cw = ldpc.encode(&msg);
    let mut llr: Vec<f32> = cw
        .iter()
        .map(|&b| if b == 0 { 6.0 } else { -6.0 })
        .collect();
    let mut e = xorshift(0x5A12);
    for _ in 0..5 {
        let pos = (e() as usize) % ldpc.n();
        llr[pos] = -llr[pos] * 0.5;
    }
    let (a, ua) = ldpc.decode_soft(&llr, 50);
    let (b, ub) = ldpc.decode_soft_with(&llr, 50, DecodeRule::SumProduct);
    assert_eq!((a, ua), (b, ub), "default decode == explicit SumProduct");
}

#[test]
fn ldpc_min_sum_rules_decode() {
    // Both min-sum variants must correct a modest error load (the scaffold is a
    // real decoder, not a stub). Scaled min-sum (0.75) is the near-recovery point.
    for rule in [DecodeRule::MinSum, DecodeRule::ScaledMinSum(0.75)] {
        for (code, n_err) in [(LdpcCode::N512R12, 5usize), (LdpcCode::N512R34, 3)] {
            let ldpc = Ldpc::new(code);
            let mut r = xorshift(0x3E00 ^ code.n() as u64);
            let msg: Vec<u8> = (0..ldpc.k()).map(|_| (r() & 1) as u8).collect();
            let cw = ldpc.encode(&msg);
            let mut llr: Vec<f32> = cw
                .iter()
                .map(|&b| if b == 0 { 6.0 } else { -6.0 })
                .collect();
            let mut e = xorshift(0x3E01 ^ code.n() as u64);
            for _ in 0..n_err {
                let pos = (e() as usize) % ldpc.n();
                llr[pos] = -llr[pos] * 0.5;
            }
            let (decoded, unsat) = ldpc.decode_soft_with(&llr, 50, rule);
            assert_eq!(unsat, 0, "{rule:?} {code:?}: should converge");
            assert_eq!(decoded, msg, "{rule:?} {code:?}: message recovered");
        }
    }
}

#[test]
fn ldpc_soft_llr_convention_matches_sign() {
    // A gentle all-correct LLR field (small magnitude, right signs) must decode
    // to the message without any correction, confirming the +⇒bit-0 convention.
    let ldpc = Ldpc::new(LdpcCode::N512R12);
    let msg: Vec<u8> = (0..ldpc.k()).map(|i| (i % 2) as u8).collect();
    let cw = ldpc.encode(&msg);
    let llr: Vec<f32> = cw
        .iter()
        .map(|&b| if b == 0 { 1.5 } else { -1.5 })
        .collect();
    let (decoded, unsat) = ldpc.decode_soft(&llr, 50);
    assert_eq!(unsat, 0);
    assert_eq!(decoded, msg);
}

// ── Punctured convolutional code ───────────────────────────────────────────

const PUNCTURE_RATES: [PunctureRate; 5] = [
    PunctureRate::R1_2,
    PunctureRate::R2_3,
    PunctureRate::R3_4,
    PunctureRate::R5_6,
    PunctureRate::R7_8,
];

#[test]
fn conv_noiseless_roundtrip_every_rate() {
    for rate in PUNCTURE_RATES {
        let mut r = xorshift(0x7A57 ^ format!("{rate:?}").len() as u64);
        let info: Vec<u8> = (0..120).map(|_| (r() & 1) as u8).collect();
        let coded = conv_encode_punctured(&info, rate);
        assert_eq!(
            coded.len(),
            punctured_coded_len(info.len(), rate),
            "coded length matches the size predictor for {rate:?}"
        );
        // Strong LLRs: bit 0 → +4, bit 1 → −4.
        let llrs: Vec<f32> = coded
            .iter()
            .map(|&b| if b == 0 { 4.0 } else { -4.0 })
            .collect();
        let decoded = viterbi_decode_soft(&llrs, info.len(), rate);
        assert_eq!(decoded, info, "noiseless roundtrip for {rate:?}");
    }
}

#[test]
fn conv_corrects_sparse_errors_rate_half() {
    let mut r = xorshift(0xC0DE);
    let info: Vec<u8> = (0..96).map(|_| (r() & 1) as u8).collect();
    let coded = conv_encode_punctured(&info, PunctureRate::R1_2);
    let mut llrs: Vec<f32> = coded
        .iter()
        .map(|&b| if b == 0 { 4.0 } else { -4.0 })
        .collect();
    // Corrupt a handful of well-separated coded bits (weak wrong-sign LLR).
    for &i in &[3usize, 44, 90, 150] {
        if i < llrs.len() {
            llrs[i] = -llrs[i] * 0.5;
        }
    }
    let decoded = viterbi_decode_soft(&llrs, info.len(), PunctureRate::R1_2);
    assert_eq!(decoded, info, "rate-1/2 Viterbi corrects sparse errors");
}

// ── DVB-T K=7 convolutional inner code ─────────────────────────────────────

#[test]
fn conv_k7_generators_known_answer() {
    // DVB-T inner code, G0 = 0o171 = 1111001, G1 = 0o133 = 1011011 (ETSI EN
    // 300 744 §4.3.3). Encode a single 1 bit from the all-zero state, no tail:
    // window = (1 << 6) | 0 = 0b1000000. Only the MSB (input tap) is set, which
    // both G0 and G1 include, so (c0, c1) = (1, 1). The next few bits shift the
    // 1 down the register and exercise the lower taps.
    //
    // Full hand-computed output for input [1, 0, 0, 0, 0, 0, 0] (a 1 then six
    // zeros — the 1 walks the whole 6-bit register), rate-1/2 (unpunctured):
    //   reg before each step (low..high after insert): the impulse response of
    //   the two generators, i.e. the generator taps themselves read MSB-first.
    // G0 = 1111001 → 1,1,1,1,0,0,1 ; G1 = 1011011 → 1,0,1,1,0,1,1
    // interleaved [g0_0,g1_0, g0_1,g1_1, …]:
    let info = [1u8, 0, 0, 0, 0, 0, 0];
    // conv_encode_punctured_with appends 6 zero tail bits; take the first 14
    // coded bits (7 steps) which are the pure impulse response.
    let coded = conv_encode_punctured_with(ConvCode::DvbK7, &info, PunctureRate::R1_2);
    let g0_impulse = [1u8, 1, 1, 1, 0, 0, 1];
    let g1_impulse = [1u8, 0, 1, 1, 0, 1, 1];
    for step in 0..7 {
        assert_eq!(
            coded[step * 2],
            g0_impulse[step],
            "K7 G0 impulse mismatch at step {step}"
        );
        assert_eq!(
            coded[step * 2 + 1],
            g1_impulse[step],
            "K7 G1 impulse mismatch at step {step}"
        );
    }
}

#[test]
fn conv_k7_noiseless_roundtrip_every_rate() {
    for rate in PUNCTURE_RATES {
        let mut r = xorshift(0x0DB7 ^ format!("{rate:?}").len() as u64);
        let info: Vec<u8> = (0..188).map(|_| (r() & 1) as u8).collect();
        let coded = conv_encode_punctured_with(ConvCode::DvbK7, &info, rate);
        assert_eq!(
            coded.len(),
            punctured_coded_len_with(ConvCode::DvbK7, info.len(), rate),
            "K7 coded length matches the size predictor for {rate:?}"
        );
        let llrs: Vec<f32> = coded
            .iter()
            .map(|&b| if b == 0 { 4.0 } else { -4.0 })
            .collect();
        let decoded = viterbi_decode_soft_with(ConvCode::DvbK7, &llrs, info.len(), rate);
        assert_eq!(decoded, info, "K7 noiseless roundtrip for {rate:?}");
    }
}

#[test]
fn conv_k7_corrects_more_errors_than_k5() {
    // The K=7 code has a larger free distance (dfree = 10 vs 7 for the K=5
    // code), so it corrects a denser error burst at rate 1/2. Verify K7
    // recovers a pattern that also stresses the decoder.
    let mut r = xorshift(0xD7B7);
    let info: Vec<u8> = (0..188).map(|_| (r() & 1) as u8).collect();
    let coded = conv_encode_punctured_with(ConvCode::DvbK7, &info, PunctureRate::R1_2);
    let mut llrs: Vec<f32> = coded
        .iter()
        .map(|&b| if b == 0 { 4.0 } else { -4.0 })
        .collect();
    // Flip several well-separated coded bits to the wrong sign.
    for &i in &[7usize, 33, 61, 100, 140, 200, 260] {
        if i < llrs.len() {
            llrs[i] = -llrs[i];
        }
    }
    let decoded = viterbi_decode_soft_with(ConvCode::DvbK7, &llrs, info.len(), PunctureRate::R1_2);
    assert_eq!(decoded, info, "K7 rate-1/2 Viterbi corrects a sparse burst");
}

#[test]
fn conv_k5_unchanged_by_generalization() {
    // The K5 path must stay bit-identical to the original. Encode via both the
    // legacy wrapper and the explicit-code entry; they must agree, and the
    // explicit K5 must match what the wrapper produced pre-generalization.
    let mut r = xorshift(0x5A5A);
    let info: Vec<u8> = (0..120).map(|_| (r() & 1) as u8).collect();
    let legacy = conv_encode_punctured(&info, PunctureRate::R2_3);
    let explicit = conv_encode_punctured_with(ConvCode::K5, &info, PunctureRate::R2_3);
    assert_eq!(legacy, explicit, "K5 legacy and explicit paths agree");
    let llrs: Vec<f32> = legacy
        .iter()
        .map(|&b| if b == 0 { 4.0 } else { -4.0 })
        .collect();
    assert_eq!(
        viterbi_decode_soft(&llrs, info.len(), PunctureRate::R2_3),
        viterbi_decode_soft_with(ConvCode::K5, &llrs, info.len(), PunctureRate::R2_3),
        "K5 legacy and explicit decoders agree"
    );
}

// ── Reed–Solomon ───────────────────────────────────────────────────────────

#[test]
fn rs_dvb_dimensions() {
    let rs = ReedSolomon::dvb();
    assert_eq!((rs.n(), rs.k(), rs.t()), (204, 188, 8));
    assert_eq!(rs.parity_bytes(), 16);
}

#[test]
fn rs_corrects_t_symbol_errors() {
    let rs = ReedSolomon::dvb();
    let mut r = xorshift(0x2004);
    let msg: Vec<u8> = (0..rs.k()).map(|_| (r() & 0xff) as u8).collect();
    let cw = rs.encode(&msg);
    assert_eq!(cw.len(), rs.n());
    assert_eq!(&cw[..rs.k()], &msg[..], "systematic prefix");

    // Corrupt exactly t=8 bytes with arbitrary nonzero magnitudes.
    let mut rx = cw.clone();
    let mut e = xorshift(0x8888);
    let mut positions = std::collections::HashSet::new();
    while positions.len() < rs.t() {
        positions.insert((e() as usize) % rs.n());
    }
    for &pos in &positions {
        rx[pos] ^= ((e() & 0xff) as u8).max(1);
    }
    assert_eq!(rs.decode(&rx).unwrap(), msg, "RS corrects t symbol errors");
}

#[test]
fn rs_zero_errors_round_trips() {
    let rs = ReedSolomon::dvb();
    let msg: Vec<u8> = (0..rs.k()).map(|i| (i % 251) as u8).collect();
    let cw = rs.encode(&msg);
    assert_eq!(rs.decode(&cw).unwrap(), msg);
}

#[test]
fn rs_flags_uncorrectable_beyond_t() {
    let rs = ReedSolomon::dvb();
    let msg: Vec<u8> = (0..rs.k()).map(|i| (i * 3 % 256) as u8).collect();
    let cw = rs.encode(&msg);
    let mut rx = cw.clone();
    // t+1 = 9 symbol errors.
    for &pos in &[0usize, 11, 22, 33, 44, 55, 66, 77, 88] {
        rx[pos] ^= 0x7E;
    }
    match rs.decode(&rx) {
        Err(RsError::Uncorrectable(_)) => {}
        Ok(d) => assert_ne!(d, msg, "9 errors must not recover the original"),
        Err(other) => panic!("unexpected RS error {other:?}"),
    }
}

#[test]
fn rs_shortened_corrects() {
    let rs = ReedSolomon::new(40, 8).unwrap(); // t = 4
    assert_eq!((rs.n(), rs.k(), rs.t()), (40, 32, 4));
    let msg: Vec<u8> = (0..rs.k()).map(|i| (i * 7 + 1) as u8).collect();
    let cw = rs.encode(&msg);
    let mut rx = cw.clone();
    for &pos in &[1usize, 15, 28, 38] {
        rx[pos] ^= 0x33;
    }
    assert_eq!(rs.decode(&rx).unwrap(), msg);
}

#[test]
fn rs_decode_counted_reports_the_number_of_bytes_it_fixed() {
    // The measurement `outer_ok` cannot make: how HARD the decoder had to work.
    // For each error count from 0 to t, the reported tally must equal exactly
    // the number of bytes corrupted — not the locator degree, not the root
    // count, and not a saturating flag.
    let rs = ReedSolomon::dvb();
    let msg: Vec<u8> = (0..rs.k()).map(|i| (i % 251) as u8).collect();
    let cw = rs.encode(&msg);

    for n_errors in 0..=rs.t() {
        let mut rx = cw.clone();
        // Spread the corruptions so no two share a position.
        for j in 0..n_errors {
            rx[j * 17 + 3] ^= 0x5A;
        }
        let (got, corrected) = rs.decode_counted(&rx).expect("within t");
        assert_eq!(got, msg, "{n_errors} errors still recover the message");
        assert_eq!(
            corrected, n_errors,
            "decode_counted must report exactly the {n_errors} bytes it fixed"
        );
    }
}

#[test]
fn rs_decode_counted_includes_parity_bytes() {
    // The count spans the whole codeword, not just the k message bytes returned.
    // A byte error in the parity region is one the channel caused and the
    // decoder absorbed; excluding it would under-report the channel by the
    // parity fraction of every codeword.
    let rs = ReedSolomon::dvb();
    let msg: Vec<u8> = (0..rs.k()).map(|i| (i * 5 % 251) as u8).collect();
    let cw = rs.encode(&msg);

    // Corrupt three bytes, all inside the parity region [k, n).
    let mut rx = cw.clone();
    for off in [0usize, 5, 11] {
        rx[rs.k() + off] ^= 0x9C;
    }
    let (got, corrected) = rs.decode_counted(&rx).expect("3 parity errors < t");
    assert_eq!(got, msg);
    assert_eq!(
        corrected, 3,
        "parity-region corrections count toward the total"
    );
}

#[test]
fn rs_decode_counted_agrees_with_decode() {
    // `decode` is a thin wrapper over `decode_counted`; the two must never
    // disagree about the recovered message, including on the uncorrectable path.
    let rs = ReedSolomon::new(40, 8).unwrap();
    let msg: Vec<u8> = (0..rs.k()).map(|i| (i * 7 + 1) as u8).collect();
    let cw = rs.encode(&msg);

    // Correctable.
    let mut rx = cw.clone();
    for &pos in &[1usize, 15, 28, 38] {
        rx[pos] ^= 0x33;
    }
    assert_eq!(rs.decode(&rx).unwrap(), rs.decode_counted(&rx).unwrap().0);

    // Uncorrectable: t+1 = 5 errors. Both arms report failure, and the counted
    // form yields no tally — a correction the decoder does not trust is not one
    // to report.
    let mut bad = cw.clone();
    for &pos in &[0usize, 8, 16, 24, 32] {
        bad[pos] ^= 0x71;
    }
    assert_eq!(rs.decode(&bad).is_err(), rs.decode_counted(&bad).is_err());
}
