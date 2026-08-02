// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// DVB-T TPS signalling unit tests: the standalone GF(2^7) BCH(67,53) code,
// the 68-bit TPS word pack/unpack, and the DBPSK-along-the-symbol-axis codec.

use num_complex::Complex32 as C32;
use orion_sdr::fec::PunctureRate;
use orion_sdr::modulate::ConstellationOrder;
use orion_sdr::waveform::dvb_t::GuardInterval;
use orion_sdr::waveform::dvb_t_tps::{
    TPS_CARRIER_COUNT, TPS_CODEWORD_BITS, TPS_INFO_BITS, TPS_PARITY_BITS, TPS_SYMBOLS_PER_FRAME,
    TPS_SYNC_WORD_13, TPS_SYNC_WORD_24, TpsDecoder, TpsEncoder, TpsWord, tps_bch_decode,
    tps_bch_encode,
};

// ── GF(2^7) BCH(67,53) t=2 (EN 300 744 §4.6.2.11) ──────────────────────────

fn parity_int(codeword: &[u8]) -> u32 {
    let mut p = 0u32;
    for &b in &codeword[TPS_INFO_BITS..] {
        p = (p << 1) | (b & 1) as u32;
    }
    p
}

fn info_from_int(mut v: u64) -> Vec<u8> {
    // 53 bits, MSB (s1) first.
    let mut bits = vec![0u8; TPS_INFO_BITS];
    for i in (0..TPS_INFO_BITS).rev() {
        bits[i] = (v & 1) as u8;
        v >>= 1;
    }
    bits
}

#[test]
fn bch_lengths_are_standard() {
    assert_eq!(TPS_INFO_BITS, 53);
    assert_eq!(TPS_PARITY_BITS, 14);
    assert_eq!(TPS_CODEWORD_BITS, 67);
}

#[test]
fn bch_all_zero_is_all_zero() {
    let cw = tps_bch_encode(&[0u8; TPS_INFO_BITS]);
    assert!(cw.iter().all(|&b| b == 0), "0 info → 0 codeword");
}

#[test]
fn bch_known_answer_vectors() {
    // Independently computed from h(x)=0x4377 over GF(2^7), prim x^7+x^3+1:
    //   info = 1 (only s53 set)      → parity 0x377
    //   info = 53 ones               → parity 0x3cd1
    let cw1 = tps_bch_encode(&info_from_int(1));
    assert_eq!(parity_int(&cw1), 0x377, "parity for info=1");

    let cw_all = tps_bch_encode(&info_from_int((1u64 << 53) - 1));
    assert_eq!(parity_int(&cw_all), 0x3cd1, "parity for 53-ones");
}

#[test]
fn bch_no_error_decodes_identity() {
    for &seed in &[
        0x0u64,
        0x1,
        0x1_5555_5555_5555,
        0x1F_FFFF_FFFF_FFFF,
        0xA_BCDE_F012_3456,
    ] {
        let info = info_from_int(seed & ((1u64 << 53) - 1));
        let cw = tps_bch_encode(&info);
        let got = tps_bch_decode(&cw).expect("clean codeword decodes");
        assert_eq!(&got[..], &info[..], "seed {seed:#x}");
    }
}

#[test]
fn bch_corrects_single_error() {
    let info = info_from_int(0x1_2345_6789_ABCD & ((1u64 << 53) - 1));
    let cw = tps_bch_encode(&info);
    for flip in 0..TPS_CODEWORD_BITS {
        let mut rx = cw;
        rx[flip] ^= 1;
        let got = tps_bch_decode(&rx).unwrap_or_else(|| panic!("1 error at {flip} must correct"));
        assert_eq!(&got[..], &info[..], "single error at bit {flip}");
    }
}

#[test]
fn bch_corrects_double_error() {
    let info = info_from_int(0x0_DEAD_BEEF_1234 & ((1u64 << 53) - 1));
    let cw = tps_bch_encode(&info);
    // A representative spread of two-error patterns (exhaustive 67·66/2 is large;
    // sample widely including info/parity boundaries).
    for a in (0..TPS_CODEWORD_BITS).step_by(7) {
        for b in (a + 1..TPS_CODEWORD_BITS).step_by(5) {
            let mut rx = cw;
            rx[a] ^= 1;
            rx[b] ^= 1;
            let got =
                tps_bch_decode(&rx).unwrap_or_else(|| panic!("2 errors at {a},{b} must correct"));
            assert_eq!(&got[..], &info[..], "double error at {a},{b}");
        }
    }
}

#[test]
fn bch_rejects_triple_error() {
    // Three errors exceed t=2. The decoder must either fail (None) or return a
    // word that does not match the original — it must NOT silently miscorrect to
    // the original info. We assert it does not claim the original.
    let info = info_from_int(0x1_0F0F_0F0F_0F0F & ((1u64 << 53) - 1));
    let cw = tps_bch_encode(&info);
    let mut miscorrect_to_original = 0;
    let mut total = 0;
    for a in (0..TPS_CODEWORD_BITS).step_by(11) {
        for b in (a + 1..TPS_CODEWORD_BITS).step_by(9) {
            for c in (b + 1..TPS_CODEWORD_BITS).step_by(13) {
                let mut rx = cw;
                rx[a] ^= 1;
                rx[b] ^= 1;
                rx[c] ^= 1;
                total += 1;
                if let Some(got) = tps_bch_decode(&rx)
                    && got[..] == info[..]
                {
                    miscorrect_to_original += 1;
                }
            }
        }
    }
    // A t=2 decoder cannot recover 3 errors; it should never land back on the
    // exact original.
    assert_eq!(
        miscorrect_to_original, 0,
        "t=2 BCH silently recovered {miscorrect_to_original}/{total} triple-error words"
    );
}

// ── TPS word pack/unpack (EN 300 744 §4.6.2) ───────────────────────────────

#[test]
fn sync_words_are_standard() {
    // The two 16-bit sync words verbatim from §4.6.2.2.
    assert_eq!(TPS_SYNC_WORD_13, 0b0011_0101_1110_1110);
    assert_eq!(TPS_SYNC_WORD_24, 0b1100_1010_0001_0001);
    // They are bit-complements of each other.
    assert_eq!(TPS_SYNC_WORD_13, !TPS_SYNC_WORD_24);
}

fn sample_word(frame_number: u8) -> TpsWord {
    TpsWord {
        frame_number,
        constellation: ConstellationOrder::Qam16,
        code_rate_hp: PunctureRate::R3_4,
        guard: GuardInterval::G1_8,
        cell_id: 0xA5,
    }
}

#[test]
fn tps_word_pack_unpack_roundtrip() {
    for fn_num in 0..4u8 {
        for &c in &[
            ConstellationOrder::Qpsk,
            ConstellationOrder::Qam16,
            ConstellationOrder::Qam64,
        ] {
            for &r in &[
                PunctureRate::R1_2,
                PunctureRate::R2_3,
                PunctureRate::R3_4,
                PunctureRate::R5_6,
                PunctureRate::R7_8,
            ] {
                for &g in &[
                    GuardInterval::G1_32,
                    GuardInterval::G1_16,
                    GuardInterval::G1_8,
                    GuardInterval::G1_4,
                ] {
                    let w = TpsWord {
                        frame_number: fn_num,
                        constellation: c,
                        code_rate_hp: r,
                        guard: g,
                        cell_id: 0x3C,
                    };
                    let bits = w.pack();
                    assert_eq!(bits.len(), 68);
                    let got = TpsWord::unpack(&bits).expect("clean TPS word decodes");
                    assert_eq!(got, w, "roundtrip fn={fn_num} c={c:?} r={r:?} g={g:?}");
                }
            }
        }
    }
}

#[test]
fn tps_word_sync_matches_frame_parity() {
    // Frames 0,2 ("1,3") carry sync word 13; frames 1,3 ("2,4") carry word 24.
    for fn_num in 0..4u8 {
        let bits = sample_word(fn_num).pack();
        // s1..s16 are bits[1..17].
        let mut sync = 0u16;
        for &b in &bits[1..17] {
            sync = (sync << 1) | b as u16;
        }
        let expect = if fn_num % 2 == 0 {
            TPS_SYNC_WORD_13
        } else {
            TPS_SYNC_WORD_24
        };
        assert_eq!(sync, expect, "frame {fn_num} sync word");
    }
}

#[test]
fn tps_word_survives_two_bit_errors() {
    // The BCH protects the whole s1..s53 block, so up to 2 flipped TPS bits
    // (in that range) still recover the fields.
    let w = sample_word(2);
    let mut bits = w.pack();
    // Flip two info-region bits (indices 1..54 are s1..s53).
    bits[5] ^= 1;
    bits[40] ^= 1;
    let got = TpsWord::unpack(&bits).expect("2 errors correctable");
    assert_eq!(got, w);
}

#[test]
fn tps_word_s0_is_ignored() {
    // s0 (bits[0]) is the DBPSK reference slot, outside the BCH; flipping it must
    // not change the decoded fields.
    let w = sample_word(1);
    let mut bits = w.pack();
    bits[0] ^= 1;
    assert_eq!(TpsWord::unpack(&bits).unwrap(), w);
}

#[test]
fn tps_word_rejects_uncorrectable() {
    let w = sample_word(0);
    let mut bits = w.pack();
    // Four errors in the info region exceed t=2.
    for i in [3usize, 10, 20, 33] {
        bits[i + 1] ^= 1;
    }
    // Either None or a mismatch — never a false-accept as the original.
    match TpsWord::unpack(&bits) {
        None => {}
        Some(got) => assert_ne!(got, w, "4-error word must not decode to the original"),
    }
}

// ── TPS DBPSK along the symbol axis (EN 300 744 §4.6.3) ────────────────────

#[test]
fn tps_counts_are_standard() {
    assert_eq!(TPS_CARRIER_COUNT, 17);
    assert_eq!(TPS_SYMBOLS_PER_FRAME, 68);
}

#[test]
fn tps_dbpsk_encode_decode_noiseless() {
    // Pack a word, DBPSK it across 68 symbols on the 17 carriers, decode back.
    let w = TpsWord {
        frame_number: 2,
        constellation: ConstellationOrder::Qam64,
        code_rate_hp: PunctureRate::R2_3,
        guard: GuardInterval::G1_16,
        cell_id: 0x5A,
    };
    let block = w.pack();

    let mut enc = TpsEncoder::new();
    let mut dec = TpsDecoder::new();
    for &bit in block.iter() {
        let cells = enc.next_symbol(bit);
        // Each cell is real ±1 (normal power).
        for c in &cells {
            assert!((c.im).abs() < 1e-6);
            assert!((c.re.abs() - 1.0).abs() < 1e-6);
        }
        dec.feed_symbol(&cells);
    }
    assert!(dec.is_complete());
    // The differential bit stream s1..s67 must match; s0 is a don't-care slot.
    assert_eq!(&dec.bits()[1..], &block[1..], "DBPSK bit stream");
    assert_eq!(dec.word(), Some(w), "recovered TPS word");
}

#[test]
fn tps_dbpsk_survives_channel_phase() {
    // Differential detection needs no absolute reference: apply a fixed channel
    // phase + gain to every TPS cell and the word still decodes.
    let w = TpsWord {
        frame_number: 1,
        constellation: ConstellationOrder::Qpsk,
        code_rate_hp: PunctureRate::R1_2,
        guard: GuardInterval::G1_4,
        cell_id: 0,
    };
    let block = w.pack();
    let h = C32::from_polar(0.7, 1.1); // arbitrary channel

    let mut enc = TpsEncoder::new();
    let mut dec = TpsDecoder::new();
    for &bit in block.iter() {
        let cells = enc.next_symbol(bit);
        let rx: Vec<C32> = cells.iter().map(|&c| c * h).collect();
        dec.feed_symbol(&rx);
    }
    assert_eq!(
        dec.word(),
        Some(w),
        "decodes through a static channel phase"
    );
}

#[test]
fn tps_dbpsk_survives_awgn() {
    // DBPSK over 17 carriers averaged is robust; modest noise still decodes.
    let w = TpsWord {
        frame_number: 3,
        constellation: ConstellationOrder::Qam16,
        code_rate_hp: PunctureRate::R3_4,
        guard: GuardInterval::G1_8,
        cell_id: 0xC3,
    };
    let block = w.pack();
    // Simple deterministic pseudo-noise.
    let mut rng = 0x1234_5678u32;
    let mut noise = || {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((rng >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.6
    };

    let mut enc = TpsEncoder::new();
    let mut dec = TpsDecoder::new();
    for &bit in block.iter() {
        let cells = enc.next_symbol(bit);
        let rx: Vec<C32> = cells
            .iter()
            .map(|&c| c + C32::new(noise(), noise()))
            .collect();
        dec.feed_symbol(&rx);
    }
    assert_eq!(
        dec.word(),
        Some(w),
        "decodes under AWGN (BCH + 17-carrier avg)"
    );
}

#[test]
fn tps_encoder_reset_restarts_reference() {
    let mut enc = TpsEncoder::new();
    let first = enc.next_symbol(0);
    // Advance a few symbols, then reset — symbol 0 reference must return.
    enc.next_symbol(1);
    enc.next_symbol(1);
    enc.reset();
    let again = enc.next_symbol(0);
    assert_eq!(first, again, "reset restores the w_k reference signs");
}
