// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use orion_sdr::fec::{
    CrcKind, HeaderFormat, InnerFec, InterleaverKind, LdpcCode, OuterFec, ScramblerKind,
    ScramblerPos, SeedMode,
};
use orion_sdr::modulate::ofdm_frame::{
    BCH_INFO_BITS, HEADER_FIELD_BYTES, append_crc, bits_to_bytes, block_plan, bytes_to_bits,
    check_and_strip_crc, pack_header_fields,
};
use orion_sdr::modulate::{ConstellationOrder, FrameConfigError, Mcs, McsTable, OfdmConfig};

fn base_config() -> OfdmConfig {
    let plan =
        orion_sdr::multicarrier::CarrierPlan::new(64, 8).with_data_carriers(vec![1, 2, 3, -1, -2]);
    OfdmConfig::new(plan, 48_000.0, 0.0, 1.0, ConstellationOrder::Bpsk)
}

// ── Bit/byte helpers ───────────────────────────────────────────────────────

#[test]
fn bytes_bits_round_trip() {
    let bytes = vec![0x00u8, 0xFF, 0xA5, 0x3C, 0x81];
    let bits = bytes_to_bits(&bytes);
    assert_eq!(bits.len(), bytes.len() * 8);
    // MSB-first: 0xA5 = 1010_0101
    assert_eq!(&bits[16..24], &[1, 0, 1, 0, 0, 1, 0, 1]);
    assert_eq!(bits_to_bytes(&bits), bytes);
}

// ── Header field packing ───────────────────────────────────────────────────

#[test]
fn header_fields_pack_big_endian() {
    let f = pack_header_fields(0x07, 0x0001_0203, 0x0405_0607, 0x09, 0x0A0B_0C0D);
    assert_eq!(f.len(), HEADER_FIELD_BYTES);
    assert_eq!(f[0], 0x07, "mcs_index");
    assert_eq!(&f[1..5], &[0x00, 0x01, 0x02, 0x03], "payload_len BE");
    assert_eq!(&f[5..9], &[0x04, 0x05, 0x06, 0x07], "sequence_num BE");
    assert_eq!(f[9], 0x09, "flags");
    assert_eq!(&f[10..14], &[0x0A, 0x0B, 0x0C, 0x0D], "scrambler_seed BE");
}

// ── CRC dispatch ───────────────────────────────────────────────────────────

#[test]
fn crc_kind_lengths() {
    assert_eq!(CrcKind::None.len_bytes(), 0);
    assert_eq!(CrcKind::Crc16.len_bytes(), 2);
    assert_eq!(CrcKind::Crc32.len_bytes(), 4);
}

#[test]
fn append_and_check_crc_round_trip() {
    let data = b"frame payload bytes".to_vec();
    for crc in [CrcKind::None, CrcKind::Crc16, CrcKind::Crc32] {
        let framed = append_crc(crc, &data);
        assert_eq!(framed.len(), data.len() + crc.len_bytes());
        let (recovered, ok) = check_and_strip_crc(crc, &framed).unwrap();
        assert_eq!(recovered, data);
        assert!(ok, "clean CRC must verify for {crc:?}");
    }
}

#[test]
fn crc_detects_corruption() {
    let data = b"important".to_vec();
    for crc in [CrcKind::Crc16, CrcKind::Crc32] {
        let mut framed = append_crc(crc, &data);
        framed[0] ^= 0x01; // corrupt a payload byte
        let (_, ok) = check_and_strip_crc(crc, &framed).unwrap();
        assert!(!ok, "{crc:?} must flag corruption");
    }
}

// ── MCS table ──────────────────────────────────────────────────────────────

#[test]
fn mcs_table_lookup() {
    let t = McsTable::default_ladder();
    assert_eq!(t.len(), 4);
    assert_eq!(t.get(0).unwrap().constellation, ConstellationOrder::Bpsk);
    assert_eq!(t.get(3).unwrap().constellation, ConstellationOrder::Qam64);
    assert!(t.get(99).is_none(), "out-of-range index returns None");
}

#[test]
fn mcs_custom_table() {
    let t = McsTable::new(vec![Mcs::new(
        ConstellationOrder::Qpsk,
        InnerFec::Ldpc(LdpcCode::N512R12),
        OuterFec::None,
    )]);
    let m = t.get(0).unwrap();
    assert_eq!(m.constellation, ConstellationOrder::Qpsk);
    assert_eq!(m.outer_fec, OuterFec::None);
}

// ── Block plan ─────────────────────────────────────────────────────────────

#[test]
fn block_plan_no_coding_is_bits() {
    // No CRC, no FEC, no interleave: coded_bits == info bytes * 8.
    let p = block_plan(
        10,
        CrcKind::None,
        OuterFec::None,
        InnerFec::None,
        InterleaverKind::None,
        InterleaverKind::None,
    );
    assert_eq!(p.framed_bytes, 10);
    assert_eq!(p.coded_bits, 80);
}

#[test]
fn block_plan_ldpc_bch_fragments() {
    // With CRC-32 + BCH(t=8) outer + LDPC rate-1/2 inner, sizes must be exact
    // multiples of the codeword sizes.
    let p = block_plan(
        40,
        CrcKind::Crc32,
        OuterFec::Bch { t: 8 },
        InnerFec::Ldpc(LdpcCode::N512R12),
        InterleaverKind::None,
        InterleaverKind::None,
    );
    assert_eq!(p.framed_bytes, 44);
    // outer BCH blocks are BCH_INFO_BITS info each; coded is a whole number of
    // BCH codewords.
    let framed_bits = 44usize * 8;
    let n_bch = framed_bits.div_ceil(BCH_INFO_BITS);
    assert!(p.outer_coded_bits >= framed_bits);
    assert_eq!(p.outer_coded_bits % n_bch, 0, "whole BCH codewords");
    // inner LDPC: coded_bits a multiple of N=512.
    assert_eq!(p.coded_bits % 512, 0, "whole LDPC codewords");
}

// ── OfdmConfig::validate ───────────────────────────────────────────────────

#[test]
fn validate_accepts_bare_defaults() {
    assert!(base_config().validate().is_ok());
}

#[test]
fn validate_rejects_per_frame_seed_without_header() {
    let cfg = base_config()
        .with_header_format(HeaderFormat::NoHeader)
        .with_scrambler(ScramblerKind::Additive {
            poly: 0b1001,
            width: 7,
            seed: SeedMode::PerFrameRandom,
        });
    assert_eq!(
        cfg.validate(),
        Err(FrameConfigError::PerFrameSeedNeedsHeader)
    );
}

#[test]
fn validate_allows_per_frame_seed_with_header() {
    let cfg = base_config().with_scrambler(ScramblerKind::Additive {
        poly: 0b1001,
        width: 7,
        seed: SeedMode::PerFrameRandom,
    });
    assert!(cfg.validate().is_ok());
}

#[test]
fn validate_rejects_zero_interleaver_dim() {
    let cfg = base_config().with_inner_interleaver(InterleaverKind::Block { rows: 0, cols: 8 });
    assert_eq!(cfg.validate(), Err(FrameConfigError::ZeroInterleaverDim));
}

#[test]
fn validate_rejects_zero_bch_t() {
    let cfg = base_config().with_outer_fec(OuterFec::Bch { t: 0 });
    assert_eq!(cfg.validate(), Err(FrameConfigError::ZeroBchT));
}

// ── Builder methods leave the base symbol pipeline untouched ────────────────

#[test]
fn builders_do_not_disturb_symbol_dimensions() {
    let bare = base_config();
    let framed = base_config()
        .with_inner_fec(InnerFec::Ldpc(LdpcCode::N512R12))
        .with_payload_crc(CrcKind::Crc32)
        .with_scrambler_pos(ScramblerPos::AfterInnerFec);
    // The per-symbol dimensions come from the plan + constellation only.
    assert_eq!(bare.bits_per_ofdm_symbol(), framed.bits_per_ofdm_symbol());
    assert_eq!(
        bare.samples_per_ofdm_symbol(),
        framed.samples_per_ofdm_symbol()
    );
}
