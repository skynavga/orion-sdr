// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::common::add_awgn;
use num_complex::Complex32 as C32;
use orion_sdr::demodulate::demodulate_frame;
use orion_sdr::fec::{
    CrcKind, FrameMetadata, FramePacket, HeaderFormat, InnerFec, InterleaverKind, OuterFec,
    ScramblerKind, ScramblerPos, SeedMode,
};
use orion_sdr::modulate::{ConstellationOrder, McsTable, OfdmConfig, OfdmFrameMod};
use orion_sdr::sync::OfdmPreamble;

fn plan_config() -> OfdmConfig {
    // A 64-point plan with all non-DC, non-Nyquist carriers as data.
    let n_fft = 64;
    let cp_len = 8;
    let half = (n_fft / 2) as i32;
    let data: Vec<i32> = (1..half).chain(-(half - 1)..0).collect();
    let plan = orion_sdr::multicarrier::CarrierPlan::new(n_fft, cp_len).with_data_carriers(data);
    OfdmConfig::new(plan, 48_000.0, 0.0, 1.0, ConstellationOrder::Bpsk)
}

fn preamble(cfg: &OfdmConfig) -> OfdmPreamble {
    OfdmPreamble::new(4, 16)
        .with_training_symbol(cfg.carrier_plan.n_fft(), cfg.carrier_plan.cp_len())
}

fn strip_preamble(cfg: &OfdmConfig, pre: &OfdmPreamble, iq: &[C32]) -> Vec<C32> {
    // demodulate_frame expects IQ starting at the first post-preamble sample.
    let _ = cfg;
    iq[pre.total_len()..].to_vec()
}

fn sample_payload(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 37 + 11) & 0xff) as u8).collect()
}

#[test]
fn roundtrip_frame_noiseless_ldpc_bch() {
    let cfg = plan_config();
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    for mcs_index in 0..table.len() as u8 {
        let payload = sample_payload(40);
        let frame = FramePacket::new(
            FrameMetadata::new(0x1234_5678 + mcs_index as u32, mcs_index),
            payload.clone(),
        );
        let iq = modu.modulate_frame(&frame, 0);
        let body = strip_preamble(&cfg, modu.preamble(), &iq);
        let got = demodulate_frame(&cfg, &table, &body).expect("decode");
        assert_eq!(got.payload, payload, "mcs {mcs_index}: payload");
        assert_eq!(got.metadata.mcs_index, mcs_index);
        assert_eq!(got.metadata.sequence_num, 0x1234_5678 + mcs_index as u32);
    }
}

#[test]
fn roundtrip_frame_awgn() {
    let cfg = plan_config();
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    // MCS 0 (BPSK + rate-1/2 LDPC + BCH) is the most robust. Noise is scaled
    // relative to the signal's own (low) per-sample power, as the uncoded OFDM
    // test does — but at 10%, twice the uncoded test's 5%, since the FEC here
    // buys coding gain the bare pipeline doesn't have.
    let payload = sample_payload(32);
    let frame = FramePacket::new(FrameMetadata::new(7, 0), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let mut body = strip_preamble(&cfg, modu.preamble(), &iq);
    let sig_power: f32 = body.iter().map(|s| s.norm_sqr()).sum::<f32>() / body.len() as f32;
    add_awgn(&mut body, sig_power * 0.10, 0xC0FFEE);
    let got = demodulate_frame(&cfg, &table, &body).expect("decode under AWGN");
    assert_eq!(got.payload, payload);
}

#[test]
fn frame_header_crc_catches_corruption() {
    let cfg = plan_config();
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let frame = FramePacket::new(FrameMetadata::new(1, 0), sample_payload(16));
    let iq = modu.modulate_frame(&frame, 0);
    let mut body = strip_preamble(&cfg, modu.preamble(), &iq);
    // Massively corrupt the header region (first few symbols) beyond FEC.
    for s in body.iter_mut().take(200) {
        *s = C32::new(-s.re, -s.im);
    }
    let res = demodulate_frame(&cfg, &table, &body);
    assert!(
        res.is_err(),
        "corrupted header must not decode to a valid frame"
    );
}

#[test]
fn roundtrip_frame_no_header_rejected_by_batch() {
    // NoHeader is not supported by the batch entry point (needs out-of-band MCS).
    let cfg = plan_config().with_header_format(HeaderFormat::NoHeader);
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let frame = FramePacket::new(FrameMetadata::new(1, 0), sample_payload(8));
    let iq = modu.modulate_frame(&frame, 0);
    let body = strip_preamble(&cfg, modu.preamble(), &iq);
    assert!(demodulate_frame(&cfg, &table, &body).is_err());
}

#[test]
fn roundtrip_frame_scrambler_positions() {
    let table = McsTable::default_ladder();
    for pos in [ScramblerPos::BeforeOuterFec, ScramblerPos::AfterInnerFec] {
        let cfg = plan_config()
            .with_scrambler(ScramblerKind::Additive {
                poly: 0b1001,
                width: 7,
                seed: SeedMode::Fixed(0x7F),
            })
            .with_scrambler_pos(pos);
        let pre = preamble(&cfg);
        let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
        let payload = sample_payload(24);
        let frame = FramePacket::new(FrameMetadata::new(2, 0), payload.clone());
        let iq = modu.modulate_frame(&frame, 0);
        let body = strip_preamble(&cfg, modu.preamble(), &iq);
        let got = demodulate_frame(&cfg, &table, &body).expect("decode with scrambler");
        assert_eq!(got.payload, payload, "scrambler pos {pos:?}");
    }
}

#[test]
fn roundtrip_frame_per_frame_random_seed() {
    let cfg = plan_config().with_scrambler(ScramblerKind::Additive {
        poly: 0b1001,
        width: 7,
        seed: SeedMode::PerFrameRandom,
    });
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let payload = sample_payload(20);
    let frame = FramePacket::new(FrameMetadata::new(3, 0), payload.clone());
    let seed = 0xABCD_1234u32;
    let iq = modu.modulate_frame(&frame, seed);
    let body = strip_preamble(&cfg, modu.preamble(), &iq);
    let got = demodulate_frame(&cfg, &table, &body).expect("decode with per-frame seed");
    assert_eq!(got.payload, payload);
}

#[test]
fn roundtrip_frame_with_interleavers() {
    let cfg = plan_config()
        .with_inner_interleaver(InterleaverKind::Block { rows: 8, cols: 16 })
        .with_outer_interleaver(InterleaverKind::Block { rows: 4, cols: 8 });
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let payload = sample_payload(30);
    let frame = FramePacket::new(FrameMetadata::new(4, 1), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let body = strip_preamble(&cfg, modu.preamble(), &iq);
    let got = demodulate_frame(&cfg, &table, &body).expect("decode with interleavers");
    assert_eq!(got.payload, payload);
}

#[test]
fn roundtrip_frame_no_fec_no_crc() {
    // Bare frame: no FEC, no CRC on payload — still round-trips noiselessly.
    let cfg = plan_config().with_payload_crc(CrcKind::None);
    let pre = preamble(&cfg);
    let table = McsTable::new(vec![orion_sdr::modulate::Mcs::new(
        ConstellationOrder::Qpsk,
        InnerFec::None,
        OuterFec::None,
    )]);
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let payload = sample_payload(16);
    let frame = FramePacket::new(FrameMetadata::new(5, 0), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let body = strip_preamble(&cfg, modu.preamble(), &iq);
    let got = demodulate_frame(&cfg, &table, &body).expect("decode bare frame");
    assert_eq!(got.payload, payload);
}
