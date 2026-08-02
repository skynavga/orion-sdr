// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// End-to-end DVB-T (narrowband) roundtrip tests. The DVB-T-conformant pieces
// here are the **payload FEC chain** — K=7 convolutional inner + Forney
// (I=12, M=17) outer interleaver + RS(204,188) outer + exact DVB-T energy
// dispersal — and the **2K-mode carrier map** with continual pilots, at the
// amateur fs-scaled bandwidths (333 kHz / 1 MHz / 2 MHz). See
// `waveform::dvb_t::dvb_t_config`.
//
// NOT yet conformant in these Phase-1 tests (scaffolding via the existing COFDM
// frame layer): acquisition uses the crate's Schmidl & Cox **preamble** rather
// than DVB-T's guard-interval + scattered-pilot/TPS correlation (no preamble),
// and the frame carries the `OrionSdr` header rather than TPS. Those — and full
// DVB-T soft-decision through the shared demapper — land in Phase 3, which drops
// the preamble/header for the conformant on-air frame. A separate test below
// proves the DVB-T-*exact* constellation decodes end to end through an OFDM
// channel via hard-decision.

use crate::common::add_awgn;
use num_complex::Complex32 as C32;
use orion_sdr::core::Block;
use orion_sdr::demodulate::demodulate_frame;
use orion_sdr::fec::{FrameMetadata, FramePacket};
use orion_sdr::modulate::{OfdmConfig, OfdmFrameMod};
use orion_sdr::multicarrier::{CarrierGrid, CyclicPrefixRemove, FftBlock, GridExtract, GridMap};
use orion_sdr::sync::OfdmPreamble;
use orion_sdr::waveform::dvb_t::{
    GuardInterval, dvb_t_2k_plan, dvb_t_config, dvb_t_demap_symbol, dvb_t_map_symbol,
    dvb_t_mcs_table,
};

// A preamble sized to the 2K plan, with a training symbol for channel/CFO.
fn preamble(cfg: &OfdmConfig) -> OfdmPreamble {
    OfdmPreamble::new(4, 64)
        .with_training_symbol(cfg.carrier_plan.n_fft(), cfg.carrier_plan.cp_len())
}

fn strip_preamble(pre: &OfdmPreamble, iq: &[C32]) -> Vec<C32> {
    iq[pre.total_len()..].to_vec()
}

fn sample_payload(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 37 + 11) & 0xff) as u8).collect()
}

/// Full DVB-T frame roundtrip at a given occupied bandwidth (fs-scaled). One
/// RS(204,188) codeword of payload (188 − 4 CRC bytes) through the conformant
/// FEC chain + 2K carrier map + DVB-T energy dispersal.
fn dvb_t_frame_roundtrip(occupied_hz: f32, mcs_index: u8) {
    let cfg = dvb_t_config(GuardInterval::G1_32, occupied_hz);
    let pre = preamble(&cfg);
    let table = dvb_t_mcs_table();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(184);
    let frame = FramePacket::new(FrameMetadata::new(1, mcs_index), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let body = strip_preamble(modu.preamble(), &iq);
    let got = demodulate_frame(&cfg, &table, &body, None).expect("DVB-T frame decode");
    assert_eq!(got.payload, payload);
}

#[test]
fn roundtrip_dvb_t_2k_noiseless_333khz() {
    // MCS 0: QPSK rate 1/2 (the robust 333 kHz-class config).
    dvb_t_frame_roundtrip(333_000.0, 0);
}

#[test]
fn roundtrip_dvb_t_2k_noiseless_1mhz() {
    // MCS 1: QPSK rate 2/3 (general-purpose 1 MHz-class). fs is the only change.
    dvb_t_frame_roundtrip(1_000_000.0, 1);
}

#[test]
fn roundtrip_dvb_t_2k_noiseless_2mhz() {
    // MCS 2: 16-QAM rate 3/4 (wider 2 MHz-class).
    dvb_t_frame_roundtrip(2_000_000.0, 2);
}

#[test]
fn roundtrip_dvb_t_2k_awgn() {
    // QPSK rate 1/2 under modest AWGN — the concatenated FEC's coding gain.
    let cfg = dvb_t_config(GuardInterval::G1_32, 1_000_000.0);
    let pre = preamble(&cfg);
    let table = dvb_t_mcs_table();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(184);
    let frame = FramePacket::new(FrameMetadata::new(2, 0), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let mut body = strip_preamble(modu.preamble(), &iq);
    let sig_power: f32 = body.iter().map(|s| s.norm_sqr()).sum::<f32>() / body.len() as f32;
    add_awgn(&mut body, sig_power * 0.06, 0xD7B_C0DE);
    let got = demodulate_frame(&cfg, &table, &body, None).expect("DVB-T AWGN decode");
    assert_eq!(got.payload, payload);
}

// ── DVB-T-exact constellation through an OFDM channel (hard-decision) ───────

/// Maps `n_syms` DVB-T symbols (each `v` bits from `bits`) with the DVB-T-exact
/// constellation, runs them through the OFDM grid/IFFT → (optional AWGN) →
/// FFT/extract chain, hard-demaps with the DVB-T-exact demapper, and returns the
/// recovered bits. Proves the DVB-T mapping decodes end to end.
#[test]
fn dvb_t_qam_through_ofdm_channel() {
    for &(v, noise) in &[(2usize, 0.0f32), (4, 0.0), (2, 0.02)] {
        // A small plan is enough to exercise the mapping through OFDM.
        let plan = dvb_t_2k_plan(GuardInterval::G1_32);
        let n_data = plan.data_carriers().len();
        let grid = CarrierGrid::from_plan(&plan);
        let n_fft = plan.n_fft();
        let cp_len = plan.cp_len();

        // One OFDM symbol's worth of DVB-T-mapped data carriers.
        let mut r = 0x1234u32;
        let mut next = || {
            r = r.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (r >> 24) as u8 & 1
        };
        let bits: Vec<u8> = (0..n_data * v).map(|_| next()).collect();
        let syms: Vec<C32> = bits
            .chunks(v)
            .map(|c| dvb_t_map_symbol(c).unwrap())
            .collect();

        // Map onto the grid, IFFT, add CP.
        let mut freq = vec![C32::default(); n_fft];
        let mut gmap = GridMap::new(grid.clone());
        gmap.process(&syms, &mut freq);
        let mut time = vec![C32::default(); n_fft];
        orion_sdr::multicarrier::IfftBlock::new(n_fft).process(&freq, &mut time);
        let mut with_cp = vec![C32::default(); n_fft + cp_len];
        orion_sdr::multicarrier::CyclicPrefixInsert::new(n_fft, cp_len)
            .process(&time, &mut with_cp);

        if noise > 0.0 {
            let p: f32 = with_cp.iter().map(|s| s.norm_sqr()).sum::<f32>() / with_cp.len() as f32;
            add_awgn(&mut with_cp, p * noise, 0xC0DE_1234);
        }

        // RX: strip CP, FFT, extract data carriers, hard-demap.
        let mut rx_time = vec![C32::default(); n_fft];
        CyclicPrefixRemove::new(n_fft, cp_len).process(&with_cp, &mut rx_time);
        let mut rx_freq = vec![C32::default(); n_fft];
        FftBlock::new(n_fft).process(&rx_time, &mut rx_freq);
        let mut rx_syms = vec![C32::default(); n_data];
        GridExtract::new(grid).process(&rx_freq, &mut rx_syms);

        let recovered: Vec<u8> = rx_syms
            .iter()
            .flat_map(|&s| dvb_t_demap_symbol(s, v).unwrap())
            .collect();

        if noise == 0.0 {
            assert_eq!(recovered, bits, "v={v}: noiseless DVB-T QAM roundtrip");
        } else {
            let errs = recovered.iter().zip(&bits).filter(|(a, b)| a != b).count();
            let ber = errs as f32 / bits.len() as f32;
            assert!(
                ber < 0.01,
                "v={v} noise={noise}: DVB-T QAM BER {ber} too high"
            );
        }
    }
}
