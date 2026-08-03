// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// Throughput benchmarks for the COFDM frame (MAC) chain: the real
// `OfdmFrameMod::modulate_frame` + batch `OfdmFrameDemod::decode` per-link path, where
// the per-instance `CodecCache` amortizes FEC-code construction across frames.
// This is the frame/MAC layer over the OFDM modem (whose raw mod/demod
// throughput lives in `ofdm.rs`) and the concatenated FEC (whose per-block
// kernels live in `fec.rs`). The shared `frame_chain_with` driver is reused by
// the DVB-T benchmarks in `dvbt.rs`.
//
// "Msps" is total frame IQ samples / wall time (the sample-domain convention),
// matching the "COFDM frame throughput" table in docs/performance.md.

use super::fec::random_bytes;
use super::{measure_throughput, minsps_from_env};
use num_complex::Complex32 as C32;
use orion_sdr::demodulate::OfdmFrameDemod;
use orion_sdr::fec::{ConvCode, FrameMetadata, FramePacket, InnerFec, OuterFec, PunctureRate};
use orion_sdr::modulate::{ConstellationOrder, Mcs, McsTable, OfdmConfig, OfdmFrameMod};
use orion_sdr::multicarrier::CarrierPlan;
use orion_sdr::sync::OfdmPreamble;
use std::hint::black_box;

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
/// (`OfdmFrameDemod::decode` with a persistent cache) for one concatenation over
/// the default small (n_fft=64) plan, printing both Msps. `mcs_index` selects the
/// entry from `table` to exercise.
fn frame_chain(table: McsTable, mcs_index: u8, label: &str) {
    let cfg = frame_config();
    let pre = frame_preamble(&cfg);
    frame_chain_with(cfg, pre, table, mcs_index, 96, label);
}

/// Like [`frame_chain`], but over a caller-supplied `cfg`/`preamble` and payload
/// size — used to measure non-default plans (e.g. the DVB-T 2K carrier map, whose
/// benchmarks live in `dvbt.rs`).
pub(crate) fn frame_chain_with(
    cfg: OfdmConfig,
    pre: OfdmPreamble,
    table: McsTable,
    mcs_index: u8,
    payload_len: usize,
    label: &str,
) {
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let payload = random_bytes(payload_len, 0xF4A3);

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

    // Demodulate path: one batch demodulator whose persistent cache is reused
    // across all passes, so decode throughput is measured on the SAME warm-cache
    // footing as the modulate path (not dominated by per-call code construction).
    // Asserts correctness each pass.
    let demod = OfdmFrameDemod::new(cfg.clone(), table.clone());
    let (demod_msps, demod_dt) = measure_throughput(
        || {
            let got = demod.decode(&body).expect("decode");
            assert_eq!(got.payload, payload, "frame chain: payload recovered");
            black_box(got.payload[0]);
            frame_samples
        },
        frame_samples,
        repeats,
    );
    println!("[Frame-Chain-Demod {label}] {demod_msps:.4} Msps in {demod_dt:.3}s");

    // Roundtrip: modulate a fresh frame, strip the preamble, and decode it — the
    // full end-to-end path in one measured pass. "Msps" is frame samples / (mod +
    // demod) wall time, so it is lower than either single-direction figure.
    let pre_len = modu.preamble().total_len();
    let (rt_msps, rt_dt) = measure_throughput(
        || {
            seed = seed.wrapping_add(1);
            let iq = modu.modulate_frame(black_box(&frame), black_box(seed));
            let body: Vec<C32> = iq[pre_len..].to_vec();
            let got = demod.decode(&body).expect("decode");
            assert_eq!(
                got.payload, payload,
                "frame chain roundtrip: payload recovered"
            );
            black_box(got.payload[0]);
            frame_samples
        },
        frame_samples,
        repeats,
    );
    println!("[Frame-Chain-Roundtrip {label}] {rt_msps:.4} Msps in {rt_dt:.3}s");
    // Measurement run for the doc table; floor guards gross regressions only.
    let floor = minsps_from_env(0.05);
    assert!(mod_msps >= floor && demod_msps >= floor && rt_msps >= floor);
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
            code: ConvCode::K5,
        },
        OuterFec::ReedSolomon { n: 60, n_parity: 8 },
    )]);
    frame_chain(table, 0, "Conv+RS");
}
