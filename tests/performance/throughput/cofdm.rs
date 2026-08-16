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
use orion_sdr::demodulate::{OfdmFrameDemod, OfdmFrameStreamDemod, OfdmRxProbe};
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

// ── Streaming receiver: the RX probe's cost, on and off ────────────────────

/// Measures the full streaming receive path — sync, CFO correction,
/// training-symbol equalization, and the concatenated decode — in three
/// configurations, so the probe's cost is separable from the encode chain it
/// shares with `with_error_rates`:
///
/// 1. `feed`, nothing opted in — the baseline a caller who does not want
///    diagnostics pays. The probe adds no branch here by construction: it is a
///    different method, not a flag.
/// 2. `feed` with `with_error_rates(true)` — one whole encode chain per frame.
/// 3. `feed_probed` — the same encode chain, plus one inner re-encode to
///    re-derive what the decoder decided, plus two buffer fills and an XOR pass.
///
/// Backs the "COFDM RX probe cost" row in docs/performance.md.
fn stream_probe_cost(table: McsTable, mcs_index: u8, label: &str) {
    let cfg = frame_config();
    let pre = frame_preamble(&cfg);
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let payload = random_bytes(96, 0xF4A3);
    let frame = FramePacket::new(FrameMetadata::new(0x2468, mcs_index), payload.clone());

    // Lead-in silence, one frame, trailing silence — what the sync search sees.
    let mut iq = vec![C32::default(); 24];
    iq.extend_from_slice(&modu.modulate_frame(&frame, 0));
    iq.extend(vec![C32::default(); 64]);
    let frame_samples = iq.len();
    // 2000, not the 200 the batch benchmarks use: at 200 the run is ~0.08 s and
    // the clock ramp dominates, which made the *baseline* read 18% slower than
    // the configurations doing strictly more work.
    let repeats = 2000;

    // The buffer is cleared between passes so the residual padding cannot
    // accumulate and drift the sync-search cost across the run.
    let measure =
        |mut rx: OfdmFrameStreamDemod, mut probe: Option<OfdmRxProbe>, tag: &str| -> f32 {
            let (msps, dt) = measure_throughput(
                || {
                    rx.clear();
                    let out = match probe.as_mut() {
                        Some(p) => rx.feed_probed(black_box(&iq), p),
                        None => rx.feed(black_box(&iq)),
                    };
                    assert_eq!(out.len(), 1, "{tag}: one frame per pass");
                    let got = out[0].as_ref().expect("decode");
                    assert_eq!(got.packet.payload, payload, "{tag}: payload recovered");
                    black_box(got.packet.payload[0]);
                    if let Some(p) = probe.as_ref() {
                        black_box(p.frames().len());
                    }
                    frame_samples
                },
                frame_samples,
                repeats,
            );
            println!("[Frame-Stream-{tag} {label}] {msps:.4} Msps in {dt:.3}s");
            msps
        };

    let plain = measure(
        OfdmFrameStreamDemod::new(cfg.clone(), table.clone(), pre),
        None,
        "Feed",
    );
    let with_ber = measure(
        OfdmFrameStreamDemod::new(cfg.clone(), table.clone(), pre).with_error_rates(true),
        None,
        "Feed-ErrorRates",
    );
    let probed = measure(
        OfdmFrameStreamDemod::new(cfg, table, pre),
        Some(OfdmRxProbe::new()),
        "FeedProbed",
    );

    let pct = |m: f32| 100.0 * (plain / m - 1.0);
    println!(
        "[Frame-Stream-Overhead {label}] error-rates {:+.1}%, probe {:+.1}%",
        pct(with_ber),
        pct(probed)
    );

    let floor = minsps_from_env(0.05);
    assert!(plain >= floor && with_ber >= floor && probed >= floor);
    // Measured on this machine: baseline ~7.5 Msps, error rates +2.5..3.0%,
    // probe +3.1..4.2% — i.e. the probe costs about a point over the encode
    // chain it shares with `with_error_rates`. The guard is deliberately loose:
    // it is here to catch a per-frame allocation storm or a lost buffer reuse,
    // not to police a few percent of timing noise.
    assert!(
        probed >= 0.75 * plain,
        "probing costs {:.1}% of the receive path, far past the ~4% budget",
        100.0 * (plain / probed - 1.0)
    );
}

#[test]
fn throughput_frame_stream_probe_ldpc_bch() {
    stream_probe_cost(McsTable::default_ladder(), 1, "LDPC+BCH");
}
