// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// Throughput benchmarks for the DVB-T / NB-DVB-T waveform: the 2K payload-FEC
// chain over the generic frame layer at each narrowband bandwidth mode, the
// conformant preamble-less frame (modulate/demodulate::dvb_t_frame), and the
// four-frame super-frame (modulate/demodulate::dvb_t_super_frame). The shared
// COFDM `frame_chain_with` driver is reused from `cofdm.rs` and the `random_bytes`
// helper from `fec.rs`; the FEC-block kernels themselves live in `fec.rs`.
//
// "Msps" is total frame samples / wall time (the sample-domain convention, not
// the info-bit convention `fec.rs` uses for its per-block kernels).

use super::cofdm::frame_chain_with;
use super::fec::random_bytes;
use super::{measure_throughput, minsps_from_env};
use num_complex::Complex32 as C32;
use orion_sdr::demodulate::{dvb_t_frame_demodulate, dvb_t_super_frame_demodulate};
use orion_sdr::fec::PunctureRate as DvbPunctureRate;
use orion_sdr::modulate::{
    ConstellationOrder, DvbTSuperFrameParams, dvb_t_frame_modulate, dvb_t_super_frame_modulate,
};
use orion_sdr::sync::OfdmPreamble;
use orion_sdr::waveform::dvb_t::{
    DvbTFrameParams, GuardInterval, NbBandwidth, dvb_t_config, dvb_t_mcs_table,
};
use std::hint::black_box;

/// DVB-T 2K payload-FEC chain over the generic frame layer, for a named NB
/// bandwidth mode. K=7 conv + Forney(12,17) + RS(204,188) + energy dispersal over
/// the 2048-point 2K carrier map. The 2048-FFT dominates; floor-guarded.
fn dvb_t_2k_frame_chain(bw: NbBandwidth, label: &str) {
    let cfg = dvb_t_config(GuardInterval::G1_32, bw.occupied_hz());
    let pre = OfdmPreamble::new(4, 64)
        .with_training_symbol(cfg.carrier_plan.n_fft(), cfg.carrier_plan.cp_len());
    frame_chain_with(cfg, pre, dvb_t_mcs_table(), 1, 184, label);
}

#[test]
fn throughput_frame_chain_dvb_t_2k_333khz() {
    dvb_t_2k_frame_chain(NbBandwidth::Bw333kHz, "DVB-T-2K-333kHz");
}

#[test]
fn throughput_frame_chain_dvb_t_2k_1mhz() {
    dvb_t_2k_frame_chain(NbBandwidth::Bw1MHz, "DVB-T-2K-1MHz");
}

#[test]
fn throughput_frame_chain_dvb_t_2k_2mhz() {
    dvb_t_2k_frame_chain(NbBandwidth::Bw2MHz, "DVB-T-2K-2MHz");
}

#[test]
fn throughput_dvb_t_conformant_frame() {
    // The full conformant preamble-less DVB-T frame (modulate::dvb_t_frame ↔
    // demodulate::dvb_t_frame): TS + energy dispersal + FEC + Figure-9a mapping +
    // four-phase scattered pilots + TPS + guard-interval acquisition. Measures TX
    // (dvb_t_frame_modulate) and RX (dvb_t_frame_demodulate, incl. GI acquisition)
    // as end-to-end Msps. One RS codeword of TS payload; 68-symbol frame.
    let params = DvbTFrameParams {
        guard: GuardInterval::G1_32,
        constellation: ConstellationOrder::Qpsk,
        code_rate: DvbPunctureRate::R1_2,
        frame_number: 0,
        cell_id: 0,
    };
    let payload = random_bytes(184, 0xD7B0);
    let frame = dvb_t_frame_modulate(params, &payload);
    let frame_samples = frame.iq.len();
    let n_symbols = frame.n_symbols;

    // A buffer with lead-in silence so the RX must GI-acquire.
    let mut buf = vec![C32::default(); 200];
    buf.extend_from_slice(&frame.iq);
    buf.extend(vec![C32::default(); frame.samples_per_symbol]);

    let repeats = 50; // 2048-FFT × 68 symbols is heavy; keep the pass count modest

    let (mod_msps, mod_dt) = measure_throughput(
        || {
            let f = dvb_t_frame_modulate(black_box(params), black_box(&payload));
            let mut acc = 0.0f32;
            for s in &f.iq {
                acc += s.re;
            }
            black_box(acc);
            frame_samples
        },
        frame_samples,
        repeats,
    );
    println!("[DVB-T-Frame-Mod] {mod_msps:.4} Msps in {mod_dt:.3}s");

    let (demod_msps, demod_dt) = measure_throughput(
        || {
            let got = dvb_t_frame_demodulate(black_box(params), black_box(&buf), n_symbols, 184)
                .expect("conformant DVB-T decode");
            assert_eq!(got.payload, payload, "conformant frame payload recovered");
            black_box(got.payload[0]);
            frame_samples
        },
        frame_samples,
        repeats,
    );
    println!("[DVB-T-Frame-Demod] {demod_msps:.4} Msps in {demod_dt:.3}s");

    // Roundtrip: modulate a fresh frame, place it after lead-in silence, and
    // GI-acquire + decode it — the full conformant path in one measured pass.
    // "Msps" is frame samples / (mod + demod) wall time.
    let (rt_msps, rt_dt) = measure_throughput(
        || {
            let f = dvb_t_frame_modulate(black_box(params), black_box(&payload));
            let mut rt_buf = vec![C32::default(); 200];
            rt_buf.extend_from_slice(&f.iq);
            rt_buf.extend(vec![C32::default(); f.samples_per_symbol]);
            let got = dvb_t_frame_demodulate(black_box(params), black_box(&rt_buf), n_symbols, 184)
                .expect("conformant DVB-T roundtrip decode");
            assert_eq!(
                got.payload, payload,
                "conformant roundtrip payload recovered"
            );
            black_box(got.payload[0]);
            frame_samples
        },
        frame_samples,
        repeats,
    );
    println!("[DVB-T-Frame-Roundtrip] {rt_msps:.4} Msps in {rt_dt:.3}s");
    let floor = minsps_from_env(0.02);
    assert!(mod_msps >= floor && demod_msps >= floor && rt_msps >= floor);
}

#[test]
fn throughput_dvb_t_super_frame() {
    // The conformant DVB-T super-frame (modulate::dvb_t_super_frame ↔
    // demodulate::dvb_t_super_frame): four consecutive frames with alternating TPS
    // sync + a 16-bit cell id split across them. This is the single-frame path ×4
    // plus the multi-frame sequencing/checks, so its cost tracks the conformant
    // frame's — measured here for completeness and as a regression floor.
    let params = DvbTSuperFrameParams {
        guard: GuardInterval::G1_32,
        constellation: ConstellationOrder::Qpsk,
        code_rate: DvbPunctureRate::R1_2,
        cell_id: 0xBEEF,
    };
    let payload = random_bytes(700, 0x5F00);
    let sf = dvb_t_super_frame_modulate(params, &payload);
    let frame_samples = sf.iq.len();
    let symbols_per_frame = sf.symbols_per_frame;
    let lens = sf.frame_payload_lens;

    let repeats = 20; // four 2048-FFT frames per pass; keep the pass count modest

    let (mod_msps, mod_dt) = measure_throughput(
        || {
            let s = dvb_t_super_frame_modulate(black_box(params), black_box(&payload));
            let mut acc = 0.0f32;
            for x in &s.iq {
                acc += x.re;
            }
            black_box(acc);
            frame_samples
        },
        frame_samples,
        repeats,
    );
    println!("[DVB-T-SuperFrame-Mod] {mod_msps:.4} Msps in {mod_dt:.3}s");

    let (demod_msps, demod_dt) = measure_throughput(
        || {
            let got = dvb_t_super_frame_demodulate(
                black_box(params),
                black_box(&sf.iq),
                symbols_per_frame,
                lens,
            )
            .expect("super-frame decode");
            assert_eq!(got.payload, payload, "super-frame payload recovered");
            black_box(got.payload[0]);
            frame_samples
        },
        frame_samples,
        repeats,
    );
    println!("[DVB-T-SuperFrame-Demod] {demod_msps:.4} Msps in {demod_dt:.3}s");

    let (rt_msps, rt_dt) = measure_throughput(
        || {
            let s = dvb_t_super_frame_modulate(black_box(params), black_box(&payload));
            let got = dvb_t_super_frame_demodulate(
                black_box(params),
                black_box(&s.iq),
                s.symbols_per_frame,
                s.frame_payload_lens,
            )
            .expect("super-frame roundtrip decode");
            assert_eq!(
                got.payload, payload,
                "super-frame roundtrip payload recovered"
            );
            black_box(got.payload[0]);
            frame_samples
        },
        frame_samples,
        repeats,
    );
    println!("[DVB-T-SuperFrame-Roundtrip] {rt_msps:.4} Msps in {rt_dt:.3}s");
    let floor = minsps_from_env(0.02);
    assert!(mod_msps >= floor && demod_msps >= floor && rt_msps >= floor);
}
