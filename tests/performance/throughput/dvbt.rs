// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// Throughput benchmarks for the DVB-T / NB-DVB-T waveform: the 2K payload-FEC
// chain over the generic frame layer at each narrowband bandwidth mode, the
// conformant preamble-less frame (DvbTFrameMod/DvbTFrameDemod), and the four-frame
// super-frame (DvbTSuperFrameMod/DvbTSuperFrameDemod). The shared COFDM
// `frame_chain_with` driver is reused from `cofdm.rs` and the `random_bytes`
// helper from `fec.rs`; the FEC-block kernels themselves live in `fec.rs`.
//
// "Msps" is total frame samples / wall time (the sample-domain convention, not
// the info-bit convention `fec.rs` uses for its per-block kernels).

use super::cofdm::frame_chain_with;
use super::fec::random_bytes;
use super::{measure_throughput, minsps_from_env};
use num_complex::Complex32 as C32;
use orion_sdr::demodulate::{DvbTFrameDemod, DvbTFrameStreamDemod, DvbTSuperFrameDemod};
use orion_sdr::fec::PunctureRate as DvbPunctureRate;
use orion_sdr::modulate::{
    ConstellationOrder, DvbTFrameMod, DvbTSuperFrameMod, DvbTSuperFrameParams,
};
use orion_sdr::sync::OfdmPreamble;
use orion_sdr::waveform::dvb_t::{
    DvbTFrameParams, DvbTLinkParams, GuardInterval, NbBandwidth, dvb_t_config, dvb_t_mcs_table,
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
    // The full conformant preamble-less DVB-T frame (DvbTFrameMod ↔
    // DvbTFrameDemod): TS + energy dispersal + FEC + Figure-9a mapping + four-phase
    // scattered pilots + TPS + guard-interval acquisition. Measures TX
    // (DvbTFrameMod::modulate) and RX (DvbTFrameDemod::decode, incl. GI acquisition)
    // as end-to-end Msps. One RS codeword of TS payload; 68-symbol frame.
    let params = DvbTFrameParams {
        link: DvbTLinkParams {
            guard: GuardInterval::G1_32,
            constellation: ConstellationOrder::Qpsk,
            code_rate: DvbPunctureRate::R1_2,
        },
        frame_number: 0,
        cell_id: 0,
    };
    let modulator = DvbTFrameMod::new(params);
    let demod = DvbTFrameDemod::new(params);
    let payload = random_bytes(184, 0xD7B0);
    let frame = modulator.modulate(&payload);
    let frame_samples = frame.iq.len();
    let n_symbols = frame.n_symbols;

    // A buffer with lead-in silence so the RX must GI-acquire.
    let mut buf = vec![C32::default(); 200];
    buf.extend_from_slice(&frame.iq);
    buf.extend(vec![C32::default(); frame.samples_per_symbol]);

    let repeats = 50; // 2048-FFT × 68 symbols is heavy; keep the pass count modest

    let (mod_msps, mod_dt) = measure_throughput(
        || {
            let f = modulator.modulate(black_box(&payload));
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
            let got = demod
                .decode(black_box(&buf), n_symbols, 184)
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
            let f = modulator.modulate(black_box(&payload));
            let mut rt_buf = vec![C32::default(); 200];
            rt_buf.extend_from_slice(&f.iq);
            rt_buf.extend(vec![C32::default(); f.samples_per_symbol]);
            let got = demod
                .decode(black_box(&rt_buf), n_symbols, 184)
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
    // The conformant DVB-T super-frame (DvbTSuperFrameMod ↔ DvbTSuperFrameDemod):
    // four consecutive frames with alternating TPS sync + a 16-bit cell id split
    // across them. This is the single-frame path ×4 plus the multi-frame
    // sequencing/checks, so its cost tracks the conformant frame's — measured here
    // for completeness and as a regression floor.
    let params = DvbTSuperFrameParams {
        link: DvbTLinkParams {
            guard: GuardInterval::G1_32,
            constellation: ConstellationOrder::Qpsk,
            code_rate: DvbPunctureRate::R1_2,
        },
        cell_id: 0xBEEF,
    };
    let modulator = DvbTSuperFrameMod::new(params);
    let demod = DvbTSuperFrameDemod::new(params);
    let payload = random_bytes(700, 0x5F00);
    let sf = modulator.modulate(&payload);
    let frame_samples = sf.iq.len();
    let symbols_per_frame = sf.symbols_per_frame;
    let lens = sf.frame_payload_lens;

    let repeats = 20; // four 2048-FFT frames per pass; keep the pass count modest

    let (mod_msps, mod_dt) = measure_throughput(
        || {
            let s = modulator.modulate(black_box(&payload));
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
            let got = demod
                .decode(black_box(&sf.iq), symbols_per_frame, lens)
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
            let s = modulator.modulate(black_box(&payload));
            let got = demod
                .decode(black_box(&s.iq), s.symbols_per_frame, s.frame_payload_lens)
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

#[test]
fn throughput_dvb_t_stream_demod() {
    // The streaming DVB-T receiver (demodulate::dvb_t_stream): a continuous run of
    // frames pushed through `feed`, which GI-acquires and decodes each frame as its
    // samples arrive. Measures the decode side (buffer accumulation + repeated GI
    // search + per-frame drain) over a multi-frame stream; "Msps" is total stream
    // samples / wall time. The per-frame work is the batch RX's, plus the streaming
    // buffer management.
    let params = DvbTFrameParams {
        link: DvbTLinkParams {
            guard: GuardInterval::G1_32,
            constellation: ConstellationOrder::Qpsk,
            code_rate: DvbPunctureRate::R1_2,
        },
        frame_number: 0,
        cell_id: 0,
    };
    let payload = random_bytes(184, 0xD7B0);
    let one = DvbTFrameMod::new(params).modulate(&payload);
    let n_symbols = one.n_symbols;

    // A stream of NF back-to-back frames, no lead-in (each frame GI-locks at the
    // front of the residual buffer).
    const NF: usize = 4;
    let mut stream: Vec<C32> = Vec::with_capacity(one.iq.len() * NF + one.samples_per_symbol);
    for _ in 0..NF {
        stream.extend_from_slice(&one.iq);
    }
    stream.extend(vec![C32::default(); one.samples_per_symbol]);
    let stream_samples = stream.len();

    let repeats = 20; // NF × 68 symbols of 2048-FFT per pass; keep it modest

    let (demod_msps, demod_dt) = measure_throughput(
        || {
            let mut rx = DvbTFrameStreamDemod::new(black_box(params), n_symbols, 184);
            let mut got = rx.feed(black_box(&stream));
            got.extend(rx.flush());
            let decoded = got.iter().filter(|r| r.is_ok()).count();
            assert_eq!(decoded, NF, "stream decoded all frames");
            black_box(decoded);
            stream_samples
        },
        stream_samples,
        repeats,
    );
    println!("[DVB-T-Stream-Demod] {demod_msps:.4} Msps in {demod_dt:.3}s");
    let floor = minsps_from_env(0.02);
    assert!(demod_msps >= floor);
}

#[test]
fn throughput_dvb_t_integer_cfo() {
    // Cost of the demod's internal integer-CFO correction, now a construction-time
    // builder flag (`DvbTFrameDemod::with_integer_cfo_correction`). Measured as
    // (a) a plain demod (flag off) and (b) a demod with correction on — both over
    // the SAME clean lead-in buffer (no applied offset), so the delta is the added
    // cost of always running the estimate + rotate. This is the "as if wired in and
    // always on" figure: the estimate/rotate happens on every decode when enabled.
    let params = DvbTFrameParams {
        link: DvbTLinkParams {
            guard: GuardInterval::G1_32,
            constellation: ConstellationOrder::Qpsk,
            code_rate: DvbPunctureRate::R1_2,
        },
        frame_number: 0,
        cell_id: 0,
    };
    let modulator = DvbTFrameMod::new(params);
    let plain = DvbTFrameDemod::new(params);
    let corrected = DvbTFrameDemod::new(params).with_integer_cfo_correction(true);
    let payload = random_bytes(184, 0xD7B0);
    let frame = modulator.modulate(&payload);
    let frame_samples = frame.iq.len();
    let n_symbols = frame.n_symbols;

    // Lead-in silence so the RX must GI-acquire.
    let mut buf = vec![C32::default(); 200];
    buf.extend_from_slice(&frame.iq);
    buf.extend(vec![C32::default(); frame.samples_per_symbol]);

    let repeats = 50;

    // (a) Baseline: correction flag off.
    let (base_msps, base_dt) = measure_throughput(
        || {
            let got = plain
                .decode(black_box(&buf), n_symbols, 184)
                .expect("decode");
            black_box(got.payload[0]);
            frame_samples
        },
        frame_samples,
        repeats,
    );
    println!("[DVB-T-IntCFO-Baseline] {base_msps:.4} Msps in {base_dt:.3}s");

    // (b) Correction flag on: the demod runs the continual-pilot integer-CFO
    //     estimate before decoding, every call. The buffer is clean (offset 0), so
    //     the estimate returns k=0 and the rotate is skipped — this is the
    //     always-on overhead on a locked link (the normal operating case).
    let (cfo_msps, cfo_dt) = measure_throughput(
        || {
            let got = corrected
                .decode(black_box(&buf), n_symbols, 184)
                .expect("decode");
            black_box(got.payload[0]);
            frame_samples
        },
        frame_samples,
        repeats,
    );
    println!("[DVB-T-IntCFO-Corrected] {cfo_msps:.4} Msps in {cfo_dt:.3}s");
    let overhead = (cfo_dt / base_dt - 1.0) * 100.0; // >0 means corrected is slower
    println!(
        "[DVB-T-IntCFO-Overhead] {overhead:.1}% (baseline {base_msps:.2} -> corrected {cfo_msps:.2} Msps)"
    );
    let floor = minsps_from_env(0.02);
    assert!(base_msps >= floor && cfo_msps >= floor);
}
