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
use orion_sdr::demodulate::{
    DvbTFrameDemod, DvbTFrameStreamDemod, DvbTRxProbe, DvbTSuperFrameDemod,
};
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

/// What the RX diagnostics ladder costs, for one payload size. Four
/// configurations over the same buffer:
///
///   (a) plain            — the default demod. Every *free* rung (CFO, sync
///                          score, timing offset, integer-CFO bins, outer-FEC
///                          verdict, RS corrected bytes) is populated here,
///                          because each is read straight off work the decode
///                          already does.
///   (b) with_error_rates — adds EVM (an ideal-point remap per data carrier),
///                          the two BERs (one shared encode chain), and a
///                          WHOLE-FRAME FEC decode.
///   (c) plain again      — a repeat of (a) at the end of the run.
///   (d) feed_probed      — the streaming probe, which needs the same
///                          whole-frame truth plus two buffer fills.
///
/// (c) exists because of the measurement trap `ofdm-rx-probe.md` hit: on a
/// ramping CPU the FIRST configuration measured looks slower than ones doing
/// strictly more work, and at 200 passes that artifact was 18%. Comparing (a)
/// against (c) bounds the drift; the real overhead is (b) against the BETTER of
/// the two plain runs, which cannot flatter the gated path.
///
/// **The payload size is the whole story, which is why this is parameterized.**
/// A DVB-T frame is a fixed 68 OFDM symbols — the TPS block — and the modulator
/// stuffs null packets to fill whatever the payload leaves empty. A plain decode
/// reads only the payload's prefix; an exact re-encode needs everything that was
/// transmitted, stuffing included, because Forney(12,17) couples the coded bits
/// across the whole stream. So the gated cost is set by the FILL RATIO, and
/// measuring one payload size measures nothing general. The generic COFDM path
/// never shows this: it sizes the frame to the payload, so its prefix is already
/// the whole block and `with_error_rates` costs it only the encode chain (~3.3%,
/// `throughput_frame_stream_probe_ldpc_bch`).
fn dvb_t_diagnostics_cost(payload_len: usize, repeats: usize, label: &str) {
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
    let measured = DvbTFrameDemod::new(params).with_error_rates(true);
    let payload = random_bytes(payload_len, 0xD1A6);
    let frame = modulator.modulate(&payload);
    let frame_samples = frame.iq.len();
    let n_symbols = frame.n_symbols;

    let mut buf = vec![C32::default(); 200];
    buf.extend_from_slice(&frame.iq);
    buf.extend(vec![C32::default(); frame.samples_per_symbol]);

    // Warm-up, unmeasured: pulls the clock up before the first timed run so the
    // ordering artifact above is bounded rather than merely detected.
    for _ in 0..10 {
        black_box(
            plain
                .decode(black_box(&buf), n_symbols, payload_len)
                .expect("decode"),
        );
        black_box(
            measured
                .decode(black_box(&buf), n_symbols, payload_len)
                .expect("decode"),
        );
    }

    let run = |demod: &DvbTFrameDemod| {
        measure_throughput(
            || {
                let got = demod
                    .decode(black_box(&buf), n_symbols, payload_len)
                    .expect("decode");
                black_box(got.payload[0]);
                frame_samples
            },
            frame_samples,
            repeats,
        )
    };

    let (plain_msps, plain_dt) = run(&plain);
    println!("[DVB-T-Diag-Plain {label}] {plain_msps:.4} Msps in {plain_dt:.3}s");
    let (meas_msps, meas_dt) = run(&measured);
    println!("[DVB-T-Diag-ErrorRates {label}] {meas_msps:.4} Msps in {meas_dt:.3}s");
    let (plain2_msps, plain2_dt) = run(&plain);
    println!("[DVB-T-Diag-Plain-Repeat {label}] {plain2_msps:.4} Msps in {plain2_dt:.3}s");

    let drift = (plain_dt / plain2_dt - 1.0) * 100.0;
    println!("[DVB-T-Diag-PlainDrift {label}] {drift:.1}% (first plain run vs repeat)");
    // Against the FASTER plain run, so the gated path is never flattered by
    // drift in its favour.
    let best_plain_dt = plain_dt.min(plain2_dt);
    let overhead = (meas_dt / best_plain_dt - 1.0) * 100.0;
    println!(
        "[DVB-T-Diag-Overhead {label}] {overhead:.1}% (plain {plain_msps:.2} -> error-rates {meas_msps:.2} Msps)"
    );

    // (d) The streaming probe. Measured on the stream receiver because that is
    //     the only place it is offered, so this row carries the stream's
    //     GI-search and buffer drain as well — read it against the probe-off
    //     stream row beside it, not against the batch numbers above.
    let mut probe = DvbTRxProbe::new();
    let (stream_msps, stream_dt) = measure_throughput(
        || {
            let mut rx = DvbTFrameStreamDemod::new(params, n_symbols, payload_len);
            black_box(rx.feed(black_box(&buf)));
            frame_samples
        },
        frame_samples,
        repeats,
    );
    println!("[DVB-T-Diag-Stream {label}] {stream_msps:.4} Msps in {stream_dt:.3}s");
    let (probed_msps, probed_dt) = measure_throughput(
        || {
            let mut rx = DvbTFrameStreamDemod::new(params, n_symbols, payload_len);
            black_box(rx.feed_probed(black_box(&buf), &mut probe));
            frame_samples
        },
        frame_samples,
        repeats,
    );
    println!("[DVB-T-Diag-Probed {label}] {probed_msps:.4} Msps in {probed_dt:.3}s");
    let probe_overhead = (probed_dt / stream_dt - 1.0) * 100.0;
    println!(
        "[DVB-T-Diag-ProbeOverhead {label}] {probe_overhead:.1}% (stream {stream_msps:.2} -> probed {probed_msps:.2} Msps)"
    );
    assert!(
        !probe.is_empty(),
        "the probed run must actually have filled the probe, or the row above \
         is measuring nothing"
    );

    // The free rungs are always populated; the gated ones are not.
    let got = plain.decode(&buf, n_symbols, payload_len).expect("decode");
    assert!(
        got.diagnostics.cfo_hz.is_some()
            && got.diagnostics.sync_score.is_some()
            && got.diagnostics.rs_corrected_bytes.is_some(),
        "the free rungs cost nothing and are always on"
    );
    assert!(
        got.diagnostics.evm_db.is_none()
            && got.diagnostics.channel_ber.is_none()
            && got.diagnostics.inner_ber.is_none(),
        "the measured rungs stay None until asked for — this is what makes the \
         plain run above a true baseline"
    );

    let floor = minsps_from_env(0.02);
    assert!(
        plain_msps >= floor
            && meas_msps >= floor
            && plain2_msps >= floor
            && stream_msps >= floor
            && probed_msps >= floor
    );
}

#[test]
fn throughput_dvb_t_diagnostics_sparse_frame() {
    // The WORST case, and an unrealistic one: 184 bytes in a frame that holds
    // ~9.7 kB, so 98% of what was transmitted is null stuffing the plain decode
    // skips and a measured decode cannot. Kept because it is the bound, and
    // because a caller who really does send one TS packet per frame pays it.
    dvb_t_diagnostics_cost(184, 60, "Sparse-184B");
}

#[test]
fn throughput_dvb_t_diagnostics_full_frame() {
    // The REALISTIC case for DATV: a payload that fills the frame, so the plain
    // decode and the measured decode read the same coded bits and the only
    // added work is the encode chain the rungs actually need. 9724 = 52 TS
    // packets x 187 payload bytes.
    dvb_t_diagnostics_cost(9724, 30, "Full-9724B");
}

#[test]
fn throughput_dvb_t_spectral_shaping_cost() {
    // What the two TX out-of-band levers cost the modulator. Both are post-passes
    // over an already-assembled frame, so the interesting number is the *delta*
    // against the same modulator with shaping off — everything before them (TS
    // packetization, payload FEC, mapping, pilots, TPS, IFFT, CP) is identical.
    //
    // The taper is O(roll_off) per symbol and touches only guard samples; the mask
    // is O(num_taps) per sample across the whole frame, so it is the one that
    // scales with the filter the guard budget affords.
    let params = DvbTFrameParams {
        link: DvbTLinkParams {
            guard: GuardInterval::G1_8, // cp_len 256: room for a real filter
            constellation: ConstellationOrder::Qpsk,
            code_rate: DvbPunctureRate::R1_2,
        },
        frame_number: 0,
        cell_id: 0,
    };
    let payload = random_bytes(184, 0xD7B1);
    let frame_samples = DvbTFrameMod::new(params).modulate(&payload).iq.len();
    let repeats = 50;

    let roll_off = 16usize;
    let variants: [(&str, DvbTFrameMod); 4] = [
        ("plain", DvbTFrameMod::new(params)),
        (
            "taper",
            DvbTFrameMod::new(params).with_symbol_window(roll_off),
        ),
        (
            "mask-45",
            DvbTFrameMod::new(params).with_tx_lowpass(DvbTFrameMod::tx_lowpass_for_2k(45, 60.0)),
        ),
        (
            "taper+mask-89",
            DvbTFrameMod::new(params)
                .with_symbol_window(roll_off)
                .with_tx_lowpass(DvbTFrameMod::tx_lowpass_for_2k(89, 60.0)),
        ),
    ];

    let min_msps = minsps_from_env(1.0);
    for (label, modulator) in &variants {
        let (msps, dt) = measure_throughput(
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
        println!("[DVB-T-Shaping-{label}] {msps:.4} Msps in {dt:.3}s");
        assert!(
            msps >= min_msps,
            "{label} modulate {msps:.2} Msps below floor {min_msps:.2}"
        );
    }
}
