// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// DVB-T receive diagnostics (`DvbTRxDiagnostics`) and the receive probe
// (`DvbTRxProbe`). Two things are under test and they fail differently:
//
//   • the diagnostics ladder — each rung must carry a value that MOVES with the
//     link, not merely be present. A rung that is always `Some(0.0)` passes an
//     is_some() check and tells an operator nothing, which is the exact bug
//     `cofdm-rx-metrics.md` shipped once already; so the CFO rung is checked
//     against an *injected* offset and the RS-correction rung against injected
//     noise.
//
//   • the probe — an observation of a decode that happens anyway. It must not
//     change what decodes, must partition its flat buffers into the right
//     per-frame spans, and must survive chunk boundaries.

use crate::common::add_awgn;
use num_complex::Complex32 as C32;
use orion_sdr::demodulate::{
    BitOutcome, DvbTFrameDemod, DvbTFrameStreamDemod, DvbTRxProbe, DvbTSuperFrameDemod,
};
use orion_sdr::dsp::Rotator;
use orion_sdr::fec::PunctureRate;
use orion_sdr::modulate::{
    ConstellationOrder, DvbTFrameMod, DvbTSuperFrameMod, DvbTSuperFrameParams,
};
use orion_sdr::waveform::dvb_t::{
    DVB_T_DATA_CARRIERS, DVB_T_N_FFT, DvbTFrameParams, DvbTLinkParams, GuardInterval,
};

fn sample_payload(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 37 + 11) & 0xff) as u8).collect()
}

fn params() -> DvbTFrameParams {
    DvbTFrameParams {
        link: DvbTLinkParams {
            guard: GuardInterval::G1_8,
            constellation: ConstellationOrder::Qpsk,
            code_rate: PunctureRate::R1_2,
        },
        frame_number: 0,
        cell_id: 0,
    }
}

/// One modulated frame with lead-in silence, so the RX must GI-acquire.
fn framed_buffer(p: DvbTFrameParams, payload: &[u8]) -> (Vec<C32>, usize) {
    noisy_buffer(p, payload, 0.0, 0)
}

/// [`framed_buffer`] with AWGN at `rel` times the frame's own mean power,
/// applied to the frame region only.
///
/// Noise is scaled to the signal rather than given in absolute terms — the
/// convention the rest of the DVB-T suite uses, and not a cosmetic one: a DVB-T
/// frame's mean sample power is ~4.4e-4, so an absolute "0.1" is 225x the
/// signal and every frame dies in TPS decode long before the payload is
/// interestingly degraded.
fn noisy_buffer(p: DvbTFrameParams, payload: &[u8], rel: f32, seed: u64) -> (Vec<C32>, usize) {
    let frame = DvbTFrameMod::new(p).modulate(payload);
    let mut iq = vec![C32::default(); 200];
    iq.extend_from_slice(&frame.iq);
    iq.extend(vec![C32::default(); frame.samples_per_symbol]);
    if rel > 0.0 {
        let sig: f32 = frame.iq.iter().map(|s| s.norm_sqr()).sum::<f32>() / frame.iq.len() as f32;
        add_awgn(&mut iq[200..200 + frame.iq.len()], sig * rel, seed);
    }
    (iq, frame.n_symbols)
}

// ── The diagnostics ladder ─────────────────────────────────────────────────

#[test]
fn free_rungs_are_populated_without_asking() {
    // The rungs read straight off work the decode already does are always
    // present: acquisition's CFO and score, the timing offset, the outer-FEC
    // verdict and its corrected-byte count. Making these opt-in would be an
    // `Option` that is `None` only because a flag was off, which is a worse API
    // than one that is always there.
    let p = params();
    let payload = sample_payload(184);
    let (iq, n_symbols) = framed_buffer(p, &payload);

    let got = DvbTFrameDemod::new(p)
        .decode(&iq, n_symbols, payload.len())
        .expect("decode");

    let d = got.diagnostics;
    assert!(d.cfo_hz.is_some(), "cfo_hz is free");
    assert!(d.sync_score.is_some(), "sync_score is free");
    assert!(
        d.timing_offset_samples.is_some(),
        "timing_offset_samples is free"
    );
    assert_eq!(d.outer_fec_ok, Some(true), "the frame verified");
    assert!(
        d.rs_corrected_bytes.is_some(),
        "the corrected-byte count is free and always reported"
    );
    // Correction is off by default, so there is no integer estimate at all.
    // `None` here means 'never looked', which is NOT the same as Some(0).
    assert_eq!(d.integer_cfo_bins, None);
    // The sync score is a real correlation, not a placeholder.
    let score = d.sync_score.unwrap();
    assert!(
        (0.5..=1.0).contains(&score),
        "a noiseless CP must correlate strongly, got {score}"
    );
}

#[test]
fn gated_rungs_stay_none_until_asked_for_and_are_not_zero() {
    // The distinction this pins down: `None` (not measured) and `Some(0.0)`
    // (measured, and the link is perfect) must be different values. If the
    // ungated path reported 0.0 instead of None, a receiver that never measured
    // anything would render as a flawless link.
    let p = params();
    let payload = sample_payload(184);
    let (iq, n_symbols) = framed_buffer(p, &payload);

    let plain = DvbTFrameDemod::new(p)
        .decode(&iq, n_symbols, payload.len())
        .expect("decode");
    assert_eq!(plain.diagnostics.evm_db, None);
    assert_eq!(plain.diagnostics.channel_ber, None);
    assert_eq!(plain.diagnostics.inner_ber, None);

    let measured = DvbTFrameDemod::new(p)
        .with_error_rates(true)
        .decode(&iq, n_symbols, payload.len())
        .expect("decode");
    // On a noiseless link the two BERs are measured and are exactly zero —
    // which is the whole point: Some(0.0) is a measurement, None is not.
    assert_eq!(
        measured.diagnostics.channel_ber,
        Some(0.0),
        "a noiseless channel delivers every coded bit"
    );
    assert_eq!(measured.diagnostics.inner_ber, Some(0.0));
    let evm = measured.diagnostics.evm_db.expect("evm measured");
    assert!(
        evm < -40.0,
        "a noiseless link should have near-perfect EVM, got {evm} dB"
    );

    // Enabling the rungs must not change what decodes.
    assert_eq!(plain.payload, measured.payload);
    assert_eq!(plain.payload, payload);
}

#[test]
fn cfo_hz_tracks_an_injected_offset() {
    // The rung that a presence check cannot validate. A `cfo_hz` that returned a
    // constant — or the residual *after* correction rather than the link's
    // actual offset — passes `is_some()` and is useless. So inject a known
    // fractional offset and require the report to track it.
    //
    // Fractional only: guard-interval acquisition resolves ±½ a subcarrier, and
    // a larger offset needs the integer estimator (exercised separately below).
    //
    // The offsets stay inside ±0.15 of a subcarrier because the DVB-T demod
    // *estimates* the fractional CFO but does not *apply* it — nothing rotates
    // the buffer by `acq.cfo_hz` — so past roughly a fifth of a bin the TPS
    // carriers stop resolving and there is no frame to read a diagnostic from.
    let p = params();
    let payload = sample_payload(184);
    let frame = DvbTFrameMod::new(p).modulate(&payload);
    let fs = p.config().fs;
    let bin_hz = fs / DVB_T_N_FFT as f32;

    for frac in [0.0f32, 0.05, 0.1, -0.1, 0.15] {
        let offset_hz = frac * bin_hz;
        let mut shifted = vec![C32::default(); frame.iq.len()];
        Rotator::new(offset_hz, fs).rotate_block(&frame.iq, &mut shifted);
        let mut iq = vec![C32::default(); 200];
        iq.extend_from_slice(&shifted);
        iq.extend(vec![C32::default(); frame.samples_per_symbol]);

        let got = DvbTFrameDemod::new(p)
            .decode(&iq, frame.n_symbols, payload.len())
            .expect("decode at a fractional offset");
        let reported = got.diagnostics.cfo_hz.expect("cfo_hz");
        // A tenth of a subcarrier of slack: the estimator is a CP correlation
        // over one symbol, not an oracle. Far tighter than the ±½ bin the rung
        // would have to be wrong by to be reporting a constant.
        assert!(
            (reported - offset_hz).abs() < 0.1 * bin_hz,
            "cfo_hz must track the injected offset: injected {offset_hz:.2} Hz, \
             reported {reported:.2} Hz (bin = {bin_hz:.2} Hz)"
        );
    }
}

#[test]
fn cfo_hz_reports_the_total_when_an_integer_offset_was_removed() {
    // With integer correction on, acquisition runs on the ALREADY-ROTATED
    // buffer, so its own `cfo_hz` is the residual. Reporting that would describe
    // the receiver's internal state rather than the link — a display would show
    // ~0 Hz while the transmitter sat five subcarriers away. The rung must add
    // the removed offset back.
    let p = params();
    let payload = sample_payload(184);
    let frame = DvbTFrameMod::new(p).modulate(&payload);
    let fs = p.config().fs;
    let bin_hz = fs / DVB_T_N_FFT as f32;

    let k_true = 5i32;
    let mut shifted = vec![C32::default(); frame.iq.len()];
    Rotator::new(k_true as f32 * bin_hz, fs).rotate_block(&frame.iq, &mut shifted);
    let mut iq = vec![C32::default(); 200];
    iq.extend_from_slice(&shifted);
    iq.extend(vec![C32::default(); frame.samples_per_symbol]);

    let got = DvbTFrameDemod::new(p)
        .with_integer_cfo_correction(true)
        .decode(&iq, frame.n_symbols, payload.len())
        .expect("decode with integer-CFO correction");

    assert_eq!(
        got.diagnostics.integer_cfo_bins,
        Some(k_true),
        "the estimator found the injected whole-subcarrier offset"
    );
    let reported = got.diagnostics.cfo_hz.expect("cfo_hz");
    let expected = k_true as f32 * bin_hz;
    assert!(
        (reported - expected).abs() < 0.1 * bin_hz,
        "cfo_hz must report the TOTAL offset ({expected:.1} Hz), not the \
         post-correction residual; got {reported:.1} Hz"
    );
}

#[test]
fn integer_cfo_bins_distinguishes_on_frequency_from_never_looked() {
    // `Some(0)` and `None` are different measurements and the API keeps them
    // apart: correction off means no estimate exists; correction on over a clean
    // buffer means the estimator ran and found nothing to remove.
    let p = params();
    let payload = sample_payload(184);
    let (iq, n_symbols) = framed_buffer(p, &payload);

    let off = DvbTFrameDemod::new(p)
        .decode(&iq, n_symbols, payload.len())
        .expect("decode");
    assert_eq!(off.diagnostics.integer_cfo_bins, None, "never looked");

    let on = DvbTFrameDemod::new(p)
        .with_integer_cfo_correction(true)
        .decode(&iq, n_symbols, payload.len())
        .expect("decode");
    assert_eq!(
        on.diagnostics.integer_cfo_bins,
        Some(0),
        "looked, and the link is on frequency"
    );
}

#[test]
fn a_prefix_decode_carries_a_structural_rs_correction_floor() {
    // A measured property of the waveform, pinned here because it is surprising
    // and it bounds what the corrected-byte count means.
    //
    // `DvbTFrameMod` stuffs null packets until the coded stream fills the frame,
    // so a receiver told `payload_len = 184` decodes a PREFIX of what was sent.
    // With a Forney(12,17) outer interleaver that prefix is not self-contained:
    // the deinterleaver's tail draws on codewords the prefix never covers, and
    // Reed–Solomon quietly repairs the shortfall. On a NOISELESS link that costs
    // one correction out of the eight RS(204,188) can make.
    //
    // Decoding the whole frame — what `with_error_rates` does, because an exact
    // re-encode requires it — has no shortfall to repair and reports zero.
    let p = params();
    let payload = sample_payload(184);
    let (iq, n_symbols) = framed_buffer(p, &payload);

    let prefix = DvbTFrameDemod::new(p)
        .decode(&iq, n_symbols, payload.len())
        .expect("decode");
    let full = DvbTFrameDemod::new(p)
        .with_error_rates(true)
        .decode(&iq, n_symbols, payload.len())
        .expect("decode");

    assert_eq!(
        full.diagnostics.rs_corrected_bytes,
        Some(0),
        "a whole-frame decode of a noiseless link needs no correction at all"
    );
    assert_eq!(
        prefix.diagnostics.rs_corrected_bytes,
        Some(1),
        "the prefix decode spends one of the eight correctable bytes on the \
         interleaver shortfall, before the channel has done anything"
    );
    // Both recover the payload identically — the floor is a cost in margin, not
    // in correctness.
    assert_eq!(prefix.payload, payload);
    assert_eq!(full.payload, payload);
}

#[test]
fn rs_corrected_bytes_rises_with_noise() {
    // The rung that degrades gracefully where `outer_fec_ok` saturates: under
    // noise the frame still delivers every byte, and the only signal that the
    // link is working for it is the correction count. A flag cannot express
    // this, which is why the count exists.
    let p = params();
    let payload = sample_payload(184);

    let (clean, n_symbols) = framed_buffer(p, &payload);
    let quiet = DvbTFrameDemod::new(p)
        .decode(&clean, n_symbols, payload.len())
        .expect("decode");
    let floor = quiet.diagnostics.rs_corrected_bytes.expect("count present");

    // Well down the curve but still delivering: measured, this sits around 3–6
    // corrections against the noiseless floor of 1.
    let (noisy, _) = noisy_buffer(p, &payload, 0.45, 0x5EED_0001);
    let got = DvbTFrameDemod::new(p)
        .decode(&noisy, n_symbols, payload.len())
        .expect("still decodes at this noise level");
    let corrected = got.diagnostics.rs_corrected_bytes.expect("count present");

    assert!(
        corrected > floor,
        "the correction count must track the channel: noiseless floor {floor}, \
         under noise {corrected}"
    );
    assert_eq!(
        got.payload, payload,
        "RS corrected {corrected} bytes and still delivered the payload exactly \
         — that is the state the count exists to report"
    );
    assert_eq!(
        got.diagnostics.outer_fec_ok,
        Some(true),
        "the flag saturates at true while the count moves"
    );
}

#[test]
fn error_rates_propagate_through_the_stream_and_super_frame_builders() {
    // `with_error_rates` is a link-constant knob and must reach every
    // constituent frame demod, exactly as `with_integer_cfo_correction` and
    // `with_rx_window_backoff` already do.
    let p = params();
    let payload = sample_payload(184);
    let (iq, n_symbols) = framed_buffer(p, &payload);

    let mut rx = DvbTFrameStreamDemod::new(p, n_symbols, payload.len()).with_error_rates(true);
    let mut got = rx.feed(&iq);
    got.extend(rx.flush());
    let ok: Vec<_> = got.into_iter().filter_map(|r| r.ok()).collect();
    assert_eq!(ok.len(), 1);
    assert_eq!(
        ok[0].diagnostics.channel_ber,
        Some(0.0),
        "the stream receiver propagated the flag to its frame demod"
    );
    assert!(ok[0].diagnostics.evm_db.is_some());

    // Super-frame: the flag reaching each of the four constituent frames.
    let sp = DvbTSuperFrameParams {
        link: p.link,
        cell_id: 0x1234,
    };
    let sf_payload = sample_payload(4 * 184);
    let sf = DvbTSuperFrameMod::new(sp).modulate(&sf_payload);
    let rx_sf = DvbTSuperFrameDemod::new(sp).with_error_rates(true);
    assert!(rx_sf.error_rates(), "the accessor reports the flag");
    let out = rx_sf
        .decode(&sf.iq, sf.symbols_per_frame, sf.frame_payload_lens)
        .expect("super-frame decode");
    assert_eq!(out.payload, sf_payload);
    assert_eq!(out.cell_id, 0x1234);
}

// ── The probe ──────────────────────────────────────────────────────────────

#[test]
fn probe_symbol_count_matches_the_scattered_grid() {
    // A DVB-T frame carries exactly 1512 data cells per OFDM symbol, on every
    // one of the four scattered-pilot phases. The probe's constellation must be
    // that grid and nothing else — an off-by-one phase would change the count.
    let p = params();
    let payload = sample_payload(184);
    let (iq, n_symbols) = framed_buffer(p, &payload);

    let mut rx = DvbTFrameStreamDemod::new(p, n_symbols, payload.len());
    let mut probe = DvbTRxProbe::new();
    // Read the probe after the call that FILLED it. A following `flush_probed`
    // would clear it and report its own (empty) result — every probed entry
    // point clears first, by design.
    let got = rx.feed_probed(&iq, &mut probe);
    assert_eq!(got.into_iter().filter(|r| r.is_ok()).count(), 1);

    assert_eq!(probe.frames().len(), 1, "one frame probed");
    let f = probe.iter().next().expect("one probed frame");
    assert!(f.meta.decoded);
    assert_eq!(
        f.symbols.len(),
        DVB_T_DATA_CARRIERS * n_symbols,
        "1512 data carriers per symbol, {n_symbols} symbols"
    );
    assert_eq!(f.meta.constellation, ConstellationOrder::Qpsk);
    assert_eq!(
        f.meta.tps.frame_number, 0,
        "the TPS word, not a fake counter"
    );
}

#[test]
fn probing_does_not_change_what_decodes() {
    // The probe is an observation of a decode that happens anyway. Payload and
    // every diagnostic rung must be identical with and without it.
    //
    // Run at r1/2 and at r3/4: the two rates sat on opposite sides of the
    // frame-filling overrun, so a regression that reaches only one of them would
    // otherwise pass here.
    for rate in [PunctureRate::R1_2, PunctureRate::R3_4] {
        let p = mode_params(GuardInterval::G1_8, ConstellationOrder::Qpsk, rate);
        let payload = sample_payload(184);
        let (iq, n_symbols) = framed_buffer(p, &payload);

        let mut plain_rx =
            DvbTFrameStreamDemod::new(p, n_symbols, payload.len()).with_error_rates(true);
        let mut plain = plain_rx.feed(&iq);
        plain.extend(plain_rx.flush());
        let plain: Vec<_> = plain.into_iter().filter_map(|r| r.ok()).collect();

        let mut probed_rx =
            DvbTFrameStreamDemod::new(p, n_symbols, payload.len()).with_error_rates(true);
        let mut probe = DvbTRxProbe::new();
        let mut probed = probed_rx.feed_probed(&iq, &mut probe);
        probed.extend(probed_rx.flush_probed(&mut probe));
        let probed: Vec<_> = probed.into_iter().filter_map(|r| r.ok()).collect();

        assert_eq!(plain.len(), 1, "{rate:?}: one frame decoded unprobed");
        assert_eq!(plain.len(), probed.len(), "{rate:?}");
        for (a, b) in plain.iter().zip(probed.iter()) {
            assert_eq!(
                a.payload, b.payload,
                "{rate:?}: payload unchanged by probing"
            );
            assert_eq!(a.tps, b.tps, "{rate:?}: TPS unchanged by probing");
            assert_eq!(
                a.diagnostics, b.diagnostics,
                "{rate:?}: every diagnostic rung unchanged by probing"
            );
        }
    }
}

#[test]
fn correction_map_is_clean_on_a_noiseless_link() {
    // Every coded bit arrived correct and the decoder left it correct. A map
    // that reported anything else here would mean its three streams are not in
    // the same index space.
    let p = params();
    let payload = sample_payload(184);
    let (iq, n_symbols) = framed_buffer(p, &payload);

    let mut rx = DvbTFrameStreamDemod::new(p, n_symbols, payload.len());
    let mut probe = DvbTRxProbe::new();
    let _ = rx.feed_probed(&iq, &mut probe);

    let f = probe.iter().next().expect("one probed frame");
    assert!(!f.correction.is_empty(), "a decoded frame has a map");
    assert!(
        f.correction.iter().all(|&o| o == BitOutcome::Clean),
        "a noiseless link must be entirely Clean; found {} non-clean of {}",
        f.correction
            .iter()
            .filter(|&&o| o != BitOutcome::Clean)
            .count(),
        f.correction.len()
    );
}

#[test]
fn correction_map_reproduces_the_channel_ber_it_expands() {
    // The map is the channel BER's per-bit expansion, not a second measurement
    // of it: counting `arrived_wrong` over the map and dividing by its length
    // must reproduce `channel_ber` exactly. If the two disagree, one of them is
    // indexing a different bit stream.
    let p = params();
    let payload = sample_payload(184);
    // Enough noise for a non-trivial error rate (~1.8%) while decoding reliably.
    let (iq, n_symbols) = noisy_buffer(p, &payload, 0.2, 0x51C4_0001);

    let mut rx = DvbTFrameStreamDemod::new(p, n_symbols, payload.len()).with_error_rates(true);
    let mut probe = DvbTRxProbe::new();
    let got = rx.feed_probed(&iq, &mut probe);
    let Some(Ok(frame)) = got.into_iter().next() else {
        panic!("expected a decode at 0.2x noise")
    };
    let f = probe.iter().next().expect("a probed frame");
    assert!(f.meta.decoded);

    let cber = frame.diagnostics.channel_ber.expect("measured");
    assert!(
        cber > 0.0,
        "the comparison is vacuous unless the channel actually errored"
    );
    let wrong = f.correction.iter().filter(|o| o.arrived_wrong()).count();
    let from_map = wrong as f32 / f.correction.len() as f32;
    assert!(
        (from_map - cber).abs() < 1e-6,
        "the map must expand the same measurement the scalar collapses: \
         map {from_map:.6} vs channel_ber {cber:.6}"
    );
}

#[test]
fn probe_is_cleared_between_calls() {
    // Every probed entry point clears first, so records do not accumulate across
    // calls — a caller reading the probe after a `feed` sees that call's frames
    // and no others. Without this the spans from an earlier call would still
    // index a buffer that has been refilled.
    let p = params();
    let payload = sample_payload(184);
    let (iq, n_symbols) = framed_buffer(p, &payload);

    let mut rx = DvbTFrameStreamDemod::new(p, n_symbols, payload.len());
    let mut probe = DvbTRxProbe::new();

    let _ = rx.feed_probed(&iq, &mut probe);
    assert_eq!(probe.frames().len(), 1);
    let first_len = probe.symbols().len();

    // A second call with no new samples produces no frames — and must leave the
    // probe empty rather than still showing the first call's.
    let _ = rx.feed_probed(&[], &mut probe);
    assert!(
        probe.is_empty(),
        "a call that produced no frames leaves an empty probe"
    );
    assert!(probe.symbols().is_empty());

    // And a fresh buffer refills it to the same size, not double.
    let (iq2, _) = framed_buffer(p, &payload);
    let _ = rx.feed_probed(&iq2, &mut probe);
    assert_eq!(probe.frames().len(), 1);
    assert_eq!(
        probe.symbols().len(),
        first_len,
        "buffers are refilled, not appended to"
    );
}

#[test]
fn feed_probed_is_chunk_boundary_invariant() {
    // Matching the discipline of the existing `dvb_t_stream_*` suite: feeding
    // the same IQ in small boundary-crossing chunks must recover the same frame
    // and the same probe contents as one shot. The probe is refilled per call,
    // so the comparison is against the call that actually produced the frame.
    let p = params();
    let payload = sample_payload(184);
    let (iq, n_symbols) = framed_buffer(p, &payload);

    let mut one_rx = DvbTFrameStreamDemod::new(p, n_symbols, payload.len());
    let mut one_probe = DvbTRxProbe::new();
    let one = one_rx.feed_probed(&iq, &mut one_probe);
    let one_ok: Vec<_> = one.into_iter().filter_map(|r| r.ok()).collect();
    assert_eq!(one_ok.len(), 1);
    let one_syms = one_probe.symbols().to_vec();
    let one_map = one_probe.correction().to_vec();

    // Chunked, with a chunk size that is not a factor of the symbol period.
    let mut ch_rx = DvbTFrameStreamDemod::new(p, n_symbols, payload.len());
    let mut ch_probe = DvbTRxProbe::new();
    let mut chunked_syms = Vec::new();
    let mut chunked_map = Vec::new();
    let mut chunked_ok = Vec::new();
    for chunk in iq.chunks(1237) {
        let out = ch_rx.feed_probed(chunk, &mut ch_probe);
        if out.iter().any(|r| r.is_ok()) {
            chunked_syms = ch_probe.symbols().to_vec();
            chunked_map = ch_probe.correction().to_vec();
        }
        chunked_ok.extend(out.into_iter().filter_map(|r| r.ok()));
    }

    assert_eq!(chunked_ok.len(), 1, "chunking must not lose the frame");
    assert_eq!(chunked_ok[0].payload, one_ok[0].payload);
    assert_eq!(chunked_ok[0].payload, payload);
    assert_eq!(
        chunked_syms.len(),
        one_syms.len(),
        "the same constellation regardless of how the samples arrived"
    );
    assert_eq!(chunked_syms, one_syms);
    assert_eq!(chunked_map, one_map);
}

#[test]
fn an_undecoded_frame_contributes_symbols_but_no_map() {
    // The state the probe exists for. When frames stop decoding, the
    // constellation is exactly what an operator looks at — so a frame that
    // reached the demapper and failed its Reed–Solomon check still records its
    // symbols, with an EMPTY map. Empty is "no ground truth", not "no errors".
    // At 0.6x signal power the payload is past its cliff while the TPS carriers
    // — differential BPSK spread across all 68 symbols, BCH-protected — still
    // resolve. That ordering is what makes the state reachable at all: TPS is
    // decoded before the payload, and a frame that loses TPS never reaches the
    // payload decode and so records nothing.
    let p = params();
    let payload = sample_payload(184);
    let (noisy, n_symbols) = noisy_buffer(p, &payload, 0.6, 0x9E11_0001);

    let mut rx = DvbTFrameStreamDemod::new(p, n_symbols, payload.len());
    let mut probe = DvbTRxProbe::new();
    let out = rx.feed_probed(&noisy, &mut probe);

    assert!(
        out.iter().any(|r| r.is_err()),
        "expected the payload to fail at this noise level"
    );
    let f = probe
        .iter()
        .next()
        .expect("a frame that demapped and then failed still records a probe");
    assert!(!f.meta.decoded, "it did not verify");
    assert_eq!(
        f.symbols.len(),
        DVB_T_DATA_CARRIERS * n_symbols,
        "a failed frame still contributes its whole constellation"
    );
    assert!(
        f.correction.is_empty(),
        "no ground truth means no map — and that must read as absent, not as a \
         map full of Clean"
    );
}

#[test]
fn probe_partitions_several_frames_into_their_own_spans() {
    // `feed` drains every complete frame in the buffer, so one probed call can
    // return several — and the flat buffers must partition into per-frame spans
    // that do not overlap and cover the whole thing. A span bug here shows up as
    // one frame's constellation rendered under another's label.
    let p = params();
    let payload = sample_payload(184);
    let frame = DvbTFrameMod::new(p).modulate(&payload);

    const N: usize = 3;
    let mut iq = vec![C32::default(); 200];
    for _ in 0..N {
        iq.extend_from_slice(&frame.iq);
    }
    iq.extend(vec![C32::default(); frame.samples_per_symbol]);

    let mut rx = DvbTFrameStreamDemod::new(p, frame.n_symbols, payload.len());
    let mut probe = DvbTRxProbe::new();
    // One call, read straight after: `feed` drains every complete frame it can,
    // so the run arrives together and the probe describes exactly this call.
    let got = rx.feed_probed(&iq, &mut probe);
    let ok: Vec<_> = got.into_iter().filter_map(|r| r.ok()).collect();

    // The stream receiver may not recover the trailing frame if its lead-out is
    // short; require at least two so the partitioning is genuinely exercised.
    assert!(ok.len() >= 2, "expected a run of frames, got {}", ok.len());
    assert_eq!(probe.frames().len(), ok.len(), "one record per frame");

    let per_frame = DVB_T_DATA_CARRIERS * frame.n_symbols;
    let mut expected_start = 0usize;
    for f in probe.iter() {
        assert_eq!(f.symbols.len(), per_frame, "each frame's own grid");
        assert!(f.meta.decoded);
        assert!(!f.correction.is_empty());
        expected_start += per_frame;
    }
    assert_eq!(
        probe.symbols().len(),
        expected_start,
        "the per-frame spans tile the flat buffer exactly"
    );
}

// ── The gate must not break the decode, in any mode or at any size ──────────
//
// `with_error_rates(true)` makes the receiver decode the WHOLE frame rather than
// the payload prefix, because a truth re-encode of a prefix does not reproduce
// what was transmitted under a Forney(12,17) interleaver. That whole-frame plan
// has to describe bits that were actually sent. It used to describe more: the
// modulator stuffed until its coded stream MET OR EXCEEDED the frame's capacity
// and then transmitted only the capacity, so the receiver asked its decoder for
// the overrun too — 1096 bits at QPSK r3/4 — and the frame failed.
//
// The overrun existed in every mode. What it cost varied, which is why this is
// swept rather than spot-checked:
//
//   • where the overrun was exactly the K=7 tail (10 of 15 modes at 184 B), the
//     frame lost only termination and every rung still read zero;
//   • where it was larger and Reed-Solomon could not absorb it (QPSK r3/4 and
//     r7/8, 16-QAM r7/8, 64-QAM r3/4), the frame failed outright;
//   • and at 64-QAM r7/8 it was larger — 474 bits — and RS absorbed it, so the
//     frame DECODED while reporting inner_ber = 5.0e-5 and rs_corrected_bytes =
//     5 on a noiseless link. No error, just wrong numbers.
//
// That third case is why these tests assert exact zeros rather than success, and
// why 64-QAM r7/8 must stay in the sweep: it is invisible to the arithmetic
// checks in `tests/unit/dvb_t.rs`, which can show its overrun but not that the
// frame survives one. It is the case that separates a real fix from a clamp.

/// Every DVB-T constellation/rate pair the TPS word can signal.
fn every_mode() -> Vec<(ConstellationOrder, PunctureRate)> {
    let mut modes = Vec::new();
    for c in [
        ConstellationOrder::Qpsk,
        ConstellationOrder::Qam16,
        ConstellationOrder::Qam64,
    ] {
        for r in [
            PunctureRate::R1_2,
            PunctureRate::R2_3,
            PunctureRate::R3_4,
            PunctureRate::R5_6,
            PunctureRate::R7_8,
        ] {
            modes.push((c, r));
        }
    }
    modes
}

fn mode_params(
    guard: GuardInterval,
    constellation: ConstellationOrder,
    code_rate: PunctureRate,
) -> DvbTFrameParams {
    DvbTFrameParams {
        link: DvbTLinkParams {
            guard,
            constellation,
            code_rate,
        },
        frame_number: 0,
        cell_id: 0,
    }
}

/// Decodes one noiseless frame with the gate on and asserts every measured rung
/// reads an exact zero. `what` names the configuration in the failure message.
fn assert_gated_decode_is_exact(p: DvbTFrameParams, payload_len: usize, what: &str) {
    let payload = sample_payload(payload_len);
    let (iq, n_symbols) = framed_buffer(p, &payload);
    let rx = DvbTFrameDemod::new(p).with_error_rates(true);
    let frame = rx
        .decode(&iq, n_symbols, payload.len())
        .unwrap_or_else(|e| panic!("{what}: gated decode failed on a noiseless link: {e}"));

    assert_eq!(frame.payload, payload, "{what}: payload");
    // Exactly zero, not merely small. A near-zero is what a truth reference
    // built over untransmitted bits looks like.
    assert_eq!(
        frame.diagnostics.channel_ber,
        Some(0.0),
        "{what}: CBER must be exactly zero on a noiseless link"
    );
    assert_eq!(
        frame.diagnostics.inner_ber,
        Some(0.0),
        "{what}: IBER must be exactly zero on a noiseless link"
    );
    assert_eq!(
        frame.diagnostics.rs_corrected_bytes,
        Some(0),
        "{what}: the whole-frame decode must spend no RS correction"
    );
    assert_eq!(
        frame.diagnostics.outer_fec_ok,
        Some(true),
        "{what}: outer FEC"
    );
}

#[test]
fn error_rates_decode_every_mode() {
    for (c, r) in every_mode() {
        let p = mode_params(GuardInterval::G1_8, c, r);
        assert_gated_decode_is_exact(p, 184, &format!("{c:?} {r:?} 184 B"));
    }
}

#[test]
fn error_rates_decode_across_payload_sizes() {
    // The mode sweep above runs at 68 symbols, where the frame's capacity is a
    // multiple of the coded step for most rates and the overrun happened to land
    // on the tail. A larger payload moves the frame off that coincidence, so the
    // modes that survived by arithmetic luck are exercised here — QPSK r1/2 used
    // to fail at all three of these sizes.
    for (c, r) in every_mode() {
        let p = mode_params(GuardInterval::G1_8, c, r);
        for n_pkt in [60usize, 100, 140] {
            let len = n_pkt * 187;
            assert_gated_decode_is_exact(p, len, &format!("{c:?} {r:?} {n_pkt} pkts"));
        }
    }
}

#[test]
fn error_rates_decode_at_every_guard() {
    // The fill rule is guard-independent by construction — capacity is
    // `n_symbols · 1512 · bits/symbol` and the guard only changes `cp_len`, i.e.
    // the sample count per symbol. Pinned rather than re-derived.
    for guard in [
        GuardInterval::G1_32,
        GuardInterval::G1_16,
        GuardInterval::G1_8,
        GuardInterval::G1_4,
    ] {
        let p = mode_params(guard, ConstellationOrder::Qpsk, PunctureRate::R3_4);
        assert_gated_decode_is_exact(p, 184, &format!("{guard:?} QPSK r3/4"));
    }
}

#[test]
fn probing_yields_a_constellation_at_a_broken_mode() {
    // `feed_probed` sets the same whole-frame flag as `with_error_rates`, so a
    // mode that failed the gated decode produced no constellation either — the
    // frame took the `push_undecoded` path and a viewer's pane stayed empty.
    // QPSK r3/4 is one of the four that failed outright.
    let p = mode_params(
        GuardInterval::G1_32,
        ConstellationOrder::Qpsk,
        PunctureRate::R3_4,
    );
    let payload = sample_payload(184);
    let (iq, n_symbols) = framed_buffer(p, &payload);

    let mut rx = DvbTFrameStreamDemod::new(p, n_symbols, payload.len());
    let mut probe = DvbTRxProbe::new();
    // `feed_probed` alone: a following `flush_probed` would clear the probe and
    // refill it with what that call produced, which is nothing.
    let frames = rx.feed_probed(&iq, &mut probe);

    assert_eq!(frames.len(), 1, "one frame");
    let got = frames[0].as_ref().expect("probed decode at QPSK r3/4");
    assert_eq!(got.payload, payload);

    let f = probe.iter().next().expect("one probed frame");
    assert!(
        f.meta.decoded,
        "a decoded frame must carry a correction map, not push_undecoded"
    );
    assert_eq!(
        f.symbols.len(),
        DVB_T_DATA_CARRIERS * n_symbols,
        "the whole constellation is recorded"
    );
    assert!(!f.correction.is_empty(), "correction map is populated");
    assert!(
        f.correction.iter().all(|&o| o == BitOutcome::Clean),
        "every coded bit is Clean on a noiseless link; found {} non-clean of {}",
        f.correction
            .iter()
            .filter(|&&o| o != BitOutcome::Clean)
            .count(),
        f.correction.len()
    );
}
