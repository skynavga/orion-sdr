// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// Streaming DVB-T frame receiver (demodulate::dvb_t_stream): feed/flush over the
// batch RX. Verifies a single frame decodes, chunk-boundary-invariance (chunked
// feed == one-shot), a continuous run of frames decodes across feeds, and AWGN.

use crate::common::add_awgn;
use num_complex::Complex32 as C32;
use orion_sdr::demodulate::DvbTFrameStreamDemod;
use orion_sdr::fec::PunctureRate;
use orion_sdr::modulate::{ConstellationOrder, DvbTFrameMod};
use orion_sdr::waveform::dvb_t::{DvbTFrameParams, DvbTLinkParams, GuardInterval};

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

#[test]
fn dvb_t_stream_single_frame() {
    let p = params();
    let payload = sample_payload(184);
    let frame = DvbTFrameMod::new(p).modulate(&payload);

    // Lead-in silence + frame + trailing silence, fed in one shot.
    let mut iq = vec![C32::default(); 200];
    iq.extend_from_slice(&frame.iq);
    iq.extend(vec![C32::default(); frame.samples_per_symbol]);

    let mut rx = DvbTFrameStreamDemod::new(p, frame.n_symbols, payload.len());
    let mut got = rx.feed(&iq);
    got.extend(rx.flush());
    let ok: Vec<_> = got.into_iter().filter_map(|r| r.ok()).collect();
    assert_eq!(ok.len(), 1, "exactly one frame decoded");
    assert_eq!(ok[0].payload, payload);
}

#[test]
fn dvb_t_stream_chunked_feed_matches_oneshot() {
    // Feeding the same IQ in small, boundary-crossing chunks must recover the same
    // frame as feeding it all at once.
    let p = params();
    let payload = sample_payload(184);
    let frame = DvbTFrameMod::new(p).modulate(&payload);
    let mut iq = vec![C32::default(); 200];
    iq.extend_from_slice(&frame.iq);
    iq.extend(vec![C32::default(); frame.samples_per_symbol]);

    // One-shot.
    let mut rx1 = DvbTFrameStreamDemod::new(p, frame.n_symbols, payload.len());
    let mut a = rx1.feed(&iq);
    a.extend(rx1.flush());
    let a: Vec<_> = a
        .into_iter()
        .filter_map(|r| r.ok().map(|f| f.payload))
        .collect();

    // Chunked at an awkward stride (not a symbol multiple).
    let mut rx2 = DvbTFrameStreamDemod::new(p, frame.n_symbols, payload.len());
    let mut b = Vec::new();
    for chunk in iq.chunks(777) {
        b.extend(rx2.feed(chunk));
    }
    b.extend(rx2.flush());
    let b: Vec<_> = b
        .into_iter()
        .filter_map(|r| r.ok().map(|f| f.payload))
        .collect();

    assert_eq!(a, b, "chunked feed must match one-shot");
    assert_eq!(a, vec![payload]);
}

#[test]
fn dvb_t_stream_multiple_frames() {
    // A continuous run of three back-to-back frames must all decode in order.
    let p = params();
    let payloads: Vec<Vec<u8>> = (0..3)
        .map(|k| {
            (0..184)
                .map(|i| ((i * 13 + k * 40 + 1) & 0xff) as u8)
                .collect()
        })
        .collect();
    let modulator = DvbTFrameMod::new(p);
    let frame0 = modulator.modulate(&payloads[0]);
    let n_symbols = frame0.n_symbols;

    // Lead-in, then three frames back to back, then trailing silence.
    let mut iq = vec![C32::default(); 200];
    for pl in &payloads {
        let f = modulator.modulate(pl);
        assert_eq!(f.n_symbols, n_symbols, "frames share a symbol count");
        iq.extend_from_slice(&f.iq);
    }
    iq.extend(vec![C32::default(); frame0.samples_per_symbol]);

    let mut rx = DvbTFrameStreamDemod::new(p, n_symbols, 184);
    // Feed in two halves to exercise cross-feed frame completion.
    let mid = iq.len() / 2;
    let mut got = rx.feed(&iq[..mid]);
    got.extend(rx.feed(&iq[mid..]));
    got.extend(rx.flush());
    let ok: Vec<Vec<u8>> = got
        .into_iter()
        .filter_map(|r| r.ok().map(|f| f.payload))
        .collect();
    assert_eq!(ok, payloads, "all three frames recovered in order");
}

#[test]
fn dvb_t_stream_survives_awgn() {
    let p = params();
    let payload = sample_payload(184);
    let frame = DvbTFrameMod::new(p).modulate(&payload);
    let mut iq = vec![C32::default(); 200];
    iq.extend_from_slice(&frame.iq);
    iq.extend(vec![C32::default(); frame.samples_per_symbol]);
    let sig_power: f32 = frame.iq.iter().map(|s| s.norm_sqr()).sum::<f32>() / frame.iq.len() as f32;
    add_awgn(
        &mut iq[200..200 + frame.iq.len()],
        sig_power * 0.03,
        0x57EA_0777,
    );

    let mut rx = DvbTFrameStreamDemod::new(p, frame.n_symbols, payload.len());
    let mut got = rx.feed(&iq);
    got.extend(rx.flush());
    let ok: Vec<_> = got.into_iter().filter_map(|r| r.ok()).collect();
    assert_eq!(ok.len(), 1);
    assert_eq!(ok[0].payload, payload);
}

#[test]
fn dvb_t_stream_holds_partial_frame() {
    // Feeding less than a full frame yields nothing until the rest arrives.
    let p = params();
    let payload = sample_payload(184);
    let frame = DvbTFrameMod::new(p).modulate(&payload);
    let mut iq = vec![C32::default(); 200];
    iq.extend_from_slice(&frame.iq);
    iq.extend(vec![C32::default(); frame.samples_per_symbol]);

    let mut rx = DvbTFrameStreamDemod::new(p, frame.n_symbols, payload.len());
    let half = iq.len() / 2;
    let early = rx.feed(&iq[..half]);
    assert!(early.is_empty(), "no frame before it fully arrives");
    let mut late = rx.feed(&iq[half..]);
    late.extend(rx.flush());
    let ok: Vec<_> = late.into_iter().filter_map(|r| r.ok()).collect();
    assert_eq!(ok.len(), 1);
    assert_eq!(ok[0].payload, payload);
}
