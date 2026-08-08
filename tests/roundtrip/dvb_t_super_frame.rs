// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// Conformant DVB-T super-frame roundtrip (EN 300 744 §4.4/§4.6): four consecutive
// frames (numbers 0..3) with alternating TPS sync + a 16-bit cell id split across
// them, modulated by `modulate::DvbTSuperFrameMod` and recovered by
// `demodulate::DvbTSuperFrameDemod` — payload, cell id, and frame sequence.

use crate::common::add_awgn;
use orion_sdr::demodulate::{DvbTRxSuperFrameError, DvbTSuperFrameDemod};
use orion_sdr::fec::PunctureRate;
use orion_sdr::modulate::{
    ConstellationOrder, DvbTFrameMod, DvbTSuperFrameMod, DvbTSuperFrameParams,
};
use orion_sdr::waveform::dvb_t::{DvbTLinkParams, GuardInterval};

fn sample_payload(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 37 + 11) & 0xff) as u8).collect()
}

fn params(
    constellation: ConstellationOrder,
    code_rate: PunctureRate,
    cell_id: u16,
) -> DvbTSuperFrameParams {
    DvbTSuperFrameParams {
        link: DvbTLinkParams {
            guard: GuardInterval::G1_8,
            constellation,
            code_rate,
        },
        cell_id,
    }
}

#[test]
fn roundtrip_dvb_t_super_frame_end_to_end() {
    // 4 frames of payload; recover the full payload, the 16-bit cell id, and the
    // implicit frame sequence (checked inside the demodulator).
    let p = params(ConstellationOrder::Qpsk, PunctureRate::R1_2, 0xBEEF);
    let payload = sample_payload(700); // split ~175 bytes/frame
    let sf = DvbTSuperFrameMod::new(p).modulate(&payload);

    // The whole super-frame is a uniform 4 × symbols_per_frame block.
    assert_eq!(sf.n_symbols(), 4 * sf.symbols_per_frame);
    assert_eq!(sf.iq.len(), sf.n_symbols() * sf.samples_per_symbol);

    let rx = DvbTSuperFrameDemod::new(p)
        .decode(&sf.iq, sf.symbols_per_frame, sf.frame_payload_lens)
        .expect("super-frame decode");
    assert_eq!(rx.payload, payload, "recovered super-frame payload");
    assert_eq!(rx.cell_id, 0xBEEF, "reassembled 16-bit cell id");
}

#[test]
fn dvb_t_super_frame_cell_id_split_across_frames() {
    // The 16-bit cell id must survive its split (hi byte in frames 1&3, lo in 2&4)
    // and reassembly — use a value whose two bytes differ.
    let p = params(ConstellationOrder::Qam16, PunctureRate::R3_4, 0x1234);
    let payload = sample_payload(400);
    let sf = DvbTSuperFrameMod::new(p).modulate(&payload);
    let rx = DvbTSuperFrameDemod::new(p)
        .decode(&sf.iq, sf.symbols_per_frame, sf.frame_payload_lens)
        .expect("super-frame decode");
    assert_eq!(rx.cell_id, 0x1234);
    assert_eq!(rx.payload, payload);
}

#[test]
fn roundtrip_dvb_t_super_frame_with_tx_lowpass() {
    // R16: the spectral mask propagates to the super-frame path, where it is
    // applied ONCE over the concatenated four frames — filtering each frame
    // separately would leave the filter's edge transient at all three interior
    // seams, which are continuous on air. The mask needs no decoding change; the
    // demod only supplies guard for its group delay via the window back-off.
    let p = params(ConstellationOrder::Qpsk, PunctureRate::R1_2, 0xBEEF);
    let cp_len = GuardInterval::G1_8.cp_len_2k(); // 256
    let backoff = 64usize; // inside the 85-sample scattered-pilot ceiling
    let lowpass = DvbTFrameMod::tx_lowpass_for_2k(89, 80.0); // group delay 44
    assert!(lowpass.fits_guard(cp_len, 0, backoff));

    let payload = sample_payload(700);
    let sf = DvbTSuperFrameMod::new(p)
        .with_tx_lowpass(lowpass)
        .modulate(&payload);

    // Same-length post-pass: the super-frame is still a uniform block.
    assert_eq!(sf.iq.len(), sf.n_symbols() * sf.samples_per_symbol);

    let rx = DvbTSuperFrameDemod::new(p)
        .with_rx_window_backoff(backoff)
        .decode(&sf.iq, sf.symbols_per_frame, sf.frame_payload_lens)
        .expect("band-limited super-frame decode");
    assert_eq!(rx.payload, payload, "recovered super-frame payload");
    assert_eq!(rx.cell_id, 0xBEEF, "reassembled 16-bit cell id");
}

#[test]
fn dvb_t_super_frame_mask_is_continuous_across_frame_seams() {
    // Filtering the concatenation (rather than each frame) is observable: the
    // samples straddling an interior frame boundary must differ from what
    // per-frame filtering would produce, since a per-frame filter would ramp its
    // state down and back up at every seam.
    let p = params(ConstellationOrder::Qpsk, PunctureRate::R1_2, 0x0102);
    let lowpass = DvbTFrameMod::tx_lowpass_for_2k(89, 80.0);
    let payload = sample_payload(700);

    let whole = DvbTSuperFrameMod::new(p)
        .with_tx_lowpass(lowpass)
        .modulate(&payload);
    // Per-frame filtering, for contrast: the same four frames, each masked alone.
    let per_frame = {
        let plain = DvbTSuperFrameMod::new(p).modulate(&payload);
        let frame_len = plain.iq.len() / 4;
        let mut out = plain.iq.clone();
        for chunk in out.chunks_mut(frame_len) {
            lowpass.apply(chunk);
        }
        out
    };
    assert_eq!(whole.iq.len(), per_frame.len());

    let seam = whole.iq.len() / 4; // first interior frame boundary
    let d = lowpass.group_delay();
    let worst = (seam - d..seam + d)
        .map(|i| (whole.iq[i] - per_frame[i]).norm())
        .fold(0.0f32, f32::max);
    assert!(
        worst > 1e-6,
        "the mask must run across the frame seam, not restart at it"
    );
}

#[test]
fn dvb_t_super_frame_survives_awgn() {
    let p = params(ConstellationOrder::Qpsk, PunctureRate::R1_2, 0x00A5);
    let payload = sample_payload(500);
    let sf = DvbTSuperFrameMod::new(p).modulate(&payload);
    let mut buf = sf.iq.clone();
    let sig_power: f32 = buf.iter().map(|s| s.norm_sqr()).sum::<f32>() / buf.len() as f32;
    add_awgn(&mut buf, sig_power * 0.03, 0x5F00_0777);

    let rx = DvbTSuperFrameDemod::new(p)
        .decode(&buf, sf.symbols_per_frame, sf.frame_payload_lens)
        .expect("super-frame AWGN decode");
    assert_eq!(rx.payload, payload);
    assert_eq!(rx.cell_id, 0x00A5);
}

#[test]
fn dvb_t_super_frame_uneven_split() {
    // A payload length not divisible by 4 (parts differ by a byte, then zero-pad
    // to a common length) must still round-trip exactly.
    for len in [1usize, 2, 3, 101, 599] {
        let p = params(ConstellationOrder::Qpsk, PunctureRate::R1_2, 0xABCD);
        let payload = sample_payload(len);
        let sf = DvbTSuperFrameMod::new(p).modulate(&payload);
        assert_eq!(sf.frame_payload_lens.iter().sum::<usize>(), len);
        let rx = DvbTSuperFrameDemod::new(p)
            .decode(&sf.iq, sf.symbols_per_frame, sf.frame_payload_lens)
            .expect("uneven super-frame decode");
        assert_eq!(rx.payload, payload, "len={len}");
        assert_eq!(rx.cell_id, 0xABCD);
    }
}

#[test]
fn dvb_t_super_frame_rejects_short_buffer() {
    let p = params(ConstellationOrder::Qpsk, PunctureRate::R1_2, 0);
    let payload = sample_payload(300);
    let sf = DvbTSuperFrameMod::new(p).modulate(&payload);
    // Truncate to just under three frames.
    let short = &sf.iq[..sf.symbols_per_frame * sf.samples_per_symbol * 3 - 1];
    let got =
        DvbTSuperFrameDemod::new(p).decode(short, sf.symbols_per_frame, sf.frame_payload_lens);
    assert!(matches!(
        got,
        Err(DvbTRxSuperFrameError::Incomplete | DvbTRxSuperFrameError::Frame { .. })
    ));
}
