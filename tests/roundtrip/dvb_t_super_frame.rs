// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// Conformant DVB-T super-frame roundtrip (EN 300 744 §4.4/§4.6): four consecutive
// frames (numbers 0..3) with alternating TPS sync + a 16-bit cell id split across
// them, modulated by `modulate::dvb_t_super_frame` and recovered by
// `demodulate::dvb_t_super_frame` — payload, cell id, and frame sequence.

use crate::common::add_awgn;
use orion_sdr::demodulate::{DvbTRxSuperFrameError, dvb_t_super_frame_demodulate};
use orion_sdr::fec::PunctureRate;
use orion_sdr::modulate::{ConstellationOrder, DvbTSuperFrameParams, dvb_t_super_frame_modulate};
use orion_sdr::waveform::dvb_t::GuardInterval;

fn sample_payload(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 37 + 11) & 0xff) as u8).collect()
}

fn params(
    constellation: ConstellationOrder,
    code_rate: PunctureRate,
    cell_id: u16,
) -> DvbTSuperFrameParams {
    DvbTSuperFrameParams {
        guard: GuardInterval::G1_8,
        constellation,
        code_rate,
        cell_id,
    }
}

#[test]
fn roundtrip_dvb_t_super_frame_end_to_end() {
    // 4 frames of payload; recover the full payload, the 16-bit cell id, and the
    // implicit frame sequence (checked inside the demodulator).
    let p = params(ConstellationOrder::Qpsk, PunctureRate::R1_2, 0xBEEF);
    let payload = sample_payload(700); // split ~175 bytes/frame
    let sf = dvb_t_super_frame_modulate(p, &payload);

    // The whole super-frame is a uniform 4 × symbols_per_frame block.
    assert_eq!(sf.n_symbols(), 4 * sf.symbols_per_frame);
    assert_eq!(sf.iq.len(), sf.n_symbols() * sf.samples_per_symbol);

    let rx = dvb_t_super_frame_demodulate(p, &sf.iq, sf.symbols_per_frame, sf.frame_payload_lens)
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
    let sf = dvb_t_super_frame_modulate(p, &payload);
    let rx = dvb_t_super_frame_demodulate(p, &sf.iq, sf.symbols_per_frame, sf.frame_payload_lens)
        .expect("super-frame decode");
    assert_eq!(rx.cell_id, 0x1234);
    assert_eq!(rx.payload, payload);
}

#[test]
fn dvb_t_super_frame_survives_awgn() {
    let p = params(ConstellationOrder::Qpsk, PunctureRate::R1_2, 0x00A5);
    let payload = sample_payload(500);
    let sf = dvb_t_super_frame_modulate(p, &payload);
    let mut buf = sf.iq.clone();
    let sig_power: f32 = buf.iter().map(|s| s.norm_sqr()).sum::<f32>() / buf.len() as f32;
    add_awgn(&mut buf, sig_power * 0.03, 0x5F00_0777);

    let rx = dvb_t_super_frame_demodulate(p, &buf, sf.symbols_per_frame, sf.frame_payload_lens)
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
        let sf = dvb_t_super_frame_modulate(p, &payload);
        assert_eq!(sf.frame_payload_lens.iter().sum::<usize>(), len);
        let rx =
            dvb_t_super_frame_demodulate(p, &sf.iq, sf.symbols_per_frame, sf.frame_payload_lens)
                .expect("uneven super-frame decode");
        assert_eq!(rx.payload, payload, "len={len}");
        assert_eq!(rx.cell_id, 0xABCD);
    }
}

#[test]
fn dvb_t_super_frame_rejects_short_buffer() {
    let p = params(ConstellationOrder::Qpsk, PunctureRate::R1_2, 0);
    let payload = sample_payload(300);
    let sf = dvb_t_super_frame_modulate(p, &payload);
    // Truncate to just under three frames.
    let short = &sf.iq[..sf.symbols_per_frame * sf.samples_per_symbol * 3 - 1];
    let got = dvb_t_super_frame_demodulate(p, short, sf.symbols_per_frame, sf.frame_payload_lens);
    assert!(matches!(
        got,
        Err(DvbTRxSuperFrameError::Incomplete | DvbTRxSuperFrameError::Frame { .. })
    ));
}
