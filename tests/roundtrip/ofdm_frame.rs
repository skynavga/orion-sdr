// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::common::add_awgn;
use num_complex::Complex32 as C32;
use orion_sdr::demodulate::ofdm_frame::ChainOutcome;
use orion_sdr::demodulate::{OfdmFrameDemod, OfdmFrameStreamDemod};
use orion_sdr::dsp::Rotator;
use orion_sdr::fec::{
    CrcKind, DecodeRule, FrameMetadata, FramePacket, HeaderFormat, InnerFec, InterleaverKind,
    OuterFec, ScramblerKind, ScramblerPos, SeedMode,
};
use orion_sdr::modulate::{CodecCache, ConstellationOrder, McsTable, OfdmConfig, OfdmFrameMod};
use orion_sdr::sync::OfdmPreamble;

fn plan_config() -> OfdmConfig {
    // A 64-point plan with all non-DC, non-Nyquist carriers as data.
    let n_fft = 64;
    let cp_len = 8;
    let half = (n_fft / 2) as i32;
    let data: Vec<i32> = (1..half).chain(-(half - 1)..0).collect();
    let plan = orion_sdr::multicarrier::CarrierPlan::new(n_fft, cp_len).with_data_carriers(data);
    OfdmConfig::new(plan, 48_000.0, 0.0, 1.0, ConstellationOrder::Bpsk)
}

fn preamble(cfg: &OfdmConfig) -> OfdmPreamble {
    OfdmPreamble::new(4, 16)
        .with_training_symbol(cfg.carrier_plan.n_fft(), cfg.carrier_plan.cp_len())
}

fn strip_preamble(cfg: &OfdmConfig, pre: &OfdmPreamble, iq: &[C32]) -> Vec<C32> {
    // OfdmFrameDemod expects IQ starting at the first post-preamble sample.
    let _ = cfg;
    iq[pre.total_len()..].to_vec()
}

fn sample_payload(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 37 + 11) & 0xff) as u8).collect()
}

#[test]
fn roundtrip_frame_noiseless_ldpc_bch() {
    let cfg = plan_config();
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    for mcs_index in 0..table.len() as u8 {
        let payload = sample_payload(40);
        let frame = FramePacket::new(
            FrameMetadata::new(0x1234_5678 + mcs_index as u32, mcs_index),
            payload.clone(),
        );
        let iq = modu.modulate_frame(&frame, 0);
        let body = strip_preamble(&cfg, modu.preamble(), &iq);
        let got = OfdmFrameDemod::new(cfg.clone(), table.clone())
            .decode(&body)
            .expect("decode");
        assert_eq!(got.payload, payload, "mcs {mcs_index}: payload");
        assert_eq!(got.metadata.mcs_index, mcs_index);
        assert_eq!(got.metadata.sequence_num, 0x1234_5678 + mcs_index as u32);
    }
}

#[test]
fn stream_frame_with_rx_window_backoff() {
    // A nonzero RX FFT-window back-off slides the window earlier into the guard.
    // This introduces a per-bin linear phase ramp (FFT shift theorem), so it is
    // only transparent on the *equalized* path, where the training-symbol
    // estimate is measured at the SAME back-off and divides the ramp back out.
    // The streaming demod is that path; prove a clean frame still round-trips
    // with the back-off set on the shared config.
    let cfg = plan_config().with_rx_window_backoff(3); // cp_len is 8
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(30);
    let frame = FramePacket::new(FrameMetadata::new(0xABCD, 1), payload.clone()); // mcs 1 = QPSK
    let mut buf = vec![C32::default(); 24];
    buf.extend_from_slice(&modu.modulate_frame(&frame, 0));
    buf.extend(vec![C32::default(); 64]);

    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
    let frames: Vec<_> = rx.feed(&buf).into_iter().filter_map(|r| r.ok()).collect();
    assert_eq!(
        frames.len(),
        1,
        "clean frame decodes with RX window back-off"
    );
    assert_eq!(frames[0].packet.payload, payload);
}

#[test]
fn stream_frame_with_symbol_windowing() {
    // End-to-end B2: a TX symbol-window taper of roll_off = cp_len/2, paired with
    // the matching RX window back-off cp_len/2, must still decode cleanly — the
    // taper falls entirely in guard samples the backed-off RX window discards, so
    // it is transparent to the demod while shaping the transmitted spectrum.
    let cp_len = 8; // from plan_config
    let cfg = plan_config()
        .with_symbol_window(cp_len / 2) // 4-sample raised-cosine taper per edge
        .with_rx_window_backoff(cp_len / 2); // 4-sample back-off pairs with it
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(30);
    let frame = FramePacket::new(FrameMetadata::new(0x5A5A, 1), payload.clone()); // mcs 1 = QPSK
    let mut buf = vec![C32::default(); 24];
    buf.extend_from_slice(&modu.modulate_frame(&frame, 0));
    buf.extend(vec![C32::default(); 64]);

    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
    let frames: Vec<_> = rx.feed(&buf).into_iter().filter_map(|r| r.ok()).collect();
    assert_eq!(
        frames.len(),
        1,
        "windowed frame decodes with matched back-off"
    );
    assert_eq!(frames[0].packet.payload, payload);
}

/// A plan with real unoccupied bandwidth for a spectral mask to work in: a
/// 128-point grid with a 20-carrier edge guard (so the occupied half-width is
/// 43 of 64) and a quarter-length guard interval.
fn masked_plan_config() -> OfdmConfig {
    let (n_fft, cp_len) = (128usize, 32usize);
    let plan =
        orion_sdr::multicarrier::CarrierPlan::new(n_fft, cp_len).with_contiguous_data(20, false);
    OfdmConfig::new(plan, 48_000.0, 0.0, 1.0, ConstellationOrder::Bpsk)
}

fn masked_preamble(cfg: &OfdmConfig) -> OfdmPreamble {
    OfdmPreamble::new(4, 32)
        .with_training_symbol(cfg.carrier_plan.n_fft(), cfg.carrier_plan.cp_len())
}

#[test]
fn stream_frame_with_tx_lowpass() {
    // R15: the TX baseband spectral mask is applied across the whole assembled
    // stream (preamble included). It is absorbed by the equalizer as an ordinary
    // linear channel, so the demod needs no new decoding step — but its group
    // delay must land in guard the receiver discards, which is what the RX
    // window back-off provides. Prove a masked frame still round-trips cleanly.
    let cp_len = 32usize;
    let taps = 21usize; // group delay 10
    let cfg = masked_plan_config()
        .with_tx_lowpass_null_band(taps, 40.0)
        .with_rx_window_backoff(cp_len / 2); // slack min(16, 16) = 16 >= 10
    let lowpass = cfg.tx_lowpass.expect("mask configured");
    assert!(
        lowpass.fits_guard(cp_len, 0, cp_len / 2),
        "test config must stay inside the guard budget"
    );

    let pre = masked_preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(48);
    let frame = FramePacket::new(FrameMetadata::new(0x00C1, 1), payload.clone()); // mcs 1 = QPSK
    let mut buf = vec![C32::default(); 40];
    buf.extend_from_slice(&modu.modulate_frame(&frame, 0));
    buf.extend(vec![C32::default(); 128]);

    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
    let frames: Vec<_> = rx.feed(&buf).into_iter().filter_map(|r| r.ok()).collect();
    assert_eq!(frames.len(), 1, "band-limited frame decodes");
    assert_eq!(frames[0].packet.payload, payload);
}

#[test]
fn stream_frame_with_tx_lowpass_and_symbol_windowing() {
    // R15/R17: the two TX shaping levers stack, and they share one guard budget:
    // `roll_off + group_delay <= min(cp_len - b, b)`. Configure both inside it
    // and the frame must still decode.
    let cp_len = 32usize;
    let (taps, roll_off) = (15usize, 6usize); // group delay 7; 7 + 6 = 13 <= 16
    let cfg = masked_plan_config()
        .with_tx_lowpass_null_band(taps, 40.0)
        .with_symbol_window(roll_off)
        .with_rx_window_backoff(cp_len / 2);
    assert!(
        cfg.tx_lowpass
            .expect("mask configured")
            .fits_guard(cp_len, roll_off, cp_len / 2),
        "mask + taper must fit the guard together"
    );

    let pre = masked_preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(48);
    let frame = FramePacket::new(FrameMetadata::new(0x00C2, 1), payload.clone());
    let mut buf = vec![C32::default(); 40];
    buf.extend_from_slice(&modu.modulate_frame(&frame, 0));
    buf.extend(vec![C32::default(); 128]);

    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
    let frames: Vec<_> = rx.feed(&buf).into_iter().filter_map(|r| r.ok()).collect();
    assert_eq!(frames.len(), 1, "masked + windowed frame decodes");
    assert_eq!(frames[0].packet.payload, payload);
}

#[test]
fn cofdm_training_hold_takes_any_back_off_the_guard_allows() {
    // The COFDM streaming demod estimates the channel once from the training
    // symbol, measured at the same back-off — a per-bin estimate at full
    // frequency resolution, so it absorbs the back-off's phase ramp exactly at
    // ANY `b`, right up to the whole cyclic prefix.
    //
    // This is the opposite of the per-symbol *pilot-interpolated* path DVB-T
    // uses, where the ramp must be reconstructed between references spaced 12
    // carriers apart and aliases past `n_fft/(2*12)` (see
    // `dvb_t_rx_window_backoff_is_capped_by_the_pilot_grid_not_the_guard`).
    // Recorded here because it inverts the natural assumption: for window
    // back-off, holding one training estimate is the *stronger* option, so
    // offering COFDM a pilot-interpolated equalizer would shrink its shaping
    // budget rather than widen it.
    let cp_len = 32usize;
    for backoff in [cp_len / 2, 3 * cp_len / 4, cp_len] {
        let cfg = masked_plan_config().with_rx_window_backoff(backoff);
        let pre = masked_preamble(&cfg);
        let table = McsTable::default_ladder();
        let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

        let payload = sample_payload(32);
        let frame = FramePacket::new(FrameMetadata::new(0x00C3, 1), payload.clone());
        let mut buf = vec![C32::default(); 40];
        buf.extend_from_slice(&modu.modulate_frame(&frame, 0));
        buf.extend(vec![C32::default(); 128]);

        let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
        let frames: Vec<_> = rx.feed(&buf).into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(frames.len(), 1, "back-off {backoff} should decode");
        assert_eq!(frames[0].packet.payload, payload, "back-off {backoff}");
    }
}

#[test]
fn tx_lowpass_defaults_off_and_is_byte_identical() {
    // The Track A/B/C regression guard: every lever is opt-in, so a config that
    // does not ask for a mask emits exactly what it emitted before.
    let cfg = masked_plan_config();
    assert!(cfg.tx_lowpass.is_none(), "mask is off by default");
    let pre = masked_preamble(&cfg);
    let table = McsTable::default_ladder();
    let frame = FramePacket::new(FrameMetadata::new(9, 0), sample_payload(24));

    let plain = OfdmFrameMod::new(cfg.clone(), table.clone(), pre).modulate_frame(&frame, 0);
    let also_plain = OfdmFrameMod::new(cfg, table, pre).modulate_frame(&frame, 0);
    assert_eq!(plain, also_plain);
}

#[test]
fn roundtrip_frame_awgn() {
    let cfg = plan_config();
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    // MCS 0 (BPSK + rate-1/2 LDPC + BCH) is the most robust. Noise is scaled
    // relative to the signal's own (low) per-sample power, as the uncoded OFDM
    // test does — but at 10%, twice the uncoded test's 5%, since the FEC here
    // buys coding gain the bare pipeline doesn't have.
    let payload = sample_payload(32);
    let frame = FramePacket::new(FrameMetadata::new(7, 0), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let mut body = strip_preamble(&cfg, modu.preamble(), &iq);
    let sig_power: f32 = body.iter().map(|s| s.norm_sqr()).sum::<f32>() / body.len() as f32;
    add_awgn(&mut body, sig_power * 0.10, 0xC0FFEE);
    let got = OfdmFrameDemod::new(cfg.clone(), table.clone())
        .decode(&body)
        .expect("decode under AWGN");
    assert_eq!(got.payload, payload);
}

#[test]
fn roundtrip_frame_awgn_scaled_min_sum() {
    // The opt-in scaled-min-sum LDPC decode rule must recover the same payload
    // through the full COFDM frame decoder. The payload honors the configured
    // rule; the header always uses sum-product (decoded before the rule matters
    // for robustness). Same modest AWGN as `roundtrip_frame_awgn`.
    let cfg = plan_config().with_ldpc_decode_rule(DecodeRule::ScaledMinSum(0.75));
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(32);
    let frame = FramePacket::new(FrameMetadata::new(9, 0), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let mut body = strip_preamble(&cfg, modu.preamble(), &iq);
    let sig_power: f32 = body.iter().map(|s| s.norm_sqr()).sum::<f32>() / body.len() as f32;
    add_awgn(&mut body, sig_power * 0.10, 0xC0FFEE);
    let got = OfdmFrameDemod::new(cfg.clone(), table.clone())
        .decode(&body)
        .expect("min-sum decode under AWGN");
    assert_eq!(got.payload, payload);
}

#[test]
fn frame_header_crc_catches_corruption() {
    let cfg = plan_config();
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let frame = FramePacket::new(FrameMetadata::new(1, 0), sample_payload(16));
    let iq = modu.modulate_frame(&frame, 0);
    let mut body = strip_preamble(&cfg, modu.preamble(), &iq);
    // Massively corrupt the header region (first few symbols) beyond FEC.
    for s in body.iter_mut().take(200) {
        *s = C32::new(-s.re, -s.im);
    }
    let res = OfdmFrameDemod::new(cfg.clone(), table.clone()).decode(&body);
    assert!(
        res.is_err(),
        "corrupted header must not decode to a valid frame"
    );
}

#[test]
fn roundtrip_frame_no_header_rejected_by_batch() {
    // NoHeader is not supported by the batch entry point (needs out-of-band MCS).
    let cfg = plan_config().with_header_format(HeaderFormat::NoHeader);
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let frame = FramePacket::new(FrameMetadata::new(1, 0), sample_payload(8));
    let iq = modu.modulate_frame(&frame, 0);
    let body = strip_preamble(&cfg, modu.preamble(), &iq);
    assert!(
        OfdmFrameDemod::new(cfg.clone(), table.clone())
            .decode(&body)
            .is_err()
    );
}

#[test]
fn roundtrip_frame_scrambler_positions() {
    let table = McsTable::default_ladder();
    for pos in [ScramblerPos::BeforeOuterFec, ScramblerPos::AfterInnerFec] {
        let cfg = plan_config()
            .with_scrambler(ScramblerKind::Additive {
                poly: 0b1001,
                width: 7,
                seed: SeedMode::Fixed(0x7F),
            })
            .with_scrambler_pos(pos);
        let pre = preamble(&cfg);
        let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
        let payload = sample_payload(24);
        let frame = FramePacket::new(FrameMetadata::new(2, 0), payload.clone());
        let iq = modu.modulate_frame(&frame, 0);
        let body = strip_preamble(&cfg, modu.preamble(), &iq);
        let got = OfdmFrameDemod::new(cfg.clone(), table.clone())
            .decode(&body)
            .expect("decode with scrambler");
        assert_eq!(got.payload, payload, "scrambler pos {pos:?}");
    }
}

#[test]
fn roundtrip_frame_per_frame_random_seed() {
    let cfg = plan_config().with_scrambler(ScramblerKind::Additive {
        poly: 0b1001,
        width: 7,
        seed: SeedMode::PerFrameRandom,
    });
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let payload = sample_payload(20);
    let frame = FramePacket::new(FrameMetadata::new(3, 0), payload.clone());
    let seed = 0xABCD_1234u32;
    let iq = modu.modulate_frame(&frame, seed);
    let body = strip_preamble(&cfg, modu.preamble(), &iq);
    let got = OfdmFrameDemod::new(cfg.clone(), table.clone())
        .decode(&body)
        .expect("decode with per-frame seed");
    assert_eq!(got.payload, payload);
}

#[test]
fn roundtrip_frame_with_interleavers() {
    let cfg = plan_config()
        .with_inner_interleaver(InterleaverKind::Block { rows: 8, cols: 16 })
        .with_outer_interleaver(InterleaverKind::Block { rows: 4, cols: 8 });
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let payload = sample_payload(30);
    let frame = FramePacket::new(FrameMetadata::new(4, 1), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let body = strip_preamble(&cfg, modu.preamble(), &iq);
    let got = OfdmFrameDemod::new(cfg.clone(), table.clone())
        .decode(&body)
        .expect("decode with interleavers");
    assert_eq!(got.payload, payload);
}

#[test]
fn roundtrip_frame_forney_outer_interleaver() {
    // The streaming Forney convolutional interleaver as the outer (byte-domain)
    // interleaver, driven in reset-per-frame mode by the frame layer. Pair it
    // with a Reed–Solomon outer + convolutional inner (the DVB-style
    // concatenation it belongs to). Use small DVB-like dims so the round-trip
    // delay stays modest for the tiny test plan.
    use orion_sdr::fec::{ConvCode, PunctureRate};
    use orion_sdr::modulate::Mcs;

    let cfg = plan_config().with_outer_interleaver(InterleaverKind::Convolutional {
        branches: 4,
        depth: 3,
    });
    let pre = preamble(&cfg);
    let table = McsTable::new(vec![Mcs::new(
        ConstellationOrder::Qpsk,
        InnerFec::Convolutional {
            rate: PunctureRate::R1_2,
            code: ConvCode::K5,
        },
        OuterFec::ReedSolomon { n: 60, n_parity: 8 },
    )]);
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let payload = sample_payload(40);
    let frame = FramePacket::new(FrameMetadata::new(6, 0), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let body = strip_preamble(&cfg, modu.preamble(), &iq);
    let got = OfdmFrameDemod::new(cfg.clone(), table.clone())
        .decode(&body)
        .expect("decode with Forney interleaver");
    assert_eq!(got.payload, payload);
}

#[test]
fn roundtrip_frame_dvb_t_payload_fec_chain() {
    // The full DVB-T-conformant *payload* FEC chain composed end to end:
    // Reed–Solomon(204,188) outer + Forney (I=12, M=17) outer interleaver +
    // K=7 (0o171,0o133) punctured convolutional inner, over QPSK. Proves the
    // Phase-0 streaming Forney interleaver and the K=7 inner code integrate
    // through the frame layer. (Pilots/TPS and the 2K carrier map come later;
    // this covers the coding chain.)
    use orion_sdr::fec::{ConvCode, PunctureRate};
    use orion_sdr::modulate::Mcs;

    let cfg = plan_config().with_outer_interleaver(InterleaverKind::Convolutional {
        branches: 12,
        depth: 17,
    });
    let pre = preamble(&cfg);
    let table = McsTable::new(vec![Mcs::new(
        ConstellationOrder::Qpsk,
        InnerFec::Convolutional {
            rate: PunctureRate::R1_2,
            code: ConvCode::DvbK7,
        },
        OuterFec::ReedSolomon {
            n: 204,
            n_parity: 16,
        },
    )]);
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    // One RS(204,188) codeword's worth of info (188 − 4 CRC-32 bytes = 184).
    let payload = sample_payload(184);
    let frame = FramePacket::new(FrameMetadata::new(8, 0), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let body = strip_preamble(&cfg, modu.preamble(), &iq);
    let got = OfdmFrameDemod::new(cfg.clone(), table.clone())
        .decode(&body)
        .expect("DVB-T payload FEC chain decode");
    assert_eq!(got.payload, payload);
}

#[test]
fn roundtrip_frame_no_fec_no_crc() {
    // Bare frame: no FEC, no CRC on payload — still round-trips noiselessly.
    let cfg = plan_config().with_payload_crc(CrcKind::None);
    let pre = preamble(&cfg);
    let table = McsTable::new(vec![orion_sdr::modulate::Mcs::new(
        ConstellationOrder::Qpsk,
        InnerFec::None,
        OuterFec::None,
    )]);
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let payload = sample_payload(16);
    let frame = FramePacket::new(FrameMetadata::new(5, 0), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let body = strip_preamble(&cfg, modu.preamble(), &iq);
    let got = OfdmFrameDemod::new(cfg.clone(), table.clone())
        .decode(&body)
        .expect("decode bare frame");
    assert_eq!(got.payload, payload);
}

// ── Streaming receiver (OfdmFrameStreamDemod) ──────────────────────────────

fn apply_fir_channel(iq: &[C32], taps: &[C32]) -> Vec<C32> {
    let mut out = vec![C32::default(); iq.len()];
    for (n, &x) in iq.iter().enumerate() {
        for (k, &h) in taps.iter().enumerate() {
            if n + k < out.len() {
                out[n + k] += x * h;
            }
        }
    }
    out
}

#[test]
fn stream_frame_unknown_start_and_cfo() {
    let cfg = plan_config();
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(40);
    let frame = FramePacket::new(FrameMetadata::new(42, 0), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);

    // Place at an unknown offset with trailing silence, then apply a CFO
    // within the fractional capture range.
    let fs = cfg.fs;
    let mut buf = vec![C32::default(); 101];
    buf.extend_from_slice(&iq);
    buf.extend(vec![C32::default(); 64]);
    let capture_hz = fs / (2.0 * pre.repeat_len as f32);
    let mut rot = Rotator::new(capture_hz * 0.25, fs);
    let mut with_cfo = vec![C32::default(); buf.len()];
    rot.rotate_block(&buf, &mut with_cfo);

    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
    let frames = rx.feed(&with_cfo);
    let ok: Vec<_> = frames.into_iter().filter_map(|r| r.ok()).collect();
    assert_eq!(ok.len(), 1, "exactly one frame should decode");
    assert_eq!(ok[0].packet.payload, payload);
    assert_eq!(ok[0].packet.metadata.sequence_num, 42);
    assert!(
        ok[0].diagnostics.cfo_hz.is_some(),
        "cfo diagnostic populated"
    );
    assert!(ok[0].diagnostics.timing_offset_samples.is_some());
}

#[test]
fn stream_back_to_back_frames() {
    let cfg = plan_config();
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let p0 = sample_payload(24);
    let p1 = sample_payload(32);
    let f0 = FramePacket::new(FrameMetadata::new(1, 0), p0.clone());
    let f1 = FramePacket::new(FrameMetadata::new(2, 0), p1.clone());

    let mut buf = vec![C32::default(); 40];
    buf.extend_from_slice(&modu.modulate_frame(&f0, 0));
    buf.extend_from_slice(&modu.modulate_frame(&f1, 0));
    buf.extend(vec![C32::default(); 64]);

    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
    let frames: Vec<_> = rx.feed(&buf).into_iter().filter_map(|r| r.ok()).collect();
    assert_eq!(frames.len(), 2, "two frames drained from one buffer");
    assert_eq!(frames[0].packet.payload, p0);
    assert_eq!(frames[1].packet.payload, p1);
    assert_eq!(frames[0].packet.metadata.sequence_num, 1);
    assert_eq!(frames[1].packet.metadata.sequence_num, 2);
}

#[test]
fn stream_frame_split_across_feeds() {
    let cfg = plan_config();
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(48);
    let frame = FramePacket::new(FrameMetadata::new(9, 0), payload.clone());
    let mut buf = vec![C32::default(); 32];
    buf.extend_from_slice(&modu.modulate_frame(&frame, 0));
    buf.extend(vec![C32::default(); 64]);

    // Split roughly in the middle of the frame — the first feed must NOT emit,
    // the second completes it.
    let split = buf.len() / 2;
    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
    let first = rx.feed(&buf[..split]);
    assert!(
        first.iter().all(|r| r.is_err()) || first.is_empty(),
        "partial frame must not decode on the first feed"
    );
    let second: Vec<_> = rx
        .feed(&buf[split..])
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();
    assert_eq!(second.len(), 1, "frame completes on the second feed");
    assert_eq!(second[0].packet.payload, payload);
}

#[test]
fn stream_frame_multipath_channel() {
    // A 2-tap frequency-selective channel with delay spread <= cp_len, decoded
    // via the training-symbol channel estimate. The channel is tuned to be
    // load-bearing: this same frame decodes WITH the training-symbol estimate
    // (below) but NOT without it (`stream_multipath_needs_channel_estimate`).
    let cfg = plan_config();
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(30);
    let frame = FramePacket::new(FrameMetadata::new(3, 1), payload.clone()); // mcs 1 = QPSK
    let mut buf = vec![C32::default(); 24];
    buf.extend_from_slice(&modu.modulate_frame(&frame, 0));
    buf.extend(vec![C32::default(); 64]);

    let taps = multipath_taps();
    assert!(taps.len() - 1 <= cfg.carrier_plan.cp_len());
    let channeled = apply_fir_channel(&buf, &taps);

    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
    let frames: Vec<_> = rx
        .feed(&channeled)
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();
    assert_eq!(
        frames.len(),
        1,
        "multipath frame decodes via training estimate"
    );
    assert_eq!(frames[0].packet.payload, payload);
}

/// The 2-tap channel used by the multipath tests — strong enough that the
/// per-carrier channel estimate is required for a correct decode.
/// A harsher echo than [`multipath_taps`], strong enough to put visible errors
/// on the demapper without equalization. The milder set no longer does: with a
/// band-limited preamble the link is clean enough that BPSK rides out its ~18
/// degrees of channel-induced phase deviation with zero bit errors.
fn harsh_multipath_taps() -> [C32; 3] {
    [
        C32::new(0.85, 0.0),
        C32::new(0.0, 0.0),
        C32::new(0.5, -0.25),
    ]
}

fn multipath_taps() -> [C32; 3] {
    [
        C32::new(0.85, 0.0),
        C32::new(0.0, 0.0),
        C32::new(0.25, -0.125),
    ]
}

#[test]
fn multipath_without_a_channel_estimate_costs_link_margin() {
    // Was `stream_multipath_needs_channel_estimate`, which asserted the frame
    // must FAIL to decode without a training symbol. That premise turned out to
    // be false, and only held because acceptance was gated on `inner_ok`: the
    // frame decodes to a *byte-exact* payload, CRC-32 verified, with both FEC
    // stages reporting non-convergence and 14.7% of channel bits wrong. The
    // concatenated code recovers it; the old gate simply discarded it.
    //
    // What equalization actually buys is margin, and that is now measurable —
    // so assert the thing the original comment meant.
    let payload = sample_payload(30);
    let cber_for = |pre: OfdmPreamble| -> f32 {
        let cfg = plan_config();
        let table = McsTable::default_ladder();
        let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
        let frame = FramePacket::new(FrameMetadata::new(3, 1), payload.clone());
        let mut buf = vec![C32::default(); 24];
        buf.extend_from_slice(&modu.modulate_frame(&frame, 0));
        buf.extend(vec![C32::default(); 64]);
        let channeled = apply_fir_channel(&buf, &harsh_multipath_taps());

        let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre).with_error_rates(true);
        let f = rx
            .feed(&channeled)
            .into_iter()
            .filter_map(|r| r.ok())
            .next()
            .expect("the concatenated code recovers this channel either way");
        assert_eq!(f.packet.payload, payload, "payload must be exact");
        f.diagnostics.channel_ber.expect("opted in")
    };

    let equalized = cber_for(preamble(&plan_config()));
    let blind = cber_for(OfdmPreamble::new(4, 16)); // no training symbol

    // Measured on the harsher channel: 0% equalized against ~4.4% blind — the
    // training-symbol estimate removes the echo entirely, where without it the
    // demapper carries real errors. Bounds sit around those figures.
    assert!(
        equalized < 0.005,
        "equalization should pull the channel well inside what the FEC absorbs, \
         got {equalized:.3}"
    );
    assert!(
        blind > 0.01 && blind > 3.0 * equalized.max(0.001),
        "equalization should materially cut the channel error rate: \
         blind {blind:.3} vs equalized {equalized:.3}"
    );
}

#[test]
fn stream_corrupted_payload_reports_error() {
    let cfg = plan_config();
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let frame = FramePacket::new(FrameMetadata::new(5, 0), sample_payload(40));
    let mut buf = vec![C32::default(); 16];
    let frame_iq = modu.modulate_frame(&frame, 0);
    let header_end = pre.total_len() + 12 * cfg.samples_per_ofdm_symbol();
    buf.extend_from_slice(&frame_iq);
    buf.extend(vec![C32::default(); 64]);

    // Corrupt deep in the payload (past the header) beyond the FEC's reach.
    for s in buf.iter_mut().skip(16 + header_end).take(400) {
        *s = C32::new(-s.re * 3.0, -s.im * 3.0);
    }

    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
    let results = rx.feed(&buf);
    // No valid frame; at least one error reported (payload CRC/FEC failure).
    assert!(
        results.iter().all(|r| r.is_err()),
        "corrupted payload must not yield Ok"
    );
    assert!(!results.is_empty(), "an error should be reported");
}

// ── DVB-style concatenated RS + convolutional frame ────────────────────────

#[test]
fn roundtrip_frame_rs_convolutional() {
    use orion_sdr::fec::{ConvCode, PunctureRate};
    use orion_sdr::modulate::Mcs;

    let cfg = plan_config();
    let pre = preamble(&cfg);
    // A DVB-style concatenation: Reed–Solomon outer + punctured convolutional
    // inner. Use a shortened RS so a small frame spans a whole codeword.
    let table = McsTable::new(vec![
        Mcs::new(
            ConstellationOrder::Qpsk,
            InnerFec::Convolutional {
                rate: PunctureRate::R1_2,
                code: ConvCode::K5,
            },
            OuterFec::ReedSolomon { n: 60, n_parity: 8 },
        ),
        Mcs::new(
            ConstellationOrder::Qpsk,
            InnerFec::Convolutional {
                rate: PunctureRate::R3_4,
                code: ConvCode::K5,
            },
            OuterFec::ReedSolomon { n: 60, n_parity: 8 },
        ),
    ]);
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    for mcs_index in 0..table.len() as u8 {
        let payload = sample_payload(40);
        let frame = FramePacket::new(
            FrameMetadata::new(100 + mcs_index as u32, mcs_index),
            payload.clone(),
        );
        let iq = modu.modulate_frame(&frame, 0);
        let body = strip_preamble(&cfg, modu.preamble(), &iq);
        let got = OfdmFrameDemod::new(cfg.clone(), table.clone())
            .decode(&body)
            .expect("RS+conv decode");
        assert_eq!(got.payload, payload, "mcs {mcs_index}: RS+conv payload");
    }
}

#[test]
fn roundtrip_frame_rs_convolutional_awgn() {
    use orion_sdr::fec::{ConvCode, PunctureRate};
    use orion_sdr::modulate::Mcs;

    let cfg = plan_config();
    let pre = preamble(&cfg);
    let table = McsTable::new(vec![Mcs::new(
        ConstellationOrder::Qpsk,
        InnerFec::Convolutional {
            rate: PunctureRate::R1_2,
            code: ConvCode::K5,
        },
        OuterFec::ReedSolomon { n: 60, n_parity: 8 },
    )]);
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(32);
    let frame = FramePacket::new(FrameMetadata::new(7, 0), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let mut body = strip_preamble(&cfg, modu.preamble(), &iq);
    let sig_power: f32 = body.iter().map(|s| s.norm_sqr()).sum::<f32>() / body.len() as f32;
    add_awgn(&mut body, sig_power * 0.06, 0xBADC0DE);
    let got = OfdmFrameDemod::new(cfg.clone(), table.clone())
        .decode(&body)
        .expect("RS+conv AWGN decode");
    assert_eq!(got.payload, payload);
}

// ── QAM-16 over RS + convolutional (higher-order constellation) ────────────
//
// These mirror the QPSK RS+conv tests but with a QAM-16 payload, to exercise
// the higher-order constellation through the concatenated FEC. Noiseless and
// AWGN decode cleanly; multipath is bounded — a QAM-16 payload tolerates a
// *milder* frequency-selective channel than QPSK before the zero-forcing
// equalizer's residual error exceeds QAM-16's tighter amplitude margins (see
// the "High-order-QAM multipath performance" open item in the plan). The
// multipath test below uses a channel within that bound.

fn qam_rs_conv_table() -> McsTable {
    use orion_sdr::fec::{ConvCode, PunctureRate};
    use orion_sdr::modulate::Mcs;
    McsTable::new(vec![Mcs::new(
        ConstellationOrder::Qam16,
        InnerFec::Convolutional {
            rate: PunctureRate::R1_2,
            code: ConvCode::K5,
        },
        OuterFec::ReedSolomon { n: 60, n_parity: 8 },
    )])
}

/// A milder 2-tap channel than `multipath_taps()` — chosen so a QAM-16 payload
/// still decodes after zero-forcing equalization (QAM-16 has less margin for
/// residual equalization error than QPSK).
fn mild_multipath_taps() -> [C32; 3] {
    [
        C32::new(0.9, 0.0),
        C32::new(0.0, 0.0),
        C32::new(0.15, -0.075),
    ]
}

#[test]
fn roundtrip_frame_qam16_rs_conv_noiseless() {
    let cfg = plan_config();
    let pre = preamble(&cfg);
    let table = qam_rs_conv_table();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(40);
    let frame = FramePacket::new(FrameMetadata::new(200, 0), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let body = strip_preamble(&cfg, modu.preamble(), &iq);
    let got = OfdmFrameDemod::new(cfg.clone(), table.clone())
        .decode(&body)
        .expect("QAM-16 RS+conv noiseless decode");
    assert_eq!(got.payload, payload);
}

#[test]
fn roundtrip_frame_qam16_rs_conv_awgn() {
    let cfg = plan_config();
    let pre = preamble(&cfg);
    let table = qam_rs_conv_table();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(32);
    let frame = FramePacket::new(FrameMetadata::new(201, 0), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let mut body = strip_preamble(&cfg, modu.preamble(), &iq);
    let sig_power: f32 = body.iter().map(|s| s.norm_sqr()).sum::<f32>() / body.len() as f32;
    // QAM-16 has denser constellation points than QPSK, so a lower noise level.
    add_awgn(&mut body, sig_power * 0.04, 0x1CE_C0DE);
    let got = OfdmFrameDemod::new(cfg.clone(), table.clone())
        .decode(&body)
        .expect("QAM-16 RS+conv AWGN decode");
    assert_eq!(got.payload, payload);
}

#[test]
fn stream_frame_qam16_rs_conv_multipath() {
    // QAM-16 over RS+conv through a mild frequency-selective channel, decoded
    // via the training-symbol estimate. The channel is within the bound QAM-16
    // tolerates (a stronger channel — e.g. `multipath_taps()` — exceeds it; see
    // the plan's high-order-QAM open item).
    let cfg = plan_config();
    let pre = preamble(&cfg);
    let table = qam_rs_conv_table();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(30);
    let frame = FramePacket::new(FrameMetadata::new(202, 0), payload.clone());
    let mut buf = vec![C32::default(); 24];
    buf.extend_from_slice(&modu.modulate_frame(&frame, 0));
    buf.extend(vec![C32::default(); 64]);

    let taps = mild_multipath_taps();
    assert!(taps.len() - 1 <= cfg.carrier_plan.cp_len());
    let channeled = apply_fir_channel(&buf, &taps);

    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
    let frames: Vec<_> = rx
        .feed(&channeled)
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();
    assert_eq!(frames.len(), 1, "QAM-16 RS+conv multipath frame decodes");
    assert_eq!(frames[0].packet.payload, payload);
}

// ── Modulator gain reaches the preamble ────────────────────────────────────
//
// `generate_ofdm_preamble` used to ignore its config, emitting the S&C repeats
// and the training symbol at unit amplitude while `OfdmMod` scaled every data
// symbol by `cfg.gain`. A caller using a large gain — as a synthetic wideband
// source must, since bare OFDM at unit gain sits below typical detection
// thresholds — then transmitted a frame its own receiver could not acquire.

/// A frame at the given gain, with a little lead-in and trailing silence.
fn framed_at_gain(gain: f32) -> (OfdmConfig, OfdmPreamble, McsTable, Vec<u8>, Vec<C32>) {
    let cfg = OfdmConfig::new(
        plan_config().carrier_plan.clone(),
        48_000.0,
        0.0,
        gain,
        ConstellationOrder::Bpsk,
    );
    let pre = preamble(&cfg);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let payload = sample_payload(30);
    let frame = FramePacket::new(FrameMetadata::new(0x2468, 1), payload.clone());
    let mut buf = vec![C32::default(); 24];
    buf.extend_from_slice(&modu.modulate_frame(&frame, 0));
    buf.extend(vec![C32::default(); 256]);
    (cfg, pre, table, payload, buf)
}

#[test]
fn preamble_and_payload_share_one_amplitude_scale() {
    // Every segment must scale together. A preamble left at unit amplitude
    // beside a scaled payload breaks acquisition *and* channel estimation.
    let rms = |s: &[C32]| (s.iter().map(|c| c.norm_sqr()).sum::<f32>() / s.len() as f32).sqrt();
    let (_, pre, _, _, unit) = framed_at_gain(1.0);
    let (_, _, _, _, loud) = framed_at_gain(64.0);

    let reps = pre.num_repeats * pre.repeat_len;
    let lead = 24; // silence prepended by the fixture
    for (name, range) in [
        ("S&C repeats", lead..lead + reps),
        ("training symbol", lead + reps..lead + pre.total_len()),
    ] {
        let (u, l) = (rms(&unit[range.clone()]), rms(&loud[range]));
        assert!(
            (l / u - 64.0).abs() < 0.01,
            "{name}: gain reached the body but not here — scaled {:.3}x, want 64x",
            l / u
        );
    }
}

#[test]
fn a_high_gain_frame_is_acquirable_by_its_own_receiver() {
    // The end-to-end consequence, and the regression that matters: at a gain
    // typical of a wideband synthetic source, the streaming receiver must still
    // sync, equalize and decode. Before the fix this returned nothing at all —
    // not even an error — because no sync candidate cleared the threshold.
    for gain in [1.0_f32, 8.0, 64.0, 121.0] {
        let (cfg, pre, table, payload, buf) = framed_at_gain(gain);
        let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
        let frames: Vec<_> = rx.feed(&buf).into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(frames.len(), 1, "gain {gain}: expected exactly one frame");
        assert_eq!(frames[0].packet.payload, payload, "gain {gain}: payload");
    }
}

// ── Acquisition and channel diagnostics ────────────────────────────────────
//
// The streaming receiver measured a sync score and a per-bin channel estimate,
// used each once, and dropped both. Carrying them on the frame's diagnostics
// is what lets a caller show lock confidence and derive a power delay profile
// without re-running acquisition.

#[test]
fn a_decoded_frame_reports_the_score_it_was_acquired_at() {
    let (cfg, pre, table, _, buf) = framed_at_gain(1.0);
    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
    let frames: Vec<_> = rx.feed(&buf).into_iter().filter_map(|r| r.ok()).collect();
    assert_eq!(frames.len(), 1);
    let score = frames[0]
        .diagnostics
        .sync_score
        .expect("streaming path measures a score");
    // It cleared the acceptance threshold by definition, and the metric is
    // normalized, so it must land in [threshold, 1].
    assert!(
        (0.5..=1.0).contains(&score),
        "score {score} outside the normalized range"
    );
}

#[test]
fn the_channel_estimate_is_opt_in_and_is_the_channel_not_the_raw_bins() {
    let (cfg, pre, table, _, buf) = framed_at_gain(1.0);

    // Off by default — most callers should not pay for the allocation.
    let mut plain = OfdmFrameStreamDemod::new(cfg.clone(), table.clone(), pre);
    let f = plain
        .feed(&buf)
        .into_iter()
        .filter_map(|r| r.ok())
        .next()
        .unwrap();
    assert!(f.diagnostics.channel_estimate.is_none(), "should be opt-in");

    let mut rx = OfdmFrameStreamDemod::new(cfg.clone(), table, pre).with_channel_estimate(true);
    let f = rx
        .feed(&buf)
        .into_iter()
        .filter_map(|r| r.ok())
        .next()
        .unwrap();
    let est = f.diagnostics.channel_estimate.expect("opted in");
    assert_eq!(est.len(), cfg.carrier_plan.n_fft(), "one bin per FFT bin");

    // A clean, noiseless, flat channel must estimate to ~unity. Raw received
    // training bins would not: they carry the pseudo-random known pattern,
    // which is what distinguishes "the channel" from "what arrived".
    let mean: f32 = est.iter().map(|h| h.norm()).sum::<f32>() / est.len() as f32;
    assert!(
        (mean - 1.0).abs() < 0.05,
        "flat channel should estimate near unity, got mean |H| = {mean:.3}"
    );
    assert!(
        est.iter().all(|h| h.norm().is_finite()),
        "no non-finite bins"
    );
}

#[test]
fn the_channel_estimate_tracks_a_gain_change() {
    // A pure gain is a flat channel of that magnitude, so |H| must follow it.
    // This is the property a delay-spread readout is later built on: the
    // estimate has to describe the channel, not be normalized away.
    let mut mags = Vec::new();
    for gain in [1.0_f32, 4.0] {
        let (cfg, pre, table, _, buf) = framed_at_gain(gain);
        let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre).with_channel_estimate(true);
        let f = rx
            .feed(&buf)
            .into_iter()
            .filter_map(|r| r.ok())
            .next()
            .unwrap();
        let est = f.diagnostics.channel_estimate.unwrap();
        mags.push(est.iter().map(|h| h.norm()).sum::<f32>() / est.len() as f32);
    }
    let ratio = mags[1] / mags[0];
    assert!(
        (ratio - 4.0).abs() < 0.1,
        "|H| should scale with the transmitted gain, got {ratio:.3}x"
    );
}

// ── Per-stage FEC outcome and EVM ──────────────────────────────────────────
//
// `decode_chain` folded CRC, inner-FEC and outer-FEC convergence into one
// bool, and the frame path hardcoded `evm_db: None` even though the machinery
// to compute it already existed. Both are what separate "the link is marginal"
// from "the link failed".

#[test]
fn a_clean_frame_reports_both_fec_stages_converged() {
    let (cfg, pre, table, _, buf) = framed_at_gain(1.0);
    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
    let f = rx
        .feed(&buf)
        .into_iter()
        .filter_map(|r| r.ok())
        .next()
        .unwrap();
    assert_eq!(f.diagnostics.inner_fec_ok, Some(true), "inner FEC");
    assert_eq!(f.diagnostics.outer_fec_ok, Some(true), "outer FEC");
}

#[test]
fn the_frame_path_measures_evm() {
    let (cfg, pre, table, _, buf) = framed_at_gain(1.0);
    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
    let f = rx
        .feed(&buf)
        .into_iter()
        .filter_map(|r| r.ok())
        .next()
        .unwrap();
    let evm = f.diagnostics.evm_db.expect("frame path measures EVM");
    // A noiseless frame's constellation sits essentially on its ideal points.
    assert!(
        evm < -25.0,
        "clean frame should have very low EVM, got {evm:.1} dB"
    );
}

#[test]
fn evm_degrades_with_noise_while_the_frame_still_decodes() {
    // EVM has to track channel quality, not just report a constant — this is
    // the property the MER readout is built on. Both frames must still decode,
    // so the comparison is like-for-like.
    let clean = {
        let (cfg, pre, table, _, buf) = framed_at_gain(1.0);
        let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
        rx.feed(&buf)
            .into_iter()
            .filter_map(|r| r.ok())
            .next()
            .unwrap()
            .diagnostics
            .evm_db
            .unwrap()
    };
    let noisy = {
        let (cfg, pre, table, _, mut buf) = framed_at_gain(1.0);
        // Noise *power* relative to the frame's own, matching the convention
        // the neighbouring AWGN tests use.
        let sig_power: f32 = buf.iter().map(|c| c.norm_sqr()).sum::<f32>() / buf.len() as f32;
        add_awgn(&mut buf, sig_power * 0.05, 0xC0FD_2026);
        let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
        rx.feed(&buf)
            .into_iter()
            .filter_map(|r| r.ok())
            .next()
            .expect("20 dB SNR still decodes")
            .diagnostics
            .evm_db
            .unwrap()
    };
    assert!(
        noisy > clean + 5.0,
        "EVM should worsen with noise: clean {clean:.1} dB, noisy {noisy:.1} dB"
    );
}

// ── True bit error rates from a re-encode ──────────────────────────────────
//
// A frame that passed its CRC is ground truth: re-running the encode chain on
// it reconstructs exactly what was transmitted, so the difference from what
// arrived at each stage is a real error rate rather than a pass/fail flag.
// Nothing about the payload need be known in advance — which is what makes
// this work over the air rather than only against a known test vector.

/// A wideband-style link: 256-point plan, QPSK, and a 4x64 preamble long
/// enough for Schmidl & Cox to acquire under noise. `plan_config`'s 4x16
/// preamble cannot — the neighbouring AWGN tests sidestep that by adding noise
/// to the body only and decoding with the batch path, which never syncs.
fn noisy_link_frame(noise_frac: f32) -> (OfdmConfig, OfdmPreamble, McsTable, Vec<C32>) {
    let plan = orion_sdr::multicarrier::CarrierPlan::new(256, 32).with_contiguous_data(111, false);
    let cfg = OfdmConfig::new(plan, 1_920_000.0, 0.0, 1.0, ConstellationOrder::Qpsk)
        .with_payload_crc(CrcKind::Crc32)
        .with_header_crc(CrcKind::Crc16)
        .with_rx_window_backoff(16);
    let pre = OfdmPreamble::new(4, 64).with_training_symbol(256, 32);
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let frame = FramePacket::new(FrameMetadata::new(1, 1), sample_payload(184));
    let frame_iq = modu.modulate_frame(&frame, 0);
    let mut buf = vec![C32::default(); 64];
    buf.extend_from_slice(&frame_iq);
    buf.extend(vec![C32::default(); 512]);
    if noise_frac > 0.0 {
        // Scale noise by the **payload's** power, not the buffer's mean.
        // The buffer includes the preamble and the silence around it, so its
        // mean tracks whatever the preamble happens to be doing: when the
        // preamble was full-band and 30 dB hot, the same `noise_frac` injected
        // ~13 dB more noise than it does now. Referencing the payload keeps
        // the figure meaning one thing across preamble changes.
        let body = &frame_iq[pre.total_len()..];
        let p: f32 = body.iter().map(|c| c.norm_sqr()).sum::<f32>() / body.len() as f32;
        add_awgn(&mut buf, p * noise_frac, 0xBE12_2026);
    }
    (cfg, pre, table, buf)
}

/// One frame at `noise_frac` of its own power, decoded with rates measured.
fn decode_with_rates(noise_frac: f32) -> Option<(f32, f32)> {
    let (cfg, pre, table, buf) = noisy_link_frame(noise_frac);
    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre).with_error_rates(true);
    rx.feed(&buf)
        .into_iter()
        .filter_map(|r| r.ok())
        .next()
        .map(|f| {
            (
                f.diagnostics.channel_ber.expect("opted in"),
                f.diagnostics.inner_ber.expect("opted in"),
            )
        })
}

#[test]
fn error_rates_are_opt_in() {
    let (cfg, pre, table, _, buf) = framed_at_gain(1.0);
    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre);
    let f = rx
        .feed(&buf)
        .into_iter()
        .filter_map(|r| r.ok())
        .next()
        .unwrap();
    assert!(f.diagnostics.channel_ber.is_none());
    assert!(f.diagnostics.inner_ber.is_none());
}

#[test]
fn a_noiseless_frame_measures_zero_channel_errors() {
    let (cber, iber) = decode_with_rates(0.0).expect("clean frame decodes");
    assert_eq!(cber, 0.0, "no channel errors without a channel");
    assert_eq!(iber, 0.0);
}

#[test]
fn the_channel_error_rate_rises_with_noise() {
    // The property that makes this a *rate*: it has to move continuously with
    // channel quality, not step between 0 and 1 the way a flag does.
    let mut prev = -1.0_f32;
    let mut measured = 0;
    for nf in [0.0_f32, 0.4, 0.5] {
        let Some((cber, _)) = decode_with_rates(nf) else {
            continue;
        };
        assert!(
            cber > prev,
            "CBER should rise with noise: {cber:.3e} after {prev:.3e} at noise {nf}"
        );
        prev = cber;
        measured += 1;
    }
    assert!(measured >= 3, "too few decoded points to judge a trend");
}

#[test]
fn a_heavily_errored_channel_still_delivers_a_clean_frame() {
    // The state a folded pass/fail flag cannot express, and the whole reason
    // for measuring here: the channel is hammering the signal while the inner
    // code absorbs it and the frame arrives intact. Reported as a high channel
    // BER against a clean inner BER.
    let (cber, iber) = decode_with_rates(0.5).expect("still decodes at this noise");
    assert!(
        cber > 1.0e-3,
        "expected a visibly errored channel, got {cber:.3e}"
    );
    assert!(
        iber <= cber,
        "the inner code must not make things worse: {iber:.3e} vs {cber:.3e}"
    );
}

// ── Acceptance is decided by the strongest end-to-end check ────────────────
//
// Acceptance used to require `crc_ok && inner_ok && outer_ok`. `inner_ok`
// reports whether the inner decoder's parity checks converged — how hard it
// worked, not whether the answer is right — so requiring it discarded frames
// whose payload was verifiably correct.

#[test]
fn a_frame_the_crc_vouches_for_is_accepted_however_the_fec_stages_fared() {
    // A badly errored channel — 4%+ of demapper decisions wrong — still yields
    // a byte-exact payload that CRC-32 vouches for.
    //
    // The sharper case, where the *inner* stage reports non-convergence and the
    // frame is still good, is asserted at the chain level in
    // `acceptance_falls_back_when_there_is_no_crc_to_ask` and was measured
    // directly on `decode_chain`. It is not reproducible through the full OFDM
    // path on this fixture: the LDPC converges even at 13.9% channel BER, and
    // past that the frame stops decoding at all, so there is no band between
    // the two to test end to end. Asserting it here would only be pinning a
    // coincidence.
    let cfg = plan_config();
    let pre = OfdmPreamble::new(4, 16); // no training symbol ⇒ unequalized
    let table = McsTable::default_ladder();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let payload = sample_payload(30);
    let mut buf = vec![C32::default(); 24];
    buf.extend_from_slice(&modu.modulate_frame(
        &FramePacket::new(FrameMetadata::new(3, 1), payload.clone()),
        0,
    ));
    buf.extend(vec![C32::default(); 64]);
    let channeled = apply_fir_channel(&buf, &harsh_multipath_taps());

    let mut rx = OfdmFrameStreamDemod::new(cfg, table, pre).with_error_rates(true);
    let f = rx
        .feed(&channeled)
        .into_iter()
        .filter_map(|r| r.ok())
        .next()
        .expect("a CRC-verified frame must be accepted");
    assert_eq!(f.packet.payload, payload, "and its payload must be exact");
    assert!(
        f.diagnostics.channel_ber.unwrap() > 0.02,
        "a badly errored channel"
    );
}

#[test]
fn acceptance_falls_back_when_there_is_no_crc_to_ask() {
    // `CrcKind::None` reports `crc_ok = true` because nothing was checked, not
    // because anything passed. A rule of "the CRC decides" must therefore know
    // whether a CRC exists at all, or a link with neither CRC nor outer code
    // would accept anything the demapper produced.
    let cache = CodecCache::new();
    let mk = |crc: CrcKind, outer: OuterFec, inner_ok: bool, outer_ok: bool, crc_ok: bool| {
        let _ = &cache;
        ChainOutcome {
            bytes: Vec::new(),
            inner_ok,
            outer_ok,
            crc_ok,
            crc_present: crc != CrcKind::None,
            outer_present: outer != OuterFec::None,
            inner_out_bits: Vec::new(),
        }
    };
    // With a CRC, the CRC alone decides.
    assert!(mk(CrcKind::Crc32, OuterFec::Bch { t: 8 }, false, false, true).is_valid());
    assert!(!mk(CrcKind::Crc32, OuterFec::Bch { t: 8 }, true, true, false).is_valid());
    // Without one — DVB-T's shape — the outer code is the integrity check.
    assert!(
        mk(
            CrcKind::None,
            OuterFec::ReedSolomon {
                n: 204,
                n_parity: 16
            },
            false,
            true,
            true
        )
        .is_valid()
    );
    assert!(
        !mk(
            CrcKind::None,
            OuterFec::ReedSolomon {
                n: 204,
                n_parity: 16
            },
            true,
            false,
            true
        )
        .is_valid()
    );
    // With neither, the inner decoder's convergence is all there is.
    assert!(mk(CrcKind::None, OuterFec::None, true, true, true).is_valid());
    assert!(
        !mk(CrcKind::None, OuterFec::None, false, true, true).is_valid(),
        "a bare inner-only link must not accept a non-converged block"
    );
}

// ── Frame assembly is baseband-only ────────────────────────────────────────
//
// `rf_hz` is honoured by the per-symbol `OfdmMod`, but the frame layer cannot:
// the spectral mask is centred on DC, the preamble generator does not apply it,
// and a fresh rotator per block steps the phase at every seam. The receiver
// never applies it either. It used to fail silently in all of those ways.

fn config_at_rf(rf_hz: f32) -> OfdmConfig {
    OfdmConfig::new(
        plan_config().carrier_plan.clone(),
        48_000.0,
        rf_hz,
        1.0,
        ConstellationOrder::Bpsk,
    )
}

#[test]
fn a_baseband_config_is_accepted_everywhere() {
    let cfg = config_at_rf(0.0);
    let table = McsTable::default_ladder();
    let pre = preamble(&cfg);
    let _ = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);
    let _ = OfdmFrameDemod::new(cfg.clone(), table.clone());
    let _ = OfdmFrameStreamDemod::new(cfg, table, pre);
}

#[test]
#[should_panic(expected = "baseband-only")]
fn the_modulator_rejects_an_if_config() {
    let cfg = config_at_rf(12_000.0);
    let pre = preamble(&cfg);
    let _ = OfdmFrameMod::new(cfg, McsTable::default_ladder(), pre);
}

#[test]
#[should_panic(expected = "baseband-only")]
fn the_batch_demodulator_rejects_an_if_config() {
    let _ = OfdmFrameDemod::new(config_at_rf(12_000.0), McsTable::default_ladder());
}

#[test]
#[should_panic(expected = "baseband-only")]
fn the_streaming_demodulator_rejects_an_if_config() {
    let cfg = config_at_rf(12_000.0);
    let pre = preamble(&cfg);
    let _ = OfdmFrameStreamDemod::new(cfg, McsTable::default_ladder(), pre);
}

#[test]
fn the_per_symbol_modulator_still_upconverts() {
    // The guard must not reach `OfdmMod`, which honours `rf_hz` correctly —
    // one rotator, one continuous stream, no shaping stage to disturb.
    use orion_sdr::modulate::OfdmMod;
    let cfg = config_at_rf(12_000.0);
    let bits: Vec<u8> = (0..256).map(|i| (i % 2) as u8).collect();
    let baseband = OfdmMod::new(&config_at_rf(0.0)).modulate(&bits);
    let upconverted = OfdmMod::new(&cfg).modulate(&bits);
    assert_eq!(baseband.len(), upconverted.len());
    // Same energy, different placement: the rotation is a pure frequency shift.
    let energy = |s: &[C32]| s.iter().map(|c| c.norm_sqr()).sum::<f32>();
    assert!(
        (energy(&baseband) / energy(&upconverted) - 1.0).abs() < 1e-3,
        "upconversion must preserve energy"
    );
    assert!(
        baseband
            .iter()
            .zip(upconverted.iter())
            .any(|(a, b)| (a - b).norm() > 1e-3),
        "and must actually move the signal"
    );
}
