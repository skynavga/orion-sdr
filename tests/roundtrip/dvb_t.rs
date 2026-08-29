// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// End-to-end DVB-T (narrowband) roundtrip tests. The DVB-T-conformant pieces
// here are the **payload FEC chain** — K=7 convolutional inner + Forney
// (I=12, M=17) outer interleaver + RS(204,188) outer + exact DVB-T energy
// dispersal — and the **2K-mode carrier map** with continual pilots, at the
// amateur fs-scaled bandwidths (333 kHz / 1 MHz / 2 MHz). See
// `waveform::dvb_t::dvb_t_config`.
//
// The tests progress from Phase-1 scaffolding to the Phase-3 conformant frame:
//   • the `roundtrip_dvb_t_2k_*` tests exercise the payload FEC + carrier map via
//     the generic COFDM frame layer (S&C preamble + `OrionSdr` header) — useful
//     coverage of the FEC chain, but not the conformant on-air frame;
//   • the `dvb_t_scattered_*` tests add the four-phase scattered-pilot rotation;
//   • the capstone `roundtrip_dvb_t_2k_tps_end_to_end` / `dvb_t_tps_frame_*` tests
//     use the fully conformant preamble-less frame (guard-interval acquisition,
//     TPS signalling, MPEG-TS payload, DVB-T soft-decision) via
//     `{modulate,demodulate}::dvb_t_frame`.

use crate::common::add_awgn;
use num_complex::Complex32 as C32;
use orion_sdr::core::Block;
use orion_sdr::demodulate::OfdmFrameDemod;
use orion_sdr::fec::{FrameMetadata, FramePacket};
use orion_sdr::modulate::{OfdmConfig, OfdmFrameMod};
use orion_sdr::multicarrier::{CarrierGrid, CyclicPrefixRemove, FftBlock, GridExtract, GridMap};
use orion_sdr::sync::OfdmPreamble;
use orion_sdr::waveform::dvb_t::{
    GuardInterval, dvb_t_2k_plan, dvb_t_config, dvb_t_demap_symbol, dvb_t_map_symbol,
    dvb_t_mcs_table, dvb_t_scattered_config,
};

// A preamble sized to the 2K plan, with a training symbol for channel/CFO.
fn preamble(cfg: &OfdmConfig) -> OfdmPreamble {
    OfdmPreamble::new(4, 64)
        .with_training_symbol(cfg.carrier_plan.n_fft(), cfg.carrier_plan.cp_len())
}

fn strip_preamble(pre: &OfdmPreamble, iq: &[C32]) -> Vec<C32> {
    iq[pre.total_len()..].to_vec()
}

fn sample_payload(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 37 + 11) & 0xff) as u8).collect()
}

/// Full DVB-T frame roundtrip at a given occupied bandwidth (fs-scaled). One
/// RS(204,188) codeword of payload (188 − 4 CRC bytes) through the conformant
/// FEC chain + 2K carrier map + DVB-T energy dispersal.
fn dvb_t_frame_roundtrip(occupied_hz: f32, mcs_index: u8) {
    let cfg = dvb_t_config(GuardInterval::G1_32, occupied_hz);
    let pre = preamble(&cfg);
    let table = dvb_t_mcs_table();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(184);
    let frame = FramePacket::new(FrameMetadata::new(1, mcs_index), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let body = strip_preamble(modu.preamble(), &iq);
    let got = OfdmFrameDemod::new(cfg.clone(), table.clone())
        .decode(&body)
        .expect("DVB-T frame decode");
    assert_eq!(got.payload, payload);
}

#[test]
fn roundtrip_dvb_t_2k_noiseless_333khz() {
    // MCS 0: QPSK rate 1/2 (the robust 333 kHz-class config).
    dvb_t_frame_roundtrip(333_000.0, 0);
}

#[test]
fn roundtrip_dvb_t_2k_noiseless_1mhz() {
    // MCS 1: QPSK rate 2/3 (general-purpose 1 MHz-class). fs is the only change.
    dvb_t_frame_roundtrip(1_000_000.0, 1);
}

#[test]
fn roundtrip_dvb_t_2k_noiseless_2mhz() {
    // MCS 2: 16-QAM rate 3/4 (wider 2 MHz-class).
    dvb_t_frame_roundtrip(2_000_000.0, 2);
}

#[test]
fn roundtrip_dvb_t_2k_awgn() {
    // QPSK rate 1/2 under modest AWGN — the concatenated FEC's coding gain.
    let cfg = dvb_t_config(GuardInterval::G1_32, 1_000_000.0);
    let pre = preamble(&cfg);
    let table = dvb_t_mcs_table();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(184);
    let frame = FramePacket::new(FrameMetadata::new(2, 0), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let mut body = strip_preamble(modu.preamble(), &iq);
    let sig_power: f32 = body.iter().map(|s| s.norm_sqr()).sum::<f32>() / body.len() as f32;
    add_awgn(&mut body, sig_power * 0.06, 0xD7B_C0DE);
    let got = OfdmFrameDemod::new(cfg.clone(), table.clone())
        .decode(&body)
        .expect("DVB-T AWGN decode");
    assert_eq!(got.payload, payload);
}

// ── DVB-T-exact constellation through an OFDM channel (hard-decision) ───────

/// Maps `n_syms` DVB-T symbols (each `v` bits from `bits`) with the DVB-T-exact
/// constellation, runs them through the OFDM grid/IFFT → (optional AWGN) →
/// FFT/extract chain, hard-demaps with the DVB-T-exact demapper, and returns the
/// recovered bits. Proves the DVB-T mapping decodes end to end.
#[test]
fn dvb_t_qam_through_ofdm_channel() {
    for &(v, noise) in &[(2usize, 0.0f32), (4, 0.0), (2, 0.02)] {
        // A small plan is enough to exercise the mapping through OFDM.
        let plan = dvb_t_2k_plan(GuardInterval::G1_32);
        let n_data = plan.data_carriers().len();
        let grid = CarrierGrid::from_plan(&plan);
        let n_fft = plan.n_fft();
        let cp_len = plan.cp_len();

        // One OFDM symbol's worth of DVB-T-mapped data carriers.
        let mut r = 0x1234u32;
        let mut next = || {
            r = r.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (r >> 24) as u8 & 1
        };
        let bits: Vec<u8> = (0..n_data * v).map(|_| next()).collect();
        let syms: Vec<C32> = bits
            .chunks(v)
            .map(|c| dvb_t_map_symbol(c).unwrap())
            .collect();

        // Map onto the grid, IFFT, add CP.
        let mut freq = vec![C32::default(); n_fft];
        let mut gmap = GridMap::new(grid.clone());
        gmap.process(&syms, &mut freq);
        let mut time = vec![C32::default(); n_fft];
        orion_sdr::multicarrier::IfftBlock::new(n_fft).process(&freq, &mut time);
        let mut with_cp = vec![C32::default(); n_fft + cp_len];
        orion_sdr::multicarrier::CyclicPrefixInsert::new(n_fft, cp_len)
            .process(&time, &mut with_cp);

        if noise > 0.0 {
            let p: f32 = with_cp.iter().map(|s| s.norm_sqr()).sum::<f32>() / with_cp.len() as f32;
            add_awgn(&mut with_cp, p * noise, 0xC0DE_1234);
        }

        // RX: strip CP, FFT, extract data carriers, hard-demap.
        let mut rx_time = vec![C32::default(); n_fft];
        CyclicPrefixRemove::new(n_fft, cp_len).process(&with_cp, &mut rx_time);
        let mut rx_freq = vec![C32::default(); n_fft];
        FftBlock::new(n_fft).process(&rx_time, &mut rx_freq);
        let mut rx_syms = vec![C32::default(); n_data];
        GridExtract::new(grid).process(&rx_freq, &mut rx_syms);

        let recovered: Vec<u8> = rx_syms
            .iter()
            .flat_map(|&s| dvb_t_demap_symbol(s, v).unwrap())
            .collect();

        if noise == 0.0 {
            assert_eq!(recovered, bits, "v={v}: noiseless DVB-T QAM roundtrip");
        } else {
            let errs = recovered.iter().zip(&bits).filter(|(a, b)| a != b).count();
            let ber = errs as f32 / bits.len() as f32;
            assert!(
                ber < 0.01,
                "v={v} noise={noise}: DVB-T QAM BER {ber} too high"
            );
        }
    }
}

// ── Scattered pilots (Phase 2) ─────────────────────────────────────────────
//
// The conformant 2K frame structure: continual + phase-rotating scattered + TPS
// pilots reserved (exactly 1512 data carriers per symbol), with per-symbol
// channel estimation off the scattered+continual pilots. The frame layer rotates
// the four symbol-phase grids underneath a representative phase-0 plan
// (`dvb_t_scattered_config`); see `waveform::dvb_t`.

/// Convolves `iq` with a short FIR channel (delay spread ≤ cp_len), mirroring
/// the multipath helper in the COFDM frame roundtrip tests.
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
fn roundtrip_dvb_t_2k_scattered_noiseless() {
    // Bit-exact frame roundtrip through the four-phase scattered-pilot rotation
    // (flat channel). Proves TX/RX agree on the per-symbol grid phase across the
    // whole header+payload symbol stream.
    let cfg = dvb_t_scattered_config(GuardInterval::G1_32, 1_000_000.0);
    let pre = preamble(&cfg);
    let table = dvb_t_mcs_table();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    // Two RS codewords, so the payload spans several rotation cycles.
    let payload = sample_payload(368);
    let frame = FramePacket::new(FrameMetadata::new(7, 0), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let body = strip_preamble(modu.preamble(), &iq);
    let got = OfdmFrameDemod::new(cfg.clone(), table.clone())
        .decode(&body)
        .expect("scattered frame decode");
    assert_eq!(got.payload, payload);
}

/// A 2-tap frequency-selective channel strong enough that an uncorrected decode
/// fails outright — the second tap is comparable to the first, giving a deep
/// frequency null and large per-carrier phase rotation. The dense per-symbol
/// scattered-pilot estimate tracks it and recovers the frame; the flat path
/// (no channel correction) cannot. Delay spread (1 sample) ≤ cp_len (64).
fn scattered_multipath_taps() -> [C32; 2] {
    [C32::new(0.8, 0.0), C32::new(0.7, -0.35)]
}

#[test]
fn dvb_t_scattered_multipath_decodes() {
    // WITH the scattered-pilot config the dense per-symbol channel estimate
    // tracks the frequency-selective channel and the frame decodes.
    let cfg = dvb_t_scattered_config(GuardInterval::G1_32, 1_000_000.0);
    let pre = preamble(&cfg);
    let table = dvb_t_mcs_table();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(184);
    let frame = FramePacket::new(FrameMetadata::new(3, 0), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let taps = scattered_multipath_taps();
    assert!(taps.len() - 1 <= cfg.carrier_plan.cp_len());
    let channeled = apply_fir_channel(&iq, &taps);
    let body = strip_preamble(modu.preamble(), &channeled);
    let got = OfdmFrameDemod::new(cfg.clone(), table.clone())
        .decode(&body)
        .expect("scattered multipath frame decode");
    assert_eq!(got.payload, payload);
}

#[test]
fn dvb_t_scattered_needed_for_multipath() {
    // The load-bearing counter-test: the SAME channel through the Phase-1
    // continual-pilots-only config (no scattered rotation, no per-symbol
    // estimate here — flat-channel demap) must FAIL. This proves the scattered
    // pilots are what carry the multipath decode, not some incidental margin.
    let cfg = dvb_t_config(GuardInterval::G1_32, 1_000_000.0);
    let pre = preamble(&cfg);
    let table = dvb_t_mcs_table();
    let modu = OfdmFrameMod::new(cfg.clone(), table.clone(), pre);

    let payload = sample_payload(184);
    let frame = FramePacket::new(FrameMetadata::new(3, 0), payload.clone());
    let iq = modu.modulate_frame(&frame, 0);
    let channeled = apply_fir_channel(&iq, &scattered_multipath_taps());
    let body = strip_preamble(modu.preamble(), &channeled);
    // The flat-channel batch demap has no channel correction, so the strong
    // 2-tap channel corrupts the payload beyond the FEC's reach.
    let got = OfdmFrameDemod::new(cfg.clone(), table.clone()).decode(&body);
    assert!(
        got.is_err() || got.as_ref().unwrap().payload != payload,
        "continual-pilots-only decode must not recover the multipath frame"
    );
}

// ── Conformant preamble-less DVB-T frame (Phase 3 capstone) ─────────────────
//
// The full DVB-T on-air frame: MPEG-TS payload + energy dispersal + payload FEC,
// Figure-9a soft-decision through the four-phase scattered-pilot grid, TPS on the
// 17 carriers, and guard-interval acquisition — NO preamble, NO OrionSdr header.
// TX: `modulate::DvbTFrameMod`; RX: `demodulate::DvbTFrameDemod`.

use orion_sdr::demodulate::DvbTFrameDemod;
use orion_sdr::fec::PunctureRate;
use orion_sdr::modulate::{ConstellationOrder, DvbTFrameMod};
use orion_sdr::multicarrier::SymbolFft;
use orion_sdr::waveform::dvb_t::{
    DVB_T_MAX_RX_WINDOW_BACKOFF, DVB_T_SCATTERED_PILOT_SPACING, DvbTFrameParams, DvbTLinkParams,
};

fn capstone_params() -> DvbTFrameParams {
    DvbTFrameParams {
        link: DvbTLinkParams {
            guard: GuardInterval::G1_32,
            constellation: ConstellationOrder::Qpsk,
            code_rate: PunctureRate::R1_2,
        },
        frame_number: 2,
        cell_id: 0x5A,
    }
}

#[test]
fn roundtrip_dvb_t_2k_tps_end_to_end() {
    // The capstone: modulate a TS payload into a conformant preamble-less frame,
    // place it at an unknown offset with trailing silence, GI-acquire it, and
    // recover BOTH the payload AND the TPS-signalled parameters.
    let params = capstone_params();
    let payload = sample_payload(184); // one RS(204,188) information block worth
    let frame = DvbTFrameMod::new(params).modulate(&payload);

    let lead = 200usize;
    let mut buf = vec![C32::default(); lead];
    buf.extend_from_slice(&frame.iq);
    buf.extend(vec![C32::default(); frame.samples_per_symbol]);

    let got = DvbTFrameDemod::new(params)
        .decode(&buf, frame.n_symbols, payload.len())
        .expect("conformant DVB-T frame decode");
    assert_eq!(got.payload, payload, "recovered TS payload");
    assert_eq!(got.tps.frame_number, params.frame_number);
    assert_eq!(got.tps.constellation, params.constellation());
    assert_eq!(got.tps.code_rate_hp, params.code_rate());
    assert_eq!(got.tps.guard, params.guard());
    assert_eq!(got.tps.cell_id, params.cell_id);
}

#[test]
fn dvb_t_symbol_windowing_attenuates_symbol_edges() {
    // R12 (DVB-T, cheap): confirm the taper is actually applied to a real DVB-T
    // frame — its symbol-edge (guard) samples are attenuated versus the
    // unwindowed frame, while an interior sample is untouched. (The spectral skirt
    // reduction is proven once on the shared `SymbolWindow` primitive in the COFDM
    // unit test; here we only verify DVB-T output is genuinely windowed, without a
    // slow 2K×68-symbol FFT.)
    let params = capstone_params(); // G1_32 -> cp_len = 64
    let cp_len = GuardInterval::G1_32.cp_len_2k();
    let roll_off = cp_len / 2;
    let payload = sample_payload(184);

    let plain = DvbTFrameMod::new(params).modulate(&payload);
    let windowed = DvbTFrameMod::new(params)
        .with_symbol_window(roll_off)
        .modulate(&payload);
    let sps = plain.samples_per_symbol;

    // Take an interior symbol (skip the first). Its very first/last samples are in
    // the ramp regions and must be attenuated; a mid-symbol sample is untouched.
    let base = 2 * sps;
    assert!(
        windowed.iq[base].norm() < plain.iq[base].norm() * 0.5,
        "leading edge sample should be strongly attenuated"
    );
    assert!(
        windowed.iq[base + sps - 1].norm() < plain.iq[base + sps - 1].norm() * 0.5,
        "trailing edge sample should be strongly attenuated"
    );
    // A sample in the flat interior is unchanged.
    let mid = base + sps / 2;
    assert!(
        (windowed.iq[mid] - plain.iq[mid]).norm() < 1e-6,
        "interior sample must be untouched by the taper"
    );
}

#[test]
fn roundtrip_dvb_t_frame_with_symbol_windowing() {
    // R11: a conformant DVB-T frame with TX symbol windowing (roll_off = cp_len/2)
    // decodes cleanly when the demod's RX window back-off is matched (cp_len/2).
    // The taper lives only in guard samples the backed-off window discards, so the
    // continual/scattered/TPS pilots and the payload are all recovered intact.
    let params = capstone_params(); // G1_32 -> cp_len = 2048/32 = 64
    let cp_len = GuardInterval::G1_32.cp_len_2k();
    let roll_off = cp_len / 2;

    let payload = sample_payload(184);
    let frame = DvbTFrameMod::new(params)
        .with_symbol_window(roll_off)
        .modulate(&payload);

    let lead = 200usize;
    let mut buf = vec![C32::default(); lead];
    buf.extend_from_slice(&frame.iq);
    buf.extend(vec![C32::default(); frame.samples_per_symbol]);

    let got = DvbTFrameDemod::new(params)
        .with_rx_window_backoff(cp_len / 2)
        .decode(&buf, frame.n_symbols, payload.len())
        .expect("windowed DVB-T frame decodes with matched back-off");
    assert_eq!(got.payload, payload, "recovered TS payload");
    // Pilots/TPS intact — windowing touched only guard samples, not carriers.
    assert_eq!(got.tps.frame_number, params.frame_number);
    assert_eq!(got.tps.constellation, params.constellation());
    assert_eq!(got.tps.guard, params.guard());
    assert_eq!(got.tps.cell_id, params.cell_id);
}

#[test]
fn roundtrip_dvb_t_frame_with_tx_lowpass() {
    // R16: a conformant DVB-T frame carrying the TX spectral mask decodes with a
    // matched RX window back-off — payload, TPS, and pilots all intact. The mask
    // is a linear channel the scattered-pilot equalizer absorbs; the only thing
    // the receiver must provide is guard for the filter's group delay to sit in.
    let params = capstone_params(); // G1_32 -> cp_len = 64
    let cp_len = GuardInterval::G1_32.cp_len_2k();
    // 45 taps at 60 dB: group delay 22 <= min(cp_len - b, b) = 32 at b = 32.
    let lowpass = DvbTFrameMod::tx_lowpass_for_2k(45, 60.0);
    assert_eq!(lowpass.group_delay(), 22);
    assert!(lowpass.fits_guard(cp_len, 0, cp_len / 2));
    assert!(
        lowpass.transition_fits(2048, 852),
        "45 taps must reach the stop band inside DVB-T's 1705-of-2048 null band"
    );

    let payload = sample_payload(184);
    let frame = DvbTFrameMod::new(params)
        .with_tx_lowpass(lowpass)
        .modulate(&payload);

    let lead = 200usize;
    let mut buf = vec![C32::default(); lead];
    buf.extend_from_slice(&frame.iq);
    buf.extend(vec![C32::default(); frame.samples_per_symbol]);

    let got = DvbTFrameDemod::new(params)
        .with_rx_window_backoff(cp_len / 2)
        .decode(&buf, frame.n_symbols, payload.len())
        .expect("band-limited DVB-T frame decodes with matched back-off");
    assert_eq!(got.payload, payload, "recovered TS payload");
    assert_eq!(got.tps.frame_number, params.frame_number);
    assert_eq!(got.tps.constellation, params.constellation());
    assert_eq!(got.tps.guard, params.guard());
    assert_eq!(got.tps.cell_id, params.cell_id);
}

fn g1_8_params() -> DvbTFrameParams {
    DvbTFrameParams {
        link: DvbTLinkParams {
            guard: GuardInterval::G1_8,
            constellation: ConstellationOrder::Qpsk,
            code_rate: PunctureRate::R1_2,
        },
        frame_number: 1,
        cell_id: 0x33,
    }
}

#[test]
fn dvb_t_rx_window_backoff_is_capped_by_the_pilot_grid_not_the_guard() {
    // A back-off `b` puts a phase ramp exp(-j2πkb/n_fft) on the spectrum, and the
    // per-symbol scattered-pilot equalizer only samples the channel every 12
    // carriers — so once the ramp advances more than ~pi between pilots, the
    // interpolation aliases and the decode dies, no matter how much guard is
    // free. G1/8 has cp_len = 256, yet `cp_len/2 = 128` is already past the
    // 85-sample ceiling. This is why the "b = cp_len/2" rule of thumb has an
    // upper bound, and why guards beyond ~170 samples buy no extra TX-shaping
    // budget (see DVB_T_MAX_RX_WINDOW_BACKOFF).
    let params = g1_8_params();
    let cp_len = GuardInterval::G1_8.cp_len_2k(); // 256
    assert_eq!(DVB_T_MAX_RX_WINDOW_BACKOFF, 85);
    assert_eq!(
        SymbolFft::max_pilot_safe_backoff(2048, DVB_T_SCATTERED_PILOT_SPACING),
        DVB_T_MAX_RX_WINDOW_BACKOFF
    );

    let payload = sample_payload(184);
    let frame = DvbTFrameMod::new(params).modulate(&payload);
    let decodes_at = |b: usize| {
        let mut buf = vec![C32::default(); 200];
        buf.extend_from_slice(&frame.iq);
        buf.extend(vec![C32::default(); frame.samples_per_symbol]);
        DvbTFrameDemod::new(params)
            .with_rx_window_backoff(b)
            .decode(&buf, frame.n_symbols, payload.len())
            .map(|f| f.payload == payload)
            .unwrap_or(false)
    };
    assert!(
        decodes_at(64),
        "a back-off inside the pilot ceiling decodes"
    );
    assert!(
        !decodes_at(cp_len / 2),
        "cp_len/2 = {} is past the {DVB_T_MAX_RX_WINDOW_BACKOFF}-sample pilot \
         ceiling and must NOT decode — the guard has room but the pilot grid does not",
        cp_len / 2
    );
}

#[test]
fn roundtrip_dvb_t_frame_with_tx_lowpass_and_symbol_windowing() {
    // R16/R17: both TX shaping levers on one DVB-T frame, sharing the one budget
    // `roll_off + group_delay <= min(cp_len - b, b)` — with `b` itself capped by
    // the pilot grid (above). G1/8 at b = 64 gives 64 samples of slack, double
    // what G1/32 can offer: the real (if bounded) long-guard win.
    let params = g1_8_params();
    let cp_len = GuardInterval::G1_8.cp_len_2k(); // 256
    let backoff = 64usize;
    assert!(backoff <= DVB_T_MAX_RX_WINDOW_BACKOFF);
    let roll_off = 16usize;
    let lowpass = DvbTFrameMod::tx_lowpass_for_2k(89, 80.0); // group delay 44
    assert!(
        lowpass.fits_guard(cp_len, roll_off, backoff),
        "44 + 16 must fit the 64-sample slack at b = 64"
    );

    let payload = sample_payload(184);
    let frame = DvbTFrameMod::new(params)
        .with_symbol_window(roll_off)
        .with_tx_lowpass(lowpass)
        .modulate(&payload);

    let mut buf = vec![C32::default(); 200];
    buf.extend_from_slice(&frame.iq);
    buf.extend(vec![C32::default(); frame.samples_per_symbol]);

    let got = DvbTFrameDemod::new(params)
        .with_rx_window_backoff(backoff)
        .decode(&buf, frame.n_symbols, payload.len())
        .expect("masked + windowed DVB-T frame decodes");
    assert_eq!(got.payload, payload);
    assert_eq!(got.tps.guard, params.guard());
    assert_eq!(got.tps.cell_id, params.cell_id);
}

/// Mean power (dB, linear-averaged) over the `take`-point FFT bins whose
/// carrier-equivalent index (in `n_fft` units) falls in `[lo_k, hi_k]`, measured
/// through a 4-term Blackman–Harris window.
///
/// The window is not optional: a raw rectangular slice leaks its own `~1/f`
/// skirt about 35 dB below the in-band power, which would swamp the very
/// attenuation being measured.
fn mean_band_power_db(samples: &[C32], take: usize, n_fft: usize, lo_k: i32, hi_k: i32) -> f32 {
    const A: [f32; 4] = [0.35875, 0.48829, 0.14128, 0.01168];
    let mut buf: Vec<rustfft::num_complex::Complex<f32>> = (0..take)
        .map(|i| {
            let s = samples.get(i).copied().unwrap_or_default();
            let x = core::f32::consts::TAU * i as f32 / take as f32;
            let w = A[0] - A[1] * x.cos() + A[2] * (2.0 * x).cos() - A[3] * (3.0 * x).cos();
            rustfft::num_complex::Complex::new(s.re * w, s.im * w)
        })
        .collect();
    rustfft::FftPlanner::new()
        .plan_fft_forward(take)
        .process(&mut buf);

    let t = take as i32;
    let (mut acc, mut count) = (0.0f64, 0usize);
    for (bin, c) in buf.iter().enumerate() {
        let signed = if (bin as i32) <= t / 2 {
            bin as i32
        } else {
            bin as i32 - t
        };
        let k = (signed.abs() * n_fft as i32) / t;
        if k >= lo_k && k <= hi_k {
            acc += (c.re * c.re + c.im * c.im) as f64;
            count += 1;
        }
    }
    (10.0 * (acc / count.max(1) as f64).log10()) as f32
}

#[test]
fn dvb_t_tx_lowpass_attenuates_the_null_band() {
    // R16 (spectral): DVB-T's out-of-band skirt lives in the 343 bins between the
    // outermost active carrier (±852) and Nyquist (±1024). Measure there, past
    // the mask's own transition, on a real conformant frame.
    let params = g1_8_params();
    let lowpass = DvbTFrameMod::tx_lowpass_for_2k(89, 60.0);
    let payload = sample_payload(184);

    let plain = DvbTFrameMod::new(params).modulate(&payload);
    let masked = DvbTFrameMod::new(params)
        .with_tx_lowpass(lowpass)
        .modulate(&payload);

    // A window from the frame interior, clear of the filter's edge transient.
    let take = 8192usize;
    let off = 4 * plain.samples_per_symbol;
    let n_fft = 2048usize;
    let stop_k = (lowpass.stopband_edge_norm() * n_fft as f32).ceil() as i32;
    assert!(
        stop_k < n_fft as i32 / 2,
        "89 taps must reach the stop band before Nyquist (got k = {stop_k})"
    );

    let band = |s: &[C32]| mean_band_power_db(&s[off..off + take], take, n_fft, stop_k, 1024);
    let in_band = |s: &[C32]| mean_band_power_db(&s[off..off + take], take, n_fft, 0, 800);

    let oob_plain = band(&plain.iq);
    let oob_masked = band(&masked.iq);
    let ib_plain = in_band(&plain.iq);
    let ib_masked = in_band(&masked.iq);
    println!(
        "DVB-T k[{stop_k}, 1024]: plain={oob_plain:.1} masked={oob_masked:.1} dB \
         (in band: {ib_plain:.1} -> {ib_masked:.1} dB)"
    );

    assert!(
        oob_masked < oob_plain - 20.0,
        "the mask should cut DVB-T's null-band emission hard \
         (plain={oob_plain:.1} dB, masked={oob_masked:.1} dB)"
    );
    // ...without touching the carriers that matter: the active band is unchanged.
    assert!(
        (ib_masked - ib_plain).abs() < 0.5,
        "in-band power must be preserved ({ib_plain:.1} -> {ib_masked:.1} dB)"
    );
}

#[test]
fn dvb_t_tx_lowpass_defaults_off() {
    // Opt-in discipline: without the builder the frame is byte-identical to the
    // pre-Track-C output.
    let params = capstone_params();
    let payload = sample_payload(184);
    let plain = DvbTFrameMod::new(params).modulate(&payload);
    let masked = DvbTFrameMod::new(params)
        .with_tx_lowpass(DvbTFrameMod::tx_lowpass_for_2k(45, 60.0))
        .modulate(&payload);
    assert_eq!(plain.iq.len(), masked.iq.len(), "same-length post-pass");
    assert_ne!(plain.iq, masked.iq, "the mask must actually change the IQ");
    let again = DvbTFrameMod::new(params).modulate(&payload);
    assert_eq!(plain.iq, again.iq, "default path unchanged");
}

#[test]
fn dvb_t_tps_frame_survives_awgn() {
    // The same conformant frame through modest AWGN: the DBPSK TPS (17-carrier
    // averaged, BCH-protected) and the soft-decision payload FEC both hold.
    let params = capstone_params();
    let payload = sample_payload(184);
    let frame = DvbTFrameMod::new(params).modulate(&payload);

    let lead = 128usize;
    let mut buf = vec![C32::default(); lead];
    buf.extend_from_slice(&frame.iq);
    buf.extend(vec![C32::default(); frame.samples_per_symbol]);
    let sig_power: f32 = frame.iq.iter().map(|s| s.norm_sqr()).sum::<f32>() / frame.iq.len() as f32;
    add_awgn(&mut buf, sig_power * 0.03, 0x0DB7_0777);

    let got = DvbTFrameDemod::new(params)
        .decode(&buf, frame.n_symbols, payload.len())
        .expect("DVB-T frame AWGN decode");
    assert_eq!(got.payload, payload);
    assert_eq!(got.tps.constellation, params.constellation());
}

// ── Equalizer channel-reference regression (the TPS-pilot bug) ──────────────
//
// Guards a subtle correctness property: the RX scattered-pilot equalizer must NOT
// use the 17 TPS carriers as channel references. The modulator transmits data-
// power DBPSK (±1.0) on the TPS bins, but the grid records them as boosted `w_k`
// pilots (±4/3); feeding them to the estimator would yield `h = ±1.0/±4/3 = ∓0.75`
// (wrong magnitude AND sign), which the equalizer's bracket interpolation then
// smears onto the ~3 data carriers straddling each TPS carrier — a
// deterministic, SNR-invariant pre-FEC error floor. A payload-only decode
// assertion at zero noise can pass on a thin
// single-codeword RS margin without exercising this, so the test checks the
// equalizer directly: noiselessly, every EQUALIZED data carrier must equal the
// transmitted one (pre-FEC coded-bit error exactly zero) — for both a robust and a
// dense config, at every guard interval.
fn assert_noiseless_equalizer_is_clean(
    constellation: ConstellationOrder,
    code_rate: PunctureRate,
    guard: GuardInterval,
) {
    use orion_sdr::demodulate::ofdm::{EqualizerMethod, OfdmEqualizer};
    use orion_sdr::waveform::dvb_t::{DVB_T_DATA_CARRIERS, DVB_T_N_FFT, ScatteredPilotExtractor};

    let params = DvbTFrameParams {
        link: DvbTLinkParams {
            guard,
            constellation,
            code_rate,
        },
        frame_number: 0,
        cell_id: 0,
    };
    let payload = sample_payload(184);
    let frame = DvbTFrameMod::new(params).modulate(&payload);
    let n_fft = DVB_T_N_FFT;
    let cp_len = guard.cp_len_2k();
    let sps = n_fft + cp_len;

    // Reconstruct each symbol's transmitted frequency-domain data carriers by
    // re-running the mapper isn't necessary: the transmitted data-carrier values
    // are recoverable by FFT-ing each TX symbol (noiseless) and gathering the data
    // bins. Compare those against the RX equalizer's output over a NOISELESS
    // channel — they must match to floating-point precision.
    let mut tx_cpr = CyclicPrefixRemove::new(n_fft, cp_len);
    let mut tx_fft = FftBlock::new(n_fft);
    let mut tx_time = vec![C32::default(); n_fft];
    let mut tx_freq = vec![C32::default(); n_fft];

    let mut ext = ScatteredPilotExtractor::new(guard);
    let mut tx_ext = ScatteredPilotExtractor::new(guard); // phase-tracks the TX gather
    let base = params.config();
    let mut eq = OfdmEqualizer::new(&base, EqualizerMethod::PerSymbolPilotInterp);
    let mut rx_cpr = CyclicPrefixRemove::new(n_fft, cp_len);
    let mut rx_fft = FftBlock::new(n_fft);
    let mut rx_time = vec![C32::default(); n_fft];
    let mut rx_freq = vec![C32::default(); n_fft];
    let mut equalized = vec![C32::default(); n_fft];
    let mut rx_data = vec![C32::default(); DVB_T_DATA_CARRIERS];
    let mut tx_data = vec![C32::default(); DVB_T_DATA_CARRIERS];

    let mut max_err = 0.0f32;
    for s in 0..frame.n_symbols {
        let off = s * sps;
        // TX truth: FFT the noiseless transmitted symbol, gather its data bins.
        tx_cpr.process(&frame.iq[off..], &mut tx_time);
        tx_fft.process(&tx_time, &mut tx_freq);
        tx_ext.extract_symbol(&tx_freq, &mut tx_data);
        // RX: same noiseless samples through CP-remove → FFT → equalize → extract.
        rx_cpr.process(&frame.iq[off..], &mut rx_time);
        rx_fft.process(&rx_time, &mut rx_freq);
        eq.set_pilot_bins(ext.phase(), ext.current_pilot_bins(), ext.data_bins());
        eq.process(&rx_freq, &mut equalized);
        ext.extract_symbol(&equalized, &mut rx_data);
        for (a, b) in rx_data.iter().zip(tx_data.iter()) {
            max_err = max_err.max((a - b).norm());
        }
    }
    // Flat unit channel: the equalizer divides by h≈1, so equalized == transmitted
    // to float precision. A TPS carrier admitted as a channel reference instead
    // corrupts the estimate on its neighbours by order-1 (the bogus ratio, or a
    // near-null blow-up through EQUALIZER_FLOOR), so 1e-3 cleanly separates the two.
    assert!(
        max_err < 1e-3,
        "noiseless equalized data carriers must equal transmitted ({constellation:?} {code_rate:?} {guard:?}): max |Δ| = {max_err}"
    );
}

#[test]
fn dvb_t_equalizer_noiseless_clean_qpsk() {
    for guard in [
        GuardInterval::G1_32,
        GuardInterval::G1_16,
        GuardInterval::G1_8,
        GuardInterval::G1_4,
    ] {
        assert_noiseless_equalizer_is_clean(ConstellationOrder::Qpsk, PunctureRate::R1_2, guard);
    }
}

#[test]
fn dvb_t_equalizer_noiseless_clean_qam16() {
    for guard in [
        GuardInterval::G1_32,
        GuardInterval::G1_16,
        GuardInterval::G1_8,
        GuardInterval::G1_4,
    ] {
        assert_noiseless_equalizer_is_clean(ConstellationOrder::Qam16, PunctureRate::R3_4, guard);
    }
}

// ── Null-packet stuffing: every data carrier is filled (EN 300 744 §4.4) ─────
//
// A short payload is padded to a full 68-symbol frame; the modulator stuffs the
// TS stream with null packets so the coded stream reaches the frame's data
// carriers (§4.4: "all symbols contain data"; §4.3.1: randomization stays active
// with no program input) — a compliant DVB-T signal never leaves data carriers
// zeroed. The stuffing must also be transparent: the RX still recovers the real
// payload.
//
// Stuffing stops at the largest packet count that FITS, so it lands short of the
// last carrier by under one packet's coded step; the modulator repeats the coded
// stream's head across the remainder. This test is what pins that filler — swept
// across all fifteen modes because the size of the remainder is a function of
// the code rate, and a rate whose remainder happened to be zero would hide a
// missing fill.
#[test]
fn dvb_t_short_frame_stuffs_all_carriers() {
    use orion_sdr::multicarrier::CyclicPrefixRemove;
    use orion_sdr::waveform::dvb_t::{DVB_T_DATA_CARRIERS, DVB_T_N_FFT, ScatteredPilotExtractor};

    let modes = [
        ConstellationOrder::Qpsk,
        ConstellationOrder::Qam16,
        ConstellationOrder::Qam64,
    ]
    .into_iter()
    .flat_map(|c| {
        [
            PunctureRate::R1_2,
            PunctureRate::R2_3,
            PunctureRate::R3_4,
            PunctureRate::R5_6,
            PunctureRate::R7_8,
        ]
        .into_iter()
        .map(move |r| (c, r))
    });

    for (const_, rate) in modes {
        let params = DvbTFrameParams {
            link: DvbTLinkParams {
                guard: GuardInterval::G1_8,
                constellation: const_,
                code_rate: rate,
            },
            frame_number: 0,
            cell_id: 0,
        };
        // A tiny payload — far short of the frame, so most of it would be padding.
        let payload = sample_payload(184);
        let frame = DvbTFrameMod::new(params).modulate(&payload);
        let n_fft = DVB_T_N_FFT;
        let cp_len = GuardInterval::G1_8.cp_len_2k();
        let sps = n_fft + cp_len;

        // Every data carrier of every symbol must be a real (non-zero) cell.
        let mut ext = ScatteredPilotExtractor::new(GuardInterval::G1_8);
        let mut cpr = CyclicPrefixRemove::new(n_fft, cp_len);
        let mut fft = FftBlock::new(n_fft);
        let mut time = vec![C32::default(); n_fft];
        let mut freq = vec![C32::default(); n_fft];
        let mut data = vec![C32::default(); DVB_T_DATA_CARRIERS];
        let mut zeroed = 0usize;
        for s in 0..frame.n_symbols {
            cpr.process(&frame.iq[s * sps..], &mut time);
            fft.process(&time, &mut freq);
            ext.extract_symbol(&freq, &mut data);
            zeroed += data.iter().filter(|v| v.norm() < 1e-6).count();
        }
        assert_eq!(
            zeroed, 0,
            "no data carrier may be zeroed after null-packet stuffing ({const_:?} {rate:?})"
        );

        // Stuffing is transparent: the payload still round-trips.
        let lead = 200usize;
        let mut buf = vec![C32::default(); lead];
        buf.extend_from_slice(&frame.iq);
        buf.extend(vec![C32::default(); frame.samples_per_symbol]);
        let got = DvbTFrameDemod::new(params)
            .decode(&buf, frame.n_symbols, payload.len())
            .expect("stuffed short frame decodes");
        assert_eq!(got.payload, payload, "payload recovered through stuffing");
    }
}

// ── Integer-CFO estimation (continual-pilot spectral correlation) ───────────
//
// `dvb_t_gi_sync` recovers only the fractional CFO (±½ subcarrier); a large
// front-end offset shifts the whole spectrum by whole subcarriers. The continual
// pilots (fixed positions, boosted) anchor that integer offset. These tests apply
// a KNOWN integer-subcarrier CFO and check `dvb_t_integer_cfo` recovers it — and
// that correcting by the estimate restores an end-to-end decode.

#[test]
fn dvb_t_integer_cfo_recovers_known_shift() {
    use orion_sdr::sync::dvb_t_integer_cfo;
    use orion_sdr::waveform::dvb_t::DVB_T_N_FFT;

    let params = DvbTFrameParams {
        link: DvbTLinkParams {
            guard: GuardInterval::G1_8,
            constellation: ConstellationOrder::Qpsk,
            code_rate: PunctureRate::R1_2,
        },
        frame_number: 0,
        cell_id: 0,
    };
    let payload = sample_payload(184);
    let frame = DvbTFrameMod::new(params).modulate(&payload);
    let n_fft = DVB_T_N_FFT;
    let cp_len = GuardInterval::G1_8.cp_len_2k();

    // FFT symbol 0 (skip its CP) to get its true frequency-domain bins.
    let mut cpr = CyclicPrefixRemove::new(n_fft, cp_len);
    let mut fft = FftBlock::new(n_fft);
    let mut time = vec![C32::default(); n_fft];
    let mut freq = vec![C32::default(); n_fft];
    cpr.process(&frame.iq, &mut time);
    fft.process(&time, &mut freq);

    // No shift → estimate 0. Confidence is the pilot-position energy over the
    // all-shifts mean; the continual pilots are boosted only ~1.78× and are 45 of
    // 1705 carriers, so the true peak sits modestly (~1.7×) above the mean — enough
    // to win the search, but not a large ratio.
    let z = dvb_t_integer_cfo(&freq, n_fft, 32).expect("estimate");
    assert_eq!(z.bins, 0, "no CFO → 0 bins (confidence {})", z.confidence);
    assert!(
        z.confidence > 1.3,
        "true pilots peak above the mean (got {})",
        z.confidence
    );

    // A spectrum shifted by k bins (circular) must be estimated as +k.
    for k in [-7i32, -1, 3, 12] {
        let shifted: Vec<C32> = (0..n_fft)
            .map(|b| freq[(b as i32 - k).rem_euclid(n_fft as i32) as usize])
            .collect();
        let est = dvb_t_integer_cfo(&shifted, n_fft, 32).expect("estimate");
        assert_eq!(est.bins, k, "shift {k} recovered (got {})", est.bins);
    }
}

#[test]
fn dvb_t_integer_cfo_builder_corrects_end_to_end() {
    // Apply a real integer-subcarrier CFO to the whole frame and prove the demod's
    // `with_integer_cfo_correction` builder flag toggles internal recovery: OFF
    // (the default) fails to decode; ON auto-estimates and removes the offset.
    use orion_sdr::dsp::Rotator;
    use orion_sdr::waveform::dvb_t::DVB_T_N_FFT;

    let params = DvbTFrameParams {
        link: DvbTLinkParams {
            guard: GuardInterval::G1_8,
            constellation: ConstellationOrder::Qpsk,
            code_rate: PunctureRate::R1_2,
        },
        frame_number: 0,
        cell_id: 0,
    };
    let payload = sample_payload(184);
    let frame = DvbTFrameMod::new(params).modulate(&payload);
    let n_fft = DVB_T_N_FFT;
    let cp_len = GuardInterval::G1_8.cp_len_2k();
    let sps = n_fft + cp_len;
    let fs = params.config().fs;
    let bin_hz = fs / n_fft as f32;

    // Apply +5 subcarriers of CFO to the whole frame, placed with lead-in.
    let k_true = 5i32;
    let mut shifted = vec![C32::default(); frame.iq.len()];
    Rotator::new(k_true as f32 * bin_hz, fs).rotate_block(&frame.iq, &mut shifted);
    let mut buf = vec![C32::default(); 200];
    buf.extend_from_slice(&shifted);
    buf.extend(vec![C32::default(); sps]);

    // Flag OFF (default): a large integer CFO breaks the demap.
    assert!(
        DvbTFrameDemod::new(params)
            .decode(&buf, frame.n_symbols, payload.len())
            .is_err(),
        "an uncorrected integer CFO must break decode"
    );

    // Flag ON: the demod estimates and removes the offset internally, then decodes.
    let got = DvbTFrameDemod::new(params)
        .with_integer_cfo_correction(true)
        .decode(&buf, frame.n_symbols, payload.len())
        .expect("decode with internal integer-CFO correction");
    assert_eq!(got.payload, payload);
}

#[test]
fn dvb_t_integer_cfo_survives_awgn() {
    // The continual-pilot peak is modest (~1.7×), so confirm the estimate still
    // holds under moderate noise. Accumulating the pilot energy over several
    // symbols sharpens it; here a single symbol at a decodable SNR already works.
    use orion_sdr::sync::dvb_t_integer_cfo;
    use orion_sdr::waveform::dvb_t::DVB_T_N_FFT;

    let params = DvbTFrameParams {
        link: DvbTLinkParams {
            guard: GuardInterval::G1_8,
            constellation: ConstellationOrder::Qpsk,
            code_rate: PunctureRate::R1_2,
        },
        frame_number: 0,
        cell_id: 0,
    };
    let payload = sample_payload(184);
    let frame = DvbTFrameMod::new(params).modulate(&payload);
    let n_fft = DVB_T_N_FFT;
    let cp_len = GuardInterval::G1_8.cp_len_2k();
    let sps = n_fft + cp_len;

    // Sum pilot-position energy over the first few symbols for a firmer estimate,
    // under AWGN at a decodable SNR (noise_scale 0.03 ≈ 15 dB).
    let sig_power: f32 = frame.iq.iter().map(|s| s.norm_sqr()).sum::<f32>() / frame.iq.len() as f32;
    let mut noisy = frame.iq.clone();
    add_awgn(&mut noisy, sig_power * 0.03, 0x1CF0_0777);

    let mut cpr = CyclicPrefixRemove::new(n_fft, cp_len);
    let mut fft = FftBlock::new(n_fft);
    let mut time = vec![C32::default(); n_fft];
    let mut freq = vec![C32::default(); n_fft];
    let mut accum = vec![C32::default(); n_fft];
    // Coherent-magnitude accumulation across 8 symbols: sum |X|² per bin.
    for s in 0..8 {
        cpr.process(&noisy[s * sps..], &mut time);
        fft.process(&time, &mut freq);
        for (a, &x) in accum.iter_mut().zip(freq.iter()) {
            *a += C32::new(x.norm_sqr(), 0.0);
        }
    }
    let est = dvb_t_integer_cfo(&accum, n_fft, 32).expect("estimate");
    assert_eq!(
        est.bins, 0,
        "integer CFO 0 recovered under AWGN (got {})",
        est.bins
    );
}

#[test]
fn dvb_t_shaped_frame_acquires_with_no_lead_in() {
    // The regression this estimator change exists for. Symbol windowing biases
    // the guard-interval ML timing estimate EARLY (roughly a third of
    // `roll_off`), because the taper attenuates each symbol's leading
    // cyclic-prefix samples but not their unwindowed copies in the interior. A
    // negative phase is not representable in a `[0, period)` search, so a plain
    // argmax reported `period − δ` — the right phase but the next symbol.
    //
    // Buffers that start exactly at the frame are not a corner case: it is how
    // `DvbTSuperFrameDemod` slices every constituent frame, and roughly how
    // `DvbTFrameStreamDemod` re-acquires inside a slice it just acquired. See
    // `GiSyncConfig::origin_score_ratio`.
    let params = g1_8_params();
    let payload = sample_payload(184);
    let backoff = 32usize; // free of the pilot-interpolation penalty

    let shapings: [(&str, usize, usize); 5] = [
        ("unshaped", 0, 0),
        ("taper 8", 8, 0),
        ("taper 32", 32, 0),
        ("mask 45", 0, 45),
        ("taper 8 + mask 45", 8, 45),
    ];
    for (label, roll_off, taps) in shapings {
        let mut modu = DvbTFrameMod::new(params);
        if roll_off > 0 {
            modu = modu.with_symbol_window(roll_off);
        }
        if taps > 0 {
            modu = modu.with_tx_lowpass(DvbTFrameMod::tx_lowpass_for_2k(taps, 60.0));
        }
        let frame = modu.modulate(&payload);

        // No lead-in at all: the frame's first sample IS the buffer's first.
        let got = DvbTFrameDemod::new(params)
            .with_rx_window_backoff(backoff)
            .decode(&frame.iq, frame.n_symbols, payload.len())
            .unwrap_or_else(|e| panic!("{label}: zero-lead-in decode failed: {e}"));
        assert_eq!(got.payload, payload, "{label}: payload");
        assert_eq!(got.tps.guard, params.guard(), "{label}: TPS guard");
    }
}
