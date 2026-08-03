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
use orion_sdr::demodulate::demodulate_frame;
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
    let got = demodulate_frame(&cfg, &table, &body, None).expect("DVB-T frame decode");
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
    let got = demodulate_frame(&cfg, &table, &body, None).expect("DVB-T AWGN decode");
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
    let got = demodulate_frame(&cfg, &table, &body, None).expect("scattered frame decode");
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
    let got =
        demodulate_frame(&cfg, &table, &body, None).expect("scattered multipath frame decode");
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
    let got = demodulate_frame(&cfg, &table, &body, None);
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
// TX: `modulate::dvb_t_frame_modulate`; RX: `demodulate::dvb_t_frame_demodulate`.

use orion_sdr::demodulate::dvb_t_frame_demodulate;
use orion_sdr::fec::PunctureRate;
use orion_sdr::modulate::{ConstellationOrder, dvb_t_frame_modulate};
use orion_sdr::waveform::dvb_t::DvbTFrameParams;

fn capstone_params() -> DvbTFrameParams {
    DvbTFrameParams {
        guard: GuardInterval::G1_32,
        constellation: ConstellationOrder::Qpsk,
        code_rate: PunctureRate::R1_2,
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
    let frame = dvb_t_frame_modulate(params, &payload);

    let lead = 200usize;
    let mut buf = vec![C32::default(); lead];
    buf.extend_from_slice(&frame.iq);
    buf.extend(vec![C32::default(); frame.samples_per_symbol]);

    let got = dvb_t_frame_demodulate(params, &buf, frame.n_symbols, payload.len())
        .expect("conformant DVB-T frame decode");
    assert_eq!(got.payload, payload, "recovered TS payload");
    assert_eq!(got.tps.frame_number, params.frame_number);
    assert_eq!(got.tps.constellation, params.constellation);
    assert_eq!(got.tps.code_rate_hp, params.code_rate);
    assert_eq!(got.tps.guard, params.guard);
    assert_eq!(got.tps.cell_id, params.cell_id);
}

#[test]
fn dvb_t_tps_frame_survives_awgn() {
    // The same conformant frame through modest AWGN: the DBPSK TPS (17-carrier
    // averaged, BCH-protected) and the soft-decision payload FEC both hold.
    let params = capstone_params();
    let payload = sample_payload(184);
    let frame = dvb_t_frame_modulate(params, &payload);

    let lead = 128usize;
    let mut buf = vec![C32::default(); lead];
    buf.extend_from_slice(&frame.iq);
    buf.extend(vec![C32::default(); frame.samples_per_symbol]);
    let sig_power: f32 = frame.iq.iter().map(|s| s.norm_sqr()).sum::<f32>() / frame.iq.len() as f32;
    add_awgn(&mut buf, sig_power * 0.03, 0x0DB7_0777);

    let got = dvb_t_frame_demodulate(params, &buf, frame.n_symbols, payload.len())
        .expect("DVB-T frame AWGN decode");
    assert_eq!(got.payload, payload);
    assert_eq!(got.tps.constellation, params.constellation);
}

// ── Equalizer channel-reference regression (the TPS-pilot bug) ──────────────
//
// Guards a subtle correctness property: the RX scattered-pilot equalizer must NOT
// use the 17 TPS carriers as channel references. The modulator transmits data-
// power DBPSK (±1.0) on the TPS bins, but the grid records them as boosted `w_k`
// pilots (±4/3); feeding them to the estimator would yield `h = ±1.0/±4/3 = ∓0.75`
// (wrong magnitude AND sign), which `interpolate_at` then smears onto the ~3 data
// carriers straddling each TPS carrier — a deterministic, SNR-invariant pre-FEC
// error floor. A payload-only decode assertion at zero noise can pass on a thin
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
        guard,
        constellation,
        code_rate,
        frame_number: 0,
        cell_id: 0,
    };
    let payload = sample_payload(184);
    let frame = dvb_t_frame_modulate(params, &payload);
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
        let pilots = ext.current_pilot_bins().to_vec();
        let data_bins = ext.data_bins().to_vec();
        eq.set_pilot_bins(&pilots, &data_bins);
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
