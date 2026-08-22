// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use num_complex::Complex32 as C32;
use orion_sdr::fec::{ConvCode, CrcKind, InnerFec, InterleaverKind, OuterFec, PunctureRate};
use orion_sdr::modulate::ConstellationOrder;
use orion_sdr::modulate::ofdm_frame::{CodecCache, block_plan};
use orion_sdr::waveform::dvb_t::{
    DVB_T_ACTIVE_CARRIERS, DVB_T_CONTINUAL_PILOTS_2K, DVB_T_DATA_CARRIERS, DVB_T_FRAME_OUTER,
    DVB_T_FRAME_OUTER_IL, DVB_T_FS_1MHZ, DVB_T_FS_2MHZ, DVB_T_FS_333KHZ, DVB_T_KMAX, DVB_T_N_FFT,
    DVB_T_PRBS_INIT, DVB_T_SCATTERED_PHASES, DVB_T_TPS_CARRIERS_2K, DvbTEnergyDispersal,
    DvbTFrameParams, DvbTLinkParams, GuardInterval, active_to_signed, boosted_pilot_value,
    dvb_t_2k_plan, dvb_t_2k_plans, dvb_t_coded_bits, dvb_t_demap_symbol, dvb_t_frame_fill,
    dvb_t_frame_fill_with, dvb_t_fs_for_bandwidth, dvb_t_map_symbol, dvb_t_mcs_table,
    dvb_t_occupied_bw, dvb_t_soft_llr, is_dvb_t_constellation, scattered_pilot_indices,
    tps_carrier_indices, wk_prbs,
};

// ── DVB-T energy dispersal (whitener) ──────────────────────────────────────

#[test]
fn energy_dispersal_prbs_known_answer() {
    // ETSI EN 300 744 §4.3.1: the first 8 PRBS output bits are 0000 0011.
    // Whitening an all-zero stream yields the raw PRBS, so byte 0 must be 0x03
    // (MSB-first) and the next PRBS bits follow. This pins the polynomial
    // (1 + X^14 + X^15), the init (100101010000000), the output tap, and the
    // MSB-first byte order all at once.
    let mut ed = DvbTEnergyDispersal::new();
    let out = ed.feed(&[0u8; 4]);
    assert_eq!(out[0], 0x03, "first PRBS byte (0000_0011) per EN 300 744");
    // Full 4-byte reference from the standard PRBS (independently reproduced):
    assert_eq!(&out, &[0x03, 0xF6, 0x08, 0x34]);
}

#[test]
fn energy_dispersal_init_constant() {
    // The documented init word 100101010000000 over the low 15 bits.
    assert_eq!(DVB_T_PRBS_INIT, 0b100_1010_1000_0000);
    assert_eq!(DVB_T_PRBS_INIT & 0x8000, 0, "init fits in 15 bits");
}

#[test]
fn energy_dispersal_self_inverse() {
    // Whitening from the init, then whitening the result from a fresh init,
    // recovers the original — the descrambler is identical to the scrambler.
    let original: Vec<u8> = (0..200).map(|i| (i * 31 + 7) as u8).collect();

    let mut enc = DvbTEnergyDispersal::new();
    let scrambled = enc.feed(&original);
    assert_ne!(scrambled, original, "whitening must change the data");

    let mut dec = DvbTEnergyDispersal::new();
    let recovered = dec.feed(&scrambled);
    assert_eq!(
        recovered, original,
        "re-whitening from init recovers the stream"
    );
}

#[test]
fn energy_dispersal_streams_across_chunks() {
    // The register carries across feed() calls: chunked feeding equals one-shot.
    let data: Vec<u8> = (0..300).map(|i| (i * 13 + 5) as u8).collect();

    let mut one_shot = DvbTEnergyDispersal::new();
    let full = one_shot.feed(&data);

    let mut chunked = DvbTEnergyDispersal::new();
    let mut acc = Vec::new();
    for c in data.chunks(29) {
        acc.extend_from_slice(&chunked.feed(c));
    }
    assert_eq!(acc, full, "chunked feeds == one-shot whitening");
}

#[test]
fn energy_dispersal_reset_restarts() {
    let data: Vec<u8> = (0..64).map(|i| (i * 3) as u8).collect();
    let mut ed = DvbTEnergyDispersal::new();
    let first = ed.feed(&data);
    ed.reset();
    let second = ed.feed(&data);
    assert_eq!(first, second, "reset returns the PRBS to its init state");
}

#[test]
fn energy_dispersal_feed_in_place_matches_feed() {
    let data: Vec<u8> = (0..80).map(|i| (i * 5 + 2) as u8).collect();
    let mut a = DvbTEnergyDispersal::new();
    let out = a.feed(&data);
    let mut b = DvbTEnergyDispersal::new();
    let mut in_place = data.clone();
    b.feed_in_place(&mut in_place);
    assert_eq!(out, in_place, "feed and feed_in_place agree");
}

// ── DVB-T constellation mapping (Figure 9a) ────────────────────────────────

fn close(a: C32, b: C32) -> bool {
    (a - b).norm() < 1e-5
}

#[test]
fn qam_mapping_known_points_fig9a() {
    let s2 = 1.0 / 2f32.sqrt(); // QPSK scale 1/√2
    let s16 = 1.0 / 10f32.sqrt(); // 16-QAM scale 1/√10
    let s64 = 1.0 / 42f32.sqrt(); // 64-QAM scale 1/√42

    // QPSK: y0y1 = 00 → (Re=+1, Im=+1) per Fig 9a.
    assert!(close(dvb_t_map_symbol(&[0, 0]).unwrap(), C32::new(s2, s2)));
    // QPSK 11 → (−1, −1).
    assert!(close(
        dvb_t_map_symbol(&[1, 1]).unwrap(),
        C32::new(-s2, -s2)
    ));

    // 16-QAM: y0y1y2y3 = 0000 → top-right (Re=+3, Im=+3).
    assert!(close(
        dvb_t_map_symbol(&[0, 0, 0, 0]).unwrap(),
        C32::new(3.0 * s16, 3.0 * s16)
    ));
    // 16-QAM 1101 → (Re=−3, Im=−1): I=(y0,y2)=(1,0), Q=(y1,y3)=(1,1).
    assert!(close(
        dvb_t_map_symbol(&[1, 1, 0, 1]).unwrap(),
        C32::new(-3.0 * s16, -s16)
    ));

    // 64-QAM: all-zero → (Re=+7, Im=+7).
    assert!(close(
        dvb_t_map_symbol(&[0, 0, 0, 0, 0, 0]).unwrap(),
        C32::new(7.0 * s64, 7.0 * s64)
    ));
}

#[test]
fn qam_even_odd_axis_assignment() {
    // Even bits (y0,y2,y4) drive I; odd bits (y1,y3,y5) drive Q. Setting only
    // odd bits must leave Re at its all-even-zero value and move only Im.
    let s16 = 1.0 / 10f32.sqrt();
    // 16-QAM y0y1y2y3 = 0101: I=(y0,y2)=(0,0)→+3, Q=(y1,y3)=(1,1)→−1.
    let sym = dvb_t_map_symbol(&[0, 1, 0, 1]).unwrap();
    assert!(close(sym, C32::new(3.0 * s16, -s16)));
}

#[test]
fn qam_map_demap_round_trip_all_points() {
    // Every constellation point must map then hard-demap back to its bits.
    for &v in &[2usize, 4, 6] {
        for code in 0u32..(1 << v) {
            let bits: Vec<u8> = (0..v).rev().map(|b| ((code >> b) & 1) as u8).collect();
            let sym = dvb_t_map_symbol(&bits).unwrap();
            let back = dvb_t_demap_symbol(sym, v).unwrap();
            assert_eq!(back, bits, "v={v} code={code:0width$b}", width = v);
        }
    }
}

#[test]
fn qam_unit_average_energy() {
    // Average symbol energy over all points must be ~1 (unit-normalized), so the
    // DVB-T constellation carries the same power as the generic mapper.
    for &v in &[2usize, 4, 6] {
        let n = 1u32 << v;
        let mut e = 0.0f32;
        for code in 0..n {
            let bits: Vec<u8> = (0..v).rev().map(|b| ((code >> b) & 1) as u8).collect();
            e += dvb_t_map_symbol(&bits).unwrap().norm_sqr();
        }
        e /= n as f32;
        assert!((e - 1.0).abs() < 1e-4, "v={v}: avg energy {e} != 1");
    }
}

#[test]
fn qam_unsupported_order_is_none() {
    assert!(dvb_t_map_symbol(&[0, 0, 0]).is_none()); // v=3 unsupported
    assert!(dvb_t_demap_symbol(C32::new(0.0, 0.0), 8).is_none()); // 256-QAM not in DVB-T
}

// ── DVB-T soft-decision LLR ─────────────────────────────────────────────────

#[test]
fn soft_llr_sign_matches_hard_demap() {
    // For every DVB-T constellation point, the sign of each LLR (positive ⇒ bit 0)
    // must agree with the hard-demapped bit at a clean received symbol.
    for &v in &[2usize, 4, 6] {
        for code in 0u32..(1 << v) {
            let bits: Vec<u8> = (0..v).rev().map(|b| ((code >> b) & 1) as u8).collect();
            let sym = dvb_t_map_symbol(&bits).unwrap();
            let llr = dvb_t_soft_llr(sym, v).unwrap();
            assert_eq!(llr.len(), v);
            for (i, (&bit, &l)) in bits.iter().zip(llr.iter()).enumerate() {
                // bit 0 ⇒ LLR > 0; bit 1 ⇒ LLR < 0.
                let hard = u8::from(l <= 0.0);
                assert_eq!(hard, bit, "v={v} code={code} bit {i}: LLR {l} vs bit {bit}");
            }
        }
    }
}

#[test]
fn soft_llr_unsupported_order_is_none() {
    assert!(dvb_t_soft_llr(C32::new(0.0, 0.0), 3).is_none());
    assert!(dvb_t_soft_llr(C32::new(0.0, 0.0), 8).is_none());
}

#[test]
fn soft_llr_magnitude_grows_with_confidence() {
    // A symbol pushed far past a QPSK decision boundary yields a larger-magnitude
    // LLR than one near the axis (more confident bit decision).
    let near = dvb_t_soft_llr(C32::new(0.05, 0.05), 2).unwrap();
    let far = dvb_t_soft_llr(C32::new(3.0, 3.0), 2).unwrap();
    assert!(
        far[0].abs() > near[0].abs(),
        "far symbol should be more confident"
    );
    // Deep in the (+,+) quadrant both bits decide to 0 (positive LLR).
    assert!(far[0] > 0.0 && far[1] > 0.0);
}

#[test]
fn is_dvb_t_constellation_membership() {
    use orion_sdr::modulate::ConstellationOrder::*;
    assert!(is_dvb_t_constellation(Qpsk));
    assert!(is_dvb_t_constellation(Qam16));
    assert!(is_dvb_t_constellation(Qam64));
    assert!(!is_dvb_t_constellation(Bpsk));
    assert!(!is_dvb_t_constellation(Qam256));
}

// ── 2K numerology and carrier map ──────────────────────────────────────────

#[test]
fn numerology_constants() {
    assert_eq!(DVB_T_N_FFT, 2048);
    assert_eq!(DVB_T_KMAX, 1704);
    assert_eq!(DVB_T_ACTIVE_CARRIERS, 1705);
    assert_eq!(DVB_T_DATA_CARRIERS, 1512);
}

#[test]
fn guard_interval_cp_lengths() {
    assert_eq!(GuardInterval::G1_32.cp_len_2k(), 64);
    assert_eq!(GuardInterval::G1_16.cp_len_2k(), 128);
    assert_eq!(GuardInterval::G1_8.cp_len_2k(), 256);
    assert_eq!(GuardInterval::G1_4.cp_len_2k(), 512);
}

#[test]
fn continual_pilots_table_valid() {
    // 45 continual pilots (EN 300 744 Table 7), in range, monotonic, unique.
    assert_eq!(DVB_T_CONTINUAL_PILOTS_2K.len(), 45);
    let mut prev = None;
    for &k in &DVB_T_CONTINUAL_PILOTS_2K {
        assert!(k <= DVB_T_KMAX, "pilot {k} out of range");
        if let Some(p) = prev {
            assert!(k > p, "pilots must be strictly increasing");
        }
        prev = Some(k);
    }
    // First and last per the table.
    assert_eq!(DVB_T_CONTINUAL_PILOTS_2K[0], 0);
    assert_eq!(*DVB_T_CONTINUAL_PILOTS_2K.last().unwrap(), 1704);
}

#[test]
fn wk_prbs_known_start() {
    // EN 300 744 §4.5.2 / figure 10: "PRBS sequence starts: 1111111111100...".
    let bits = wk_prbs(13);
    assert_eq!(&bits[..13], &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]);
}

#[test]
fn boosted_pilot_amplitude() {
    // w_k=0 → +4/3, w_k=1 → −4/3; boosted so |c|² = 16/9.
    let p0 = boosted_pilot_value(0);
    let p1 = boosted_pilot_value(1);
    assert!((p0.re - 4.0 / 3.0).abs() < 1e-6);
    assert!((p1.re + 4.0 / 3.0).abs() < 1e-6);
    assert!((p0.norm_sqr() - 16.0 / 9.0).abs() < 1e-5);
}

#[test]
fn active_to_signed_centering() {
    assert_eq!(active_to_signed(0), -852);
    assert_eq!(active_to_signed(852), 0); // active 852 → DC
    assert_eq!(active_to_signed(1704), 852);
}

#[test]
fn dvb_t_2k_plan_is_valid() {
    let plan = dvb_t_2k_plan(GuardInterval::G1_32);
    assert_eq!(plan.n_fft(), 2048);
    assert_eq!(plan.cp_len(), 64);
    assert_eq!(plan.pilot_carriers().len(), 45);
    // Phase 1: scattered/TPS positions carry data → 1705 − 45 = 1660 data.
    assert_eq!(plan.data_carriers().len(), 1660);
    // No data/pilot overlap, all indices in range.
    plan.validate()
        .expect("2K plan must be a valid CarrierPlan");
}

#[test]
fn fs_bandwidth_scaling() {
    // fs = BW · 2048/1705 ; round-trips with the inverse.
    let fs_1m = dvb_t_fs_for_bandwidth(1_000_000.0);
    assert!((fs_1m - DVB_T_FS_1MHZ).abs() < 1.0);
    assert!((dvb_t_occupied_bw(fs_1m) - 1_000_000.0).abs() < 1.0);
    // The three mode constants scale linearly.
    assert!((DVB_T_FS_2MHZ - 2.0 * DVB_T_FS_1MHZ).abs() < 1.0);
    // ~1 MHz mode is ~1.2012 MS/s (1e6 · 2048/1705).
    assert!((DVB_T_FS_1MHZ - 1_201_173.0).abs() < 100.0);
    // 333 kHz is below Pluto's ~521 kS/s floor (documented, still valid). Route
    // through the runtime helper so this is a real check, not a const compare.
    let fs_333k = dvb_t_fs_for_bandwidth(333_000.0);
    assert!((DVB_T_FS_333KHZ - fs_333k).abs() < 1.0);
    assert!(
        fs_333k < 521_000.0,
        "333 kHz mode fs {fs_333k} below Pluto floor"
    );
}

#[test]
fn nb_bandwidth_modes() {
    use orion_sdr::waveform::dvb_t::NbBandwidth;
    // occupied_hz() matches the nominal channel widths.
    assert_eq!(NbBandwidth::Bw333kHz.occupied_hz(), 333_000.0);
    assert_eq!(NbBandwidth::Bw1MHz.occupied_hz(), 1_000_000.0);
    assert_eq!(NbBandwidth::Bw2MHz.occupied_hz(), 2_000_000.0);
    // fs() equals the corresponding constant (fs = BW · 2048/1705).
    assert!((NbBandwidth::Bw333kHz.fs() - DVB_T_FS_333KHZ).abs() < 1.0);
    assert!((NbBandwidth::Bw1MHz.fs() - DVB_T_FS_1MHZ).abs() < 1.0);
    assert!((NbBandwidth::Bw2MHz.fs() - DVB_T_FS_2MHZ).abs() < 1.0);
    // Pluto continuous-TX floor: 333 kHz is below, 1/2 MHz are above.
    assert!(!NbBandwidth::Bw333kHz.is_pluto_continuous_tx());
    assert!(NbBandwidth::Bw1MHz.is_pluto_continuous_tx());
    assert!(NbBandwidth::Bw2MHz.is_pluto_continuous_tx());
}

#[test]
fn nb_bandwidth_composes_with_config_builders() {
    use orion_sdr::waveform::dvb_t::{NbBandwidth, dvb_t_config, dvb_t_scattered_config};
    // A named mode composes with the existing config builders via occupied_hz().
    let a = dvb_t_config(GuardInterval::G1_32, NbBandwidth::Bw1MHz.occupied_hz());
    let b = dvb_t_config(GuardInterval::G1_32, 1_000_000.0);
    assert_eq!(a.fs, b.fs);
    assert_eq!(a.carrier_plan, b.carrier_plan);

    // Or set fs directly with the generic with_fs() builder.
    let c =
        dvb_t_scattered_config(GuardInterval::G1_8, 1_000_000.0).with_fs(NbBandwidth::Bw2MHz.fs());
    assert!((c.fs - NbBandwidth::Bw2MHz.fs()).abs() < 1.0);
    assert!(c.dvb_t_scattered);
}

#[test]
fn mcs_table_dvb_t() {
    let t = dvb_t_mcs_table();
    assert_eq!(t.len(), 3);
    // Every entry uses the K=7 conv inner + RS(204,188) outer.
    let m0 = t.get(0).unwrap();
    assert_eq!(m0.constellation, ConstellationOrder::Qpsk);
    assert!(matches!(
        m0.inner_fec,
        InnerFec::Convolutional {
            code: ConvCode::DvbK7,
            ..
        }
    ));
    assert!(matches!(
        m0.outer_fec,
        OuterFec::ReedSolomon {
            n: 204,
            n_parity: 16
        }
    ));
    assert_eq!(t.get(2).unwrap().constellation, ConstellationOrder::Qam16);
}

// ── Scattered pilots (Phase 2) ─────────────────────────────────────────────

#[test]
fn tps_carriers_table_valid() {
    // 17 TPS carriers (EN 300 744 Table 8, 2K), in range, unique, sorted.
    let tps = tps_carrier_indices();
    assert_eq!(tps, &DVB_T_TPS_CARRIERS_2K);
    assert_eq!(tps.len(), 17);
    let mut prev = None;
    for &k in tps {
        assert!(k <= DVB_T_KMAX, "TPS carrier {k} out of range");
        if let Some(p) = prev {
            assert!(k > p, "TPS carriers must be strictly increasing");
        }
        prev = Some(k);
    }
    // Endpoints per the table.
    assert_eq!(tps[0], 34);
    assert_eq!(*tps.last().unwrap(), 1687);
}

#[test]
fn scattered_indices_satisfy_formula() {
    // EN 300 744 §4.5.3: scattered pilots for symbol phase l are the carriers
    // with k mod 12 == 3·(l mod 4), Kmin=0..=Kmax.
    for phase in 0..DVB_T_SCATTERED_PHASES {
        let idx = scattered_pilot_indices(phase);
        assert!(!idx.is_empty());
        let start = 3 * phase;
        assert_eq!(idx[0], start, "phase {phase} starts at 3·phase");
        let mut prev = None;
        for &k in &idx {
            assert!(k <= DVB_T_KMAX);
            assert_eq!(k % 12, (3 * phase) % 12, "phase {phase}: k mod 12");
            if let Some(p) = prev {
                assert_eq!(k - p, 12, "scattered pilots step by 12");
            }
            prev = Some(k);
        }
    }
}

#[test]
fn scattered_plans_have_exactly_1512_data() {
    // The load-bearing invariant: all four symbol-phase plans expose the same,
    // conformant 1512 data carriers (EN 300 744 §4.5 fixes the spacing so the
    // useful-carrier count is constant). This keeps the frame layer's count-based
    // bookkeeping valid while the physical bins rotate.
    for guard in [
        GuardInterval::G1_32,
        GuardInterval::G1_16,
        GuardInterval::G1_8,
        GuardInterval::G1_4,
    ] {
        let plans = dvb_t_2k_plans(guard);
        assert_eq!(plans.len(), DVB_T_SCATTERED_PHASES);
        for (phase, plan) in plans.iter().enumerate() {
            assert_eq!(
                plan.data_carriers().len(),
                DVB_T_DATA_CARRIERS,
                "guard {guard:?} phase {phase}: data count"
            );
            assert_eq!(plan.n_fft(), DVB_T_N_FFT);
            assert_eq!(plan.cp_len(), guard.cp_len_2k());
            // No data/pilot overlap, all indices in range.
            plan.validate()
                .unwrap_or_else(|e| panic!("phase {phase} plan invalid: {e}"));
        }
    }
}

#[test]
fn scattered_plans_reserve_continual_scattered_tps() {
    // Every plan reserves the 45 continual + phase-p scattered + 17 TPS carriers
    // as pilots (deduped), and nothing else, so data = 1705 − reserved = 1512.
    for phase in 0..DVB_T_SCATTERED_PHASES {
        let plan = &dvb_t_2k_plans(GuardInterval::G1_32)[phase];
        let pilot_set: std::collections::BTreeSet<i32> =
            plan.pilot_carriers().iter().map(|&(a, _)| a).collect();
        // Every continual pilot is reserved.
        for &c in &DVB_T_CONTINUAL_PILOTS_2K {
            assert!(pilot_set.contains(&active_to_signed(c)), "continual {c}");
        }
        // Every TPS carrier is reserved.
        for &t in &DVB_T_TPS_CARRIERS_2K {
            assert!(pilot_set.contains(&active_to_signed(t)), "TPS {t}");
        }
        // Every phase-p scattered pilot is reserved.
        for k in scattered_pilot_indices(phase) {
            assert!(pilot_set.contains(&active_to_signed(k)), "scattered {k}");
        }
    }
}

#[test]
fn scattered_pilots_are_boosted_wk_valued() {
    // Pilots carry the boosted ±4/3 value derived from w_k at their carrier index
    // (continual and scattered share the same reference sequence, §4.5).
    let wk = wk_prbs(DVB_T_ACTIVE_CARRIERS);
    let plan = &dvb_t_2k_plans(GuardInterval::G1_32)[1];
    for &(signed, value) in plan.pilot_carriers() {
        let a = (signed + 852) as usize;
        let expect = boosted_pilot_value(wk[a]);
        assert!((value - expect).norm() < 1e-6, "pilot at active {a}");
        assert!(
            (value.norm_sqr() - 16.0 / 9.0).abs() < 1e-4,
            "boosted power"
        );
    }
}

// ── Frame filling (the shared TX/RX/downstream rule) ────────────────────────
//
// `dvb_t_frame_fill` is the single statement of how a DVB-T frame's data
// carriers are filled. Three callers depend on it agreeing with itself — the
// modulator maps by it, the receiver decodes and re-encodes against it, and
// downstream consumers size payloads by it — and any two of them disagreeing is
// a silent wrong-numbers bug, not a loud one. The combinatorial claim is pure
// arithmetic, so it is asserted exhaustively here rather than sampled through an
// end-to-end decode.

/// Every DVB-T constellation/rate pair (the fifteen the TPS word can signal).
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

fn fill_params(constellation: ConstellationOrder, code_rate: PunctureRate) -> DvbTFrameParams {
    DvbTFrameParams {
        link: DvbTLinkParams {
            guard: GuardInterval::G1_8,
            constellation,
            code_rate,
        },
        frame_number: 0,
        cell_id: 0,
    }
}

/// The frame geometry the modulator derives for `n_pkt` real payload packets:
/// enough symbols for their coded stream, padded to a full 68-symbol TPS block.
fn symbols_for_packets(params: DvbTFrameParams, n_pkt: usize) -> usize {
    let bits_per_sym = DVB_T_DATA_CARRIERS * params.constellation().bits_per_symbol();
    dvb_t_coded_bits(params, n_pkt)
        .div_ceil(bits_per_sym)
        .max(68)
}

#[test]
fn frame_fill_never_overruns_the_carriers() {
    // The property the whole fix rests on, and the one the old `>=` rule
    // inverted: the coded stream must END ON OR BEFORE the last data carrier, so
    // a receiver reconstructing what was transmitted never asks its decoder for
    // bits that were never sent. Asserted with maximality, because a rule that
    // merely fits could satisfy this by stuffing nothing.
    for (c, r) in every_mode() {
        let params = fill_params(c, r);
        for n_pkt in (1..=200).step_by(7) {
            let n_symbols = symbols_for_packets(params, n_pkt);
            let fill = dvb_t_frame_fill(params, n_pkt, n_symbols);

            assert!(
                fill.coded_bits <= fill.capacity_bits,
                "{c:?} {r:?} {n_pkt} pkts: coded {} overruns capacity {}",
                fill.coded_bits,
                fill.capacity_bits
            );
            // Maximal: one more packet would not fit.
            assert!(
                dvb_t_coded_bits(params, fill.n_ts_packets + 1) > fill.capacity_bits,
                "{c:?} {r:?} {n_pkt} pkts: {} is not the largest count that fits",
                fill.n_ts_packets
            );
            // "Largest that fits" never drops a real payload packet — `n_symbols`
            // is derived from the payload's own coded length, so it always fits.
            assert!(
                fill.n_ts_packets >= n_pkt,
                "{c:?} {r:?} {n_pkt} pkts: filling dropped payload down to {}",
                fill.n_ts_packets
            );
            // The remainder the modulator repeats is under one packet's coded
            // step, so `extend_from_within` can always draw it from the head.
            assert!(
                fill.filler_bits() < fill.coded_bits,
                "{c:?} {r:?} {n_pkt} pkts: filler {} exceeds the stream it repeats",
                fill.filler_bits()
            );
        }
    }
}

#[test]
fn frame_fill_is_the_modulators_block_plan() {
    // The rule and the coding chain must be the same statement, not two that
    // agree by inspection: `coded_bits` has to be exactly what `block_plan`
    // computes for the packet count the rule chose. If these ever diverge, the
    // receiver's truth re-encode compares against the wrong length and reports a
    // plausible-but-wrong BER rather than failing.
    let cache = CodecCache::new();
    for (c, r) in every_mode() {
        let params = fill_params(c, r);
        for n_pkt in (1..=200).step_by(11) {
            let n_symbols = symbols_for_packets(params, n_pkt);
            let fill = dvb_t_frame_fill(params, n_pkt, n_symbols);
            let plan = block_plan(
                fill.n_ts_packets * 188,
                CrcKind::None,
                DVB_T_FRAME_OUTER,
                params.inner(),
                DVB_T_FRAME_OUTER_IL,
                InterleaverKind::None,
                &cache,
            );
            assert_eq!(
                plan.coded_bits, fill.coded_bits,
                "{c:?} {r:?} {n_pkt} pkts: fill and block plan disagree"
            );
            // And the cached form is the same function.
            assert_eq!(
                dvb_t_frame_fill_with(params, n_pkt, n_symbols, &cache),
                fill,
                "{c:?} {r:?} {n_pkt} pkts: cached and uncached fill disagree"
            );
        }
    }
}
