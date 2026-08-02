// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use num_complex::Complex32 as C32;
use orion_sdr::fec::{ConvCode, InnerFec, OuterFec};
use orion_sdr::modulate::ConstellationOrder;
use orion_sdr::waveform::dvb_t::{
    DVB_T_ACTIVE_CARRIERS, DVB_T_CONTINUAL_PILOTS_2K, DVB_T_DATA_CARRIERS, DVB_T_FS_1MHZ,
    DVB_T_FS_2MHZ, DVB_T_FS_333KHZ, DVB_T_KMAX, DVB_T_N_FFT, DVB_T_PRBS_INIT, DvbTEnergyDispersal,
    GuardInterval, active_to_signed, boosted_pilot_value, dvb_t_2k_plan, dvb_t_demap_symbol,
    dvb_t_fs_for_bandwidth, dvb_t_map_symbol, dvb_t_mcs_table, dvb_t_occupied_bw, wk_prbs,
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
