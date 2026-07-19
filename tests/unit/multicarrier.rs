// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use num_complex::Complex32 as C32;
use orion_sdr::core::Block;
use orion_sdr::multicarrier::{
    CarrierGrid, CarrierPlan, CarrierPlanError, CyclicPrefixInsert, CyclicPrefixRemove, FftBlock,
    GridExtract, GridMap, IfftBlock, SubcarrierRole,
};

fn tone(n: usize, cycles: f32) -> Vec<C32> {
    (0..n)
        .map(|k| {
            let phase = std::f32::consts::TAU * cycles * (k as f32) / (n as f32);
            C32::new(phase.cos(), phase.sin())
        })
        .collect()
}

#[test]
fn fft_ifft_roundtrip_identity() {
    let n_fft = 64;
    let input = tone(n_fft, 3.0);
    let mut freq = vec![C32::default(); n_fft];
    let mut time = vec![C32::default(); n_fft];

    let mut fft = FftBlock::new(n_fft);
    let mut ifft = IfftBlock::new(n_fft);

    fft.process(&input, &mut freq);
    ifft.process(&freq, &mut time);

    let eps = 1e-4f32;
    for (a, b) in input.iter().zip(time.iter()) {
        assert!(
            (a - b).norm() < eps,
            "roundtrip mismatch: {:?} vs {:?}",
            a,
            b
        );
    }
}

#[test]
fn fft_dc_bin_impulse() {
    // A constant (DC) input should produce all its energy in bin 0 under a
    // unity-gain forward FFT.
    let n_fft = 32;
    let input = vec![C32::new(1.0, 0.0); n_fft];
    let mut freq = vec![C32::default(); n_fft];

    FftBlock::new(n_fft).process(&input, &mut freq);

    let eps = 1e-3f32;
    assert!(
        (freq[0].re - n_fft as f32).abs() < eps,
        "DC bin expected {}, got {:?}",
        n_fft,
        freq[0]
    );
    for bin in &freq[1..] {
        assert!(bin.norm() < eps, "non-DC bin not silent: {:?}", bin);
    }
}

#[test]
fn fft_partial_chunk_is_noop() {
    let n_fft = 64;
    let input = tone(n_fft - 1, 1.0); // one sample short
    let mut output = vec![C32::default(); n_fft];

    let wr = FftBlock::new(n_fft).process(&input, &mut output);
    assert_eq!(wr.in_read, 0);
    assert_eq!(wr.out_written, 0);
}

#[test]
fn fft_multi_symbol_chunk() {
    // process() only consumes one n_fft-sized symbol per call, even if more
    // input/output is available.
    let n_fft = 16;
    let input = tone(n_fft * 3, 2.0);
    let mut output = vec![C32::default(); n_fft * 3];

    let wr = FftBlock::new(n_fft).process(&input, &mut output);
    assert_eq!(wr.in_read, n_fft);
    assert_eq!(wr.out_written, n_fft);
}

#[test]
fn cyclic_prefix_insert_content() {
    let n_fft = 8;
    let cp_len = 3;
    let input: Vec<C32> = (0..n_fft).map(|k| C32::new(k as f32, 0.0)).collect();
    let mut output = vec![C32::default(); n_fft + cp_len];

    let wr = CyclicPrefixInsert::new(n_fft, cp_len).process(&input, &mut output);
    assert_eq!(wr.in_read, n_fft);
    assert_eq!(wr.out_written, n_fft + cp_len);

    // CP is the last cp_len samples of the symbol, copied to the front.
    assert_eq!(output[..cp_len], input[n_fft - cp_len..]);
    assert_eq!(output[cp_len..], input[..]);
}

#[test]
fn cyclic_prefix_roundtrip() {
    let n_fft = 16;
    let cp_len = 4;
    let input: Vec<C32> = (0..n_fft)
        .map(|k| C32::new(k as f32 * 0.5, -(k as f32)))
        .collect();
    let mut with_cp = vec![C32::default(); n_fft + cp_len];
    let mut restored = vec![C32::default(); n_fft];

    CyclicPrefixInsert::new(n_fft, cp_len).process(&input, &mut with_cp);
    CyclicPrefixRemove::new(n_fft, cp_len).process(&with_cp, &mut restored);

    // Exact match expected — no float error since this is a pure copy.
    assert_eq!(restored, input);
}

#[test]
fn cyclic_prefix_zero_length_cp() {
    let n_fft = 8;
    let cp_len = 0;
    let input: Vec<C32> = (0..n_fft).map(|k| C32::new(k as f32, 0.0)).collect();
    let mut output = vec![C32::default(); n_fft];

    let wr = CyclicPrefixInsert::new(n_fft, cp_len).process(&input, &mut output);
    assert_eq!(wr.in_read, n_fft);
    assert_eq!(wr.out_written, n_fft);
    assert_eq!(output, input);
}

#[test]
fn carrier_plan_validate_rejects_overlap() {
    let plan = CarrierPlan::new(64, 8)
        .with_data_carriers([1, 2, 3])
        .with_pilot_carriers([(3, C32::new(1.0, 0.0))]);
    assert_eq!(plan.validate(), Err(CarrierPlanError::Overlap(3)));
}

#[test]
fn carrier_plan_validate_rejects_out_of_range() {
    // n_fft=64 → valid signed range is -32..=31
    let plan = CarrierPlan::new(64, 8).with_data_carriers([1, 2, 32]);
    assert_eq!(plan.validate(), Err(CarrierPlanError::OutOfRange(32, 64)));
}

#[test]
fn carrier_plan_validate_rejects_empty_data_set() {
    let plan = CarrierPlan::new(64, 8);
    assert_eq!(plan.validate(), Err(CarrierPlanError::EmptyDataSet));
}

#[test]
fn carrier_plan_validate_accepts_well_formed_plan() {
    let plan = CarrierPlan::new(64, 8)
        .with_data_carriers([-26, -25, 1, 2, 25, 26])
        .with_pilot_carriers([(-21, C32::new(1.0, 0.0)), (21, C32::new(-1.0, 0.0))]);
    assert_eq!(plan.validate(), Ok(()));
}

#[test]
#[should_panic(expected = "invalid CarrierPlan")]
fn carrier_grid_from_plan_panics_on_overlap() {
    // A bin used as both data and pilot fails validate(); from_plan must
    // reject it rather than silently push it into both data_bins and
    // pilot_bins (where GridMap would overwrite the data value with the
    // pilot). Guards the whole Rust OFDM pipeline at construction.
    let plan = CarrierPlan::new(64, 8)
        .with_data_carriers([1, 2, 3])
        .with_pilot_carriers([(3, C32::new(1.0, 0.0))]);
    let _ = CarrierGrid::from_plan(&plan);
}

#[test]
#[should_panic(expected = "invalid CarrierPlan")]
fn carrier_grid_from_plan_panics_on_out_of_range() {
    // Carrier 8 is out of range for n_fft=16 (valid signed range -8..=7):
    // +8 is the Nyquist bin, excluded on the positive side. Without the
    // validate() gate, rem_euclid would silently accept it.
    let plan = CarrierPlan::new(16, 4).with_data_carriers([1, 8]);
    let _ = CarrierGrid::from_plan(&plan);
}

#[test]
fn carrier_grid_bin_mapping_negative_wraps() {
    let n_fft = 16;
    let plan = CarrierPlan::new(n_fft, 4).with_data_carriers([-1, -2, 1, 2]);
    let grid = CarrierGrid::from_plan(&plan);

    // Negative carrier indices wrap into the top half of the FFT (natural
    // rustfft bin order): -1 -> n_fft-1, -2 -> n_fft-2.
    assert_eq!(grid.role()[n_fft - 1], SubcarrierRole::Data);
    assert_eq!(grid.role()[n_fft - 2], SubcarrierRole::Data);
    assert_eq!(grid.role()[1], SubcarrierRole::Data);
    assert_eq!(grid.role()[2], SubcarrierRole::Data);
    // DC (bin 0) is implicitly null since it wasn't explicitly included.
    assert_eq!(grid.role()[0], SubcarrierRole::Null);
}

#[test]
fn carrier_grid_data_bins_order_matches_carrier_order() {
    let n_fft = 16;
    let plan = CarrierPlan::new(n_fft, 4).with_data_carriers([2, -1, 5]);
    let grid = CarrierGrid::from_plan(&plan);

    assert_eq!(grid.data_bins(), &[2, n_fft - 1, 5]);
}

fn small_grid() -> CarrierGrid {
    let n_fft = 8;
    let plan = CarrierPlan::new(n_fft, 2)
        .with_data_carriers([1, 2, 3])
        .with_pilot_carriers([(-1, C32::new(0.5, 0.5))]);
    CarrierGrid::from_plan(&plan)
}

#[test]
fn grid_map_extract_roundtrip() {
    let grid = small_grid();
    let n_fft = grid.n_fft();
    let n_data = grid.num_data_carriers();

    let mut map = GridMap::new(grid.clone());
    let mut extract = GridExtract::new(grid);

    let data_in: Vec<C32> = (0..n_data)
        .map(|k| C32::new(k as f32 + 1.0, -(k as f32)))
        .collect();
    let mut freq = vec![C32::default(); n_fft];
    let mut data_out = vec![C32::default(); n_data];

    let wr_map = map.process(&data_in, &mut freq);
    assert_eq!(wr_map.in_read, n_data);
    assert_eq!(wr_map.out_written, n_fft);

    let wr_extract = extract.process(&freq, &mut data_out);
    assert_eq!(wr_extract.in_read, n_fft);
    assert_eq!(wr_extract.out_written, n_data);

    assert_eq!(data_out, data_in);
}

#[test]
fn grid_map_zeros_null_and_writes_pilots() {
    let grid = small_grid();
    let n_fft = grid.n_fft();
    let n_data = grid.num_data_carriers();

    let mut map = GridMap::new(grid);
    let data_in = vec![C32::new(1.0, 0.0); n_data];
    let mut freq = vec![C32::new(99.0, 99.0); n_fft]; // pre-poison to catch missed nulls

    map.process(&data_in, &mut freq);

    // Pilot bin (-1 -> n_fft-1) carries its known value.
    assert_eq!(freq[n_fft - 1], C32::new(0.5, 0.5));
    // Data bins carry the mapped input.
    assert_eq!(freq[1], C32::new(1.0, 0.0));
    assert_eq!(freq[2], C32::new(1.0, 0.0));
    assert_eq!(freq[3], C32::new(1.0, 0.0));
    // All remaining bins are null (zeroed), including DC.
    for &bin in &[0usize, 4, 5, 6] {
        assert_eq!(freq[bin], C32::default(), "bin {} not zeroed", bin);
    }
}

#[test]
fn grid_map_partial_chunk_is_noop() {
    let grid = small_grid();
    let n_fft = grid.n_fft();
    let n_data = grid.num_data_carriers();

    let mut map = GridMap::new(grid);
    let data_in = vec![C32::default(); n_data - 1]; // one symbol short
    let mut freq = vec![C32::default(); n_fft];

    let wr = map.process(&data_in, &mut freq);
    assert_eq!(wr.in_read, 0);
    assert_eq!(wr.out_written, 0);
}
