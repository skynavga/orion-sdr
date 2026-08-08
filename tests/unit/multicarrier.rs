// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use num_complex::Complex32 as C32;
use orion_sdr::core::Block;
use orion_sdr::multicarrier::{
    CarrierGrid, CarrierPlan, CarrierPlanError, CyclicPrefixInsert, CyclicPrefixRemove, FftBlock,
    GridExtract, GridMap, IfftBlock, SubcarrierRole, SymbolFft, SymbolWindow,
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
fn symbol_fft_matches_manual_cp_remove_then_fft() {
    // SymbolFft must be bit-identical to the inline CyclicPrefixRemove -> FftBlock
    // sequence it replaces across the RX paths (the R7 refactor's core guarantee).
    let n_fft = 32;
    let cp_len = 8;
    let sps = n_fft + cp_len;
    // A CP'd symbol: distinct complex ramp, then a real cyclic prefix.
    let core: Vec<C32> = (0..n_fft)
        .map(|k| C32::new(k as f32 * 0.25, -(k as f32) * 0.1))
        .collect();
    let mut symbol = vec![C32::default(); sps];
    CyclicPrefixInsert::new(n_fft, cp_len).process(&core, &mut symbol);

    // Reference: the exact inline sequence.
    let mut cp_remove = CyclicPrefixRemove::new(n_fft, cp_len);
    let mut fft = FftBlock::new(n_fft);
    let mut time = vec![C32::default(); n_fft];
    let mut ref_freq = vec![C32::default(); n_fft];
    cp_remove.process(&symbol, &mut time);
    fft.process(&time, &mut ref_freq);

    // Helper under test.
    let mut sf = SymbolFft::new(n_fft, cp_len);
    assert_eq!(sf.symbol_len(), sps);
    assert_eq!(sf.n_fft(), n_fft);
    let freq = sf.demod_symbol(&symbol).expect("full symbol demaps");
    assert_eq!(
        freq,
        ref_freq.as_slice(),
        "SymbolFft must match inline CP-remove+FFT"
    );
}

#[test]
fn symbol_fft_partial_input_is_none() {
    let n_fft = 16;
    let cp_len = 4;
    let mut sf = SymbolFft::new(n_fft, cp_len);
    let short = vec![C32::default(); n_fft + cp_len - 1]; // one sample short
    assert!(sf.demod_symbol(&short).is_none());
}

#[test]
fn symbol_fft_default_backoff_is_zero() {
    let n_fft = 16;
    let cp_len = 4;
    let sf = SymbolFft::new(n_fft, cp_len);
    assert_eq!(sf.window_backoff(), 0, "default back-off must be 0");
}

#[test]
fn symbol_fft_backoff_clamps_to_cp_len() {
    let n_fft = 16;
    let cp_len = 4;
    // A back-off larger than the guard is clamped to cp_len (the window cannot
    // start before the symbol's first sample).
    let sf = SymbolFft::new(n_fft, cp_len).with_window_backoff(100);
    assert_eq!(sf.window_backoff(), cp_len);
    let sf2 = SymbolFft::new(n_fft, cp_len).with_window_backoff(3);
    assert_eq!(sf2.window_backoff(), 3);
}

#[test]
fn symbol_fft_backoff_applies_expected_phase_ramp() {
    // For a properly cyclic symbol (CP is a verbatim copy of the tail), backing
    // the window off by `b` reads a circularly-shifted time window. By the FFT
    // shift theorem the spectrum gains a per-bin linear phase
    // exp(-j 2π k b / n_fft) relative to the back-off-0 spectrum, with unchanged
    // magnitudes. This pins the window-slide direction and amount precisely.
    let n_fft = 32;
    let cp_len = 8;
    let b = 3usize;
    let core: Vec<C32> = (0..n_fft)
        .map(|k| C32::new((k as f32 * 0.3).cos(), (k as f32 * 0.17).sin()))
        .collect();
    let mut symbol = vec![C32::default(); n_fft + cp_len];
    CyclicPrefixInsert::new(n_fft, cp_len).process(&core, &mut symbol);

    let mut sf0 = SymbolFft::new(n_fft, cp_len);
    let f0: Vec<C32> = sf0.demod_symbol(&symbol).unwrap().to_vec();

    let mut sfb = SymbolFft::new(n_fft, cp_len).with_window_backoff(b);
    let fb: Vec<C32> = sfb.demod_symbol(&symbol).unwrap().to_vec();

    let eps = 1e-3f32;
    for k in 0..n_fft {
        // magnitudes preserved
        assert!(
            (f0[k].norm() - fb[k].norm()).abs() < eps,
            "bin {k}: magnitude changed by back-off"
        );
        // phase ramp: fb[k] == f0[k] * exp(-j 2π k b / n_fft)
        let theta = -std::f32::consts::TAU * (k as f32) * (b as f32) / n_fft as f32;
        let expected = f0[k] * C32::new(theta.cos(), theta.sin());
        assert!(
            (fb[k] - expected).norm() < eps,
            "bin {k}: back-off phase ramp mismatch: {:?} vs {:?}",
            fb[k],
            expected
        );
    }
}

// ── SymbolWindow (TX raised-cosine edge taper, Piece B2 / R9) ───────────────

#[test]
fn symbol_window_zero_rolloff_is_identity() {
    let sps = 40;
    let input: Vec<C32> = (0..sps).map(|k| C32::new(k as f32, -(k as f32))).collect();
    let mut out = vec![C32::default(); sps];
    let wr = SymbolWindow::new(sps, 0).process(&input, &mut out);
    assert_eq!(wr.out_written, sps);
    assert_eq!(out, input, "roll_off=0 must be identity");
}

#[test]
fn symbol_window_rolloff_clamps_to_half() {
    let sps = 40;
    // A roll-off past half the symbol would make the two ramps overlap.
    let w = SymbolWindow::new(sps, 100);
    assert_eq!(w.roll_off(), sps / 2);
}

#[test]
fn symbol_window_tapers_edges_leaves_interior() {
    let sps = 40;
    let l = 6;
    let ones = vec![C32::new(1.0, 0.0); sps];
    let mut out = vec![C32::default(); sps];
    SymbolWindow::new(sps, l).process(&ones, &mut out);

    // Interior (flat region) untouched.
    for s in out.iter().take(sps - l).skip(l) {
        assert!((s.re - 1.0).abs() < 1e-6 && s.im.abs() < 1e-6);
    }
    // Edges strictly attenuated and symmetric (leading i mirrors trailing).
    for i in 0..l {
        let lead = out[i].norm();
        let trail = out[sps - 1 - i].norm();
        assert!(lead < 1.0, "leading edge {i} not attenuated: {lead}");
        assert!((lead - trail).abs() < 1e-6, "edge {i} not symmetric");
        // Monotone rise toward the interior.
        if i > 0 {
            assert!(
                out[i].norm() > out[i - 1].norm(),
                "ramp not monotone at {i}"
            );
        }
    }
    // The very outermost samples are the most attenuated (near zero).
    assert!(out[0].norm() < 0.1, "symbol edge should be near-zero");
    assert!(out[sps - 1].norm() < 0.1);
}

#[test]
fn symbol_window_is_rx_transparent_at_half_cp_backoff() {
    // The load-bearing B2 property: a TX taper of L = cp_len/2 with the RX
    // window backed off by b = cp_len/2 leaves the receiver's n_fft-sample core
    // BIT-IDENTICAL to the unwindowed symbol's core (up to f32 rounding), because
    // both ramps fall entirely in guard samples the RX at that back-off discards.
    let n_fft = 64;
    let cp_len = 16;
    let sps = n_fft + cp_len;
    let b = cp_len / 2; // 8
    let l = cp_len / 2; // 8  == min(cp_len - b, b)

    // A realistic CP'd symbol (IFFT of a data grid, then CP).
    let core: Vec<C32> = (0..n_fft)
        .map(|k| C32::new((k as f32 * 0.21).cos(), (k as f32 * 0.13).sin()))
        .collect();
    let mut symbol = vec![C32::default(); sps];
    CyclicPrefixInsert::new(n_fft, cp_len).process(&core, &mut symbol);

    // Windowed copy.
    let mut windowed = vec![C32::default(); sps];
    SymbolWindow::new(sps, l).process(&symbol, &mut windowed);

    // RX at back-off b reads symbol[cp_len - b .. cp_len - b + n_fft].
    let mut sf = SymbolFft::new(n_fft, cp_len).with_window_backoff(b);
    let core_plain: Vec<C32> = sf.demod_symbol(&symbol).unwrap().to_vec();
    let core_windowed: Vec<C32> = sf.demod_symbol(&windowed).unwrap().to_vec();

    // The FFT'd cores must match — the taper touched only discarded guard.
    let eps = 1e-5f32;
    for (a, b) in core_plain.iter().zip(core_windowed.iter()) {
        assert!(
            (a - b).norm() < eps,
            "RX core changed by TX taper: {a:?} vs {b:?}"
        );
    }
}

#[test]
fn symbol_window_time_window_leaves_rx_range_untouched() {
    // Directly (time domain): at b = cp_len/2, L = cp_len/2, the taper must not
    // touch any sample the RX reads, i.e. indices [cp_len - b, cp_len - b + n_fft).
    let n_fft = 32;
    let cp_len = 12;
    let sps = n_fft + cp_len;
    let b = cp_len / 2; // 6
    let l = cp_len / 2; // 6
    let win_start = cp_len - b; // 6
    let win_end = win_start + n_fft; // 38

    let input: Vec<C32> = (0..sps).map(|k| C32::new(1.0 + k as f32, 2.0)).collect();
    let mut out = vec![C32::default(); sps];
    SymbolWindow::new(sps, l).process(&input, &mut out);

    // Every sample inside the RX window is unchanged; only outer guard changes.
    for i in win_start..win_end {
        assert!(
            (out[i] - input[i]).norm() < 1e-6,
            "RX-window sample {i} was modified by the taper"
        );
    }
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

// ── CarrierPlan::with_contiguous_data / validate_edge_guard (edge guard) ────

// For n_fft, the full contiguous span (guard 0, DC excluded) is every
// representable index except the Nyquist bin -(n_fft/2) and DC.
fn full_span_len(n_fft: usize) -> usize {
    let n = n_fft as i32;
    let (lo, hi) = (-(n / 2), (n - 1) / 2);
    // count lo..=hi, drop the Nyquist bin (lo) and DC
    ((hi - lo + 1) - 2) as usize
}

#[test]
fn contiguous_guard_zero_reproduces_full_span() {
    let n_fft = 64;
    let plan = CarrierPlan::new(n_fft, 16).with_contiguous_data(0, false);
    assert_eq!(plan.data_carriers().len(), full_span_len(n_fft));
    // Nyquist bin and DC absent; every other index present.
    assert!(!plan.data_carriers().contains(&-(n_fft as i32 / 2)));
    assert!(!plan.data_carriers().contains(&0));
    assert!(plan.data_carriers().contains(&(n_fft as i32 / 2 - 1)));
    assert!(plan.data_carriers().contains(&-(n_fft as i32 / 2 - 1)));
    plan.validate().expect("full-span plan must validate");
}

#[test]
fn contiguous_guard_k_drops_2k_carriers() {
    let n_fft = 64;
    for k in 0..=4 {
        let plan = CarrierPlan::new(n_fft, 16).with_contiguous_data(k, false);
        // full span minus k carriers per edge
        let expected = full_span_len(n_fft) - 2 * k;
        assert_eq!(
            plan.data_carriers().len(),
            expected,
            "edge_guard={k} should drop 2k carriers"
        );
        // outermost k usable indices at each edge are gone (the low edge
        // is measured from lo+1, since the Nyquist bin lo is never filled)
        let (lo, hi) = plan.index_bounds();
        for g in 0..k as i32 {
            assert!(!plan.data_carriers().contains(&(lo + 1 + g)));
            assert!(!plan.data_carriers().contains(&(hi - g)));
        }
        plan.validate().expect("guarded plan must validate");
        plan.validate_edge_guard(k)
            .expect("guarded plan must honor its own guard");
    }
}

#[test]
fn contiguous_include_dc_toggles_dc() {
    let n_fft = 64;
    let without = CarrierPlan::new(n_fft, 16).with_contiguous_data(2, false);
    let with = CarrierPlan::new(n_fft, 16).with_contiguous_data(2, true);
    assert!(!without.data_carriers().contains(&0));
    assert!(with.data_carriers().contains(&0));
    assert_eq!(
        with.data_carriers().len(),
        without.data_carriers().len() + 1
    );
}

#[test]
fn contiguous_indices_in_range_and_unique() {
    let n_fft = 128;
    let plan = CarrierPlan::new(n_fft, 32).with_contiguous_data(3, false);
    let (lo, hi) = plan.index_bounds();
    let mut seen = std::collections::HashSet::new();
    for &idx in plan.data_carriers() {
        assert!(idx >= lo && idx <= hi, "index {idx} out of range");
        assert!(seen.insert(idx), "duplicate index {idx}");
    }
    plan.validate().expect("plan must validate");
}

#[test]
fn contiguous_data_excludes_pilots_and_composes() {
    let n_fft = 64;
    // Pilots at a few interior indices; the data fill must skip them.
    let pilots = [
        (-10, C32::new(1.0, 0.0)),
        (7, C32::new(1.0, 0.0)),
        (20, C32::new(1.0, 0.0)),
    ];
    let plan = CarrierPlan::new(n_fft, 16)
        .with_pilot_carriers(pilots)
        .with_contiguous_data(2, false);

    // No pilot index appears in the data list.
    for &(pidx, _) in &pilots {
        assert!(
            !plan.data_carriers().contains(&pidx),
            "data fill must exclude pilot index {pidx}"
        );
    }
    // Data count = full guarded span minus DC minus the 3 pilots (all interior).
    let full_guarded = full_span_len(n_fft) - 2 * 2;
    assert_eq!(plan.data_carriers().len(), full_guarded - pilots.len());
    // The whole plan validates with no data/pilot Overlap.
    plan.validate()
        .expect("pilots + edge guard must compose without overlap");
}

#[test]
fn validate_edge_guard_rejects_intruding_index() {
    let n_fft = 64;
    let (lo, _) = CarrierPlan::new(n_fft, 16).index_bounds();
    // A data carrier sitting in the outer guard band.
    let plan = CarrierPlan::new(n_fft, 16).with_data_carriers([lo + 1, 0i32, 5]);
    // Plain validate passes (index is in range, no overlap)...
    plan.validate().expect("plain validate should pass");
    // ...but validate_edge_guard(4) rejects the lo+1 intruder.
    match plan.validate_edge_guard(4) {
        Err(CarrierPlanError::InGuardBand(idx, g)) => {
            assert_eq!(idx, lo + 1);
            assert_eq!(g, 4);
        }
        other => panic!("expected InGuardBand, got {other:?}"),
    }
}

#[test]
fn validate_edge_guard_accepts_interior_only() {
    let n_fft = 64;
    let plan = CarrierPlan::new(n_fft, 16).with_contiguous_data(4, false);
    plan.validate_edge_guard(4)
        .expect("a plan built with guard g must pass validate_edge_guard(g)");
}
