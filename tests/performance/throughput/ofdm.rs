// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{measure_throughput, minsps_from_env};
use orion_sdr::core::Block;
use orion_sdr::modulate::{ConstellationOrder, OfdmConfig, OfdmMod};
use orion_sdr::multicarrier::CarrierPlan;
use std::hint::black_box;

fn throughput_ofdm_mod(constellation: ConstellationOrder, label: &str) {
    let n_fft = 1024;
    let cp_len = 128;
    let half = (n_fft / 2) as i32;
    let data: Vec<i32> = (1..half).chain(-(half - 1)..0).collect();
    let plan = CarrierPlan::new(n_fft, cp_len).with_data_carriers(data);
    let cfg = OfdmConfig::new(plan, 1.0, 0.0, 1.0, constellation);

    let mut modstage = OfdmMod::new(&cfg);
    let bits_per_symbol = cfg.bits_per_ofdm_symbol();
    let samples_per_symbol = cfg.samples_per_ofdm_symbol();

    let n_symbols = 256;
    let repeats = 30;

    let bits_in: Vec<u8> = (0..n_symbols * bits_per_symbol)
        .map(|i| (i & 1) as u8)
        .collect();
    let mut out = vec![num_complex::Complex32::default(); samples_per_symbol];

    let (msps, dt) = measure_throughput(
        || {
            for s in 0..n_symbols {
                let chunk = &bits_in[s * bits_per_symbol..(s + 1) * bits_per_symbol];
                modstage.process(chunk, &mut out);
                black_box(out[0]);
            }
            n_symbols * samples_per_symbol
        },
        n_symbols * samples_per_symbol,
        repeats,
    );

    println!("[{}] {:.2} Msps in {:.3}s", label, msps, dt);
    let min_msps = minsps_from_env(0.2);
    assert!(
        msps >= min_msps,
        "{} throughput {:.2} Msps < min {:.2} Msps",
        label,
        msps,
        min_msps
    );
}

#[test]
fn throughput_ofdm_mod_qpsk() {
    throughput_ofdm_mod(ConstellationOrder::Qpsk, "OFDM-Mod QPSK");
}

#[test]
fn throughput_ofdm_mod_qam64() {
    throughput_ofdm_mod(ConstellationOrder::Qam64, "OFDM-Mod QAM-64");
}
