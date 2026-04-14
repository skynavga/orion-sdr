// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{measure_throughput, minsps_from_env, real_tone};
use orion_sdr::core::{AudioToIqChain, IqToAudioChain};
use orion_sdr::demodulate::PmQuadratureDemod;
use orion_sdr::modulate::PmDirectPhaseMod;

#[test]
fn throughput_pm_roundtrip() {
    let fs = 48_000.0;
    let n = 65_536;
    let repeats = 30;

    let audio = real_tone(fs, 900.0, n, 0.5);
    let mut tx = AudioToIqChain::new(PmDirectPhaseMod::new(fs, 0.9, 0.0));
    let mut rx = IqToAudioChain::new(PmQuadratureDemod::new(fs, 0.9, 5_000.0));

    let (msps, dt) = measure_throughput(
        || {
            let iq = tx.process(audio.clone());
            let out = rx.process(iq);
            out.len()
        },
        n,
        repeats,
    );

    println!("[PM] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.25);
    assert!(
        msps >= min_msps,
        "PM throughput {:.2} Msps < min {:.2} Msps",
        msps,
        min_msps
    );
}
