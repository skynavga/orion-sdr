// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{measure_throughput, minsps_from_env, real_tone};
use orion_sdr::core::{AudioToIqChain, IqToAudioChain};
use orion_sdr::demodulate::SsbProductDemod;
use orion_sdr::modulate::SsbPhasingMod;

#[test]
fn throughput_ssb_usb_roundtrip() {
    let fs = 48_000.0;
    let n = 65_536;
    let repeats = 20;

    let audio = real_tone(fs, 1200.0, n, 0.4);
    let mut tx = AudioToIqChain::new(SsbPhasingMod::new(fs, 2_800.0, 1_500.0, 0.0, true));
    let mut rx = IqToAudioChain::new(SsbProductDemod::new(fs, 0.0, 2_800.0));

    let (msps, dt) = measure_throughput(
        || {
            let iq = tx.process(audio.clone());
            let out = rx.process(iq);
            out.len()
        },
        n,
        repeats,
    );

    println!("[SSB-USB] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.15);
    assert!(
        msps >= min_msps,
        "SSB throughput {:.2} Msps < min {:.2} Msps",
        msps,
        min_msps
    );
}
