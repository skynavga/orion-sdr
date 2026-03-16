
use crate::core::{AudioToIqChain, IqToAudioChain};
use crate::modulate::FmPhaseAccumMod;
use crate::demodulate::FmQuadratureDemod;
use super::{real_tone, minsps_from_env, measure_throughput};

#[test]
fn throughput_fm_roundtrip() {
    let fs = 48_000.0;
    let n = 65_536;
    let repeats = 30;

    let audio = real_tone(fs, 1000.0, n, 0.5);
    let mut tx = AudioToIqChain::new(FmPhaseAccumMod::new(fs, 2_500.0, 0.0));
    let mut rx = IqToAudioChain::new(FmQuadratureDemod::new(fs, 2_500.0, 5_000.0));

    let (msps, dt) = measure_throughput(
        || {
            let iq = tx.process(audio.clone());
            let out = rx.process(iq);
            out.len()
        },
        n,
        repeats,
    );

    println!("[FM] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.25);
    assert!(msps >= min_msps, "FM throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}
