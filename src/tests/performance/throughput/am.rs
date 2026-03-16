
use crate::core::{AudioToIqChain, IqToAudioChain};
use crate::modulate::AmDsbMod;
use crate::demodulate::AmEnvelopeDemod;
use super::{real_tone, minsps_from_env, measure_throughput};

#[test]
fn throughput_am_powersqrt_roundtrip() {
    let fs = 48_000.0;
    let n = 65_536;
    let repeats = 30;

    let audio = real_tone(fs, 1000.0, n, 0.5);
    let mut tx = AudioToIqChain::new(AmDsbMod::new(fs, 0.0, 0.8, 0.5));
    let mut rx = IqToAudioChain::new(AmEnvelopeDemod::new(fs, 5_000.0));

    let (msps, dt) = measure_throughput(
        || {
            let iq = tx.process(audio.clone());
            let out = rx.process(iq);
            out.len()
        },
        n,
        repeats,
    );

    println!("[AM](PowerSqrt) {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.2);
    assert!(msps >= min_msps, "AM-PowerSqrt throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}

#[test]
fn throughput_am_absapprox_roundtrip() {
    let fs = 48_000.0;
    let n = 65_536;
    let repeats = 30;

    let audio = real_tone(fs, 1000.0, n, 0.5);
    let mut tx = AudioToIqChain::new(AmDsbMod::new(fs, 0.0, 0.8, 0.5));
    let mut rx = IqToAudioChain::new(
        AmEnvelopeDemod::new(fs, 5_000.0).with_abs_approx(0.9475, 0.3925)
    );

    let (msps, dt) = measure_throughput(
        || {
            let iq = tx.process(audio.clone());
            let out = rx.process(iq);
            out.len()
        },
        n,
        repeats,
    );

    println!("[AM](AbsApprox) {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.2);
    assert!(msps >= min_msps,"AM-AbsApprox throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}
