
use crate::core::{AudioToIqChain, IqToAudioChain};
use crate::modulate::CwKeyedMod;
use crate::demodulate::CwEnvelopeDemod;
use num_complex::Complex32 as C32;
use super::{key_envelope_square, minsps_from_env, measure_throughput};

#[test]
fn throughput_cw_roundtrip() {
    let fs = 48_000.0;
    let n = 65_536;
    let repeats = 30;

    let key = key_envelope_square(fs, 5.0, n);
    let mut tx = AudioToIqChain::new(CwKeyedMod::new(fs, 700.0, 3.0, 3.0));
    let mut rx = IqToAudioChain::new(CwEnvelopeDemod::new(fs, 700.0, 300.0));

    let (msps, dt) = measure_throughput(
        || {
            let iq: Vec<C32> = tx.process(key.clone());
            let _audio: Vec<f32> = rx.process(iq);
            _audio.len()
        },
        n,
        repeats,
    );

    println!("[CW] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.2);
    assert!(msps >= min_msps, "CW throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}
