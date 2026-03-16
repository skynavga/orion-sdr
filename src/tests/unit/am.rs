
use crate::core::AudioToIqChain;
use crate::modulate::AmDsbMod;
use crate::util::tone;

#[test]
fn audio_to_iq_chain_runs_and_lengths_match() {
    let fs = 48_000.0;
    let n = 4096;
    let audio = tone(fs, 1000.0, n, 0.5);

    let am = AmDsbMod::new(fs, 0.0, 0.8, 0.5);
    let mut chain = AudioToIqChain::new(am);

    let iq = chain.process(audio);
    assert_eq!(iq.len(), n);
    let power: f32 = iq.iter().map(|z| z.norm_sqr()).sum::<f32>() / (n as f32);
    assert!(power > 1e-6, "IQ power too small");
}
