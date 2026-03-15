
use crate::core::IqToAudioChain;
use crate::dsp::{Nco, mix_with_nco};
use crate::demodulate::{CwEnvelopeDemod, AmEnvelopeDemod, SsbProductDemod};
use num_complex::Complex32 as C32;

#[test]
fn chain_runs_ssb_cw_and_am() {
    let fs = 48_000.0;
    let n = 4096;
    let mut tone = Vec::with_capacity(n);
    let mut nco = Nco::new(1_000.0, fs);
    for _ in 0..n {
        tone.push(mix_with_nco(C32::new(1.0,0.0), &mut nco));
    }

    // CW chain
    let mut cw = IqToAudioChain::new(CwEnvelopeDemod::new(fs, 700.0, 300.0));
    let y_cw = cw.process(tone.clone());
    assert_eq!(y_cw.len(), n);

    // AM chain
    let mut am = IqToAudioChain::new(AmEnvelopeDemod::new(fs, 5_000.0));
    let y_am = am.process(tone.clone());
    assert_eq!(y_am.len(), n);

    // SSB chain
    let mut ssb = IqToAudioChain::new(SsbProductDemod::new(fs, 0.0, 2_800.0));
    let y_ssb = ssb.process(tone);
    assert_eq!(y_ssb.len(), n);
}
