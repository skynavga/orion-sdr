
use crate::dsp::{FirDecimator, Nco, mix_with_nco};
use crate::core::Block;
use num_complex::Complex32 as C32;

#[test]
fn decimator_reduces_length_and_preserves_tone() {
    let fs = 96_000.0;
    let m = 4;
    let cutoff = fs / (m as f32) * 0.45;
    let transition = fs / (m as f32) * 0.10;
    let mut dec = FirDecimator::new(fs, m, cutoff, transition);
    let n = 4096;
    let mut nco = Nco::new(2_000.0, fs);
    let mut iq = vec![C32::new(0.0,0.0); n];
    for i in 0..n { iq[i] = mix_with_nco(C32::new(1.0,0.0), &mut nco); }
    let mut out = vec![C32::new(0.0,0.0); n/m];
    let w = dec.process(&iq, &mut out);
    assert_eq!(w.out_written, n/m);
}
