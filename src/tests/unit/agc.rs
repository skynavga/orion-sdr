
use crate::dsp::AgcRmsIq;
use crate::core::Block;
use num_complex::Complex32 as C32;

#[test]
fn agc_rms_converges_on_iq() {
    let fs = 48_000.0;
    let mut agc = AgcRmsIq::new(fs, 0.2, 5.0, 0.2);
    let n = 8_000;
    let mut input = Vec::with_capacity(n);
    for k in 0..n {
        let a = if k < n/2 { 0.02 } else { 1.0 };
        input.push(C32::new(a, 0.0));
    }
    let mut out = vec![C32::new(0.0, 0.0); n];
    let _ = agc.process(&input, &mut out);
    let tail_len = 1000.min(n/2);
    let tail = &out[n - tail_len..];
    let mut acc = 0.0f32;
    for z in tail { acc += z.norm_sqr(); }
    let rms_tail = (acc / (tail_len as f32)).sqrt();
    assert!((rms_tail - 0.2).abs() < 0.03, "tail RMS={} not near target 0.2", rms_tail);
}
