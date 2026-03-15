
use crate::demodulate::FmQuadratureDemod;
use crate::core::Block;
use super::snr_db_at;
use num_complex::Complex32 as C32;

#[test]
fn fm_quadrature_demod_recovers_tone() {
    let fs = 48_000.0;
    let n = 16_384;
    let f_mod = 1_000.0;
    let dev = 2_500.0;
    let mut phi = 0.0f32;
    let mut iq = Vec::with_capacity(n);
    for k in 0..n {
        let t = k as f32 / fs;
        let f_inst = dev * (2.0*std::f32::consts::PI * f_mod * t).sin();
        phi += 2.0*std::f32::consts::PI * f_inst / fs;
        iq.push(C32::new(phi.cos(), phi.sin()));
    }
    let mut dem = FmQuadratureDemod::new(fs, dev, 5_000.0);
    let mut y = vec![0.0f32; n];
    let _ = dem.process(&iq, &mut y);
    let snr = snr_db_at(f_mod, fs, &y);
    assert!(snr > 20.0, "FM SNR too low: {:.1} dB", snr);
}
