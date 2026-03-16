
/// Tiny single-bin DFT; good enough for power at a specific frequency.
pub fn dft_power(signal: &[f32], fs: f32, f_hz: f32) -> f32 {
    let n = signal.len();
    let w = -2.0 * std::f32::consts::PI * f_hz / fs;
    let mut re = 0.0f32;
    let mut im = 0.0f32;
    for (k, &x) in signal.iter().enumerate() {
        let t = w * (k as f32);
        re += x * t.cos();
        im += x * t.sin();
    }
    let mag2 = (re * re + im * im) / (n as f32 * n as f32);
    mag2
}

pub fn snr_db_at(freq_hz: f32, fs: f32, x: &[f32]) -> f32 {
    let n = x.len();
    let w = -2.0 * std::f32::consts::PI * freq_hz / fs;
    let (mut re, mut im) = (0.0f32, 0.0f32);
    for (k, &s) in x.iter().enumerate() {
        let t = w * (k as f32);
        re += s * t.cos();
        im += s * t.sin();
    }
    let p_sig = (re*re + im*im) / (n as f32 * n as f32);
    let f2 = freq_hz * 0.7;
    let w2 = -2.0 * std::f32::consts::PI * f2 / fs;
    let (mut r2, mut i2) = (0.0f32, 0.0f32);
    for (k, &s) in x.iter().enumerate() { let t = w2 * (k as f32); r2 += s*t.cos(); i2 += s*t.sin(); }
    let p_off = (r2*r2 + i2*i2) / (n as f32 * n as f32);
    10.0 * (p_sig / (p_off + 1e-20)).log10()
}

pub mod agc;
pub mod am;
pub mod bpsk;
pub mod chains;
pub mod codec;
pub mod dsp;
pub mod fm;
pub mod ft4;
pub mod ft8;
pub mod message;
pub mod pm;
pub mod qam;
pub mod qpsk;
pub mod ssb;
pub mod sync;
