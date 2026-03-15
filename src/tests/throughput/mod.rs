
use std::time::Instant;
use std::hint::black_box;

pub fn real_tone(fs: f32, f_hz: f32, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|k| amp * (std::f32::consts::TAU * f_hz * (k as f32) / fs).sin())
        .collect()
}

pub fn key_envelope_square(fs: f32, key_hz: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| ((k as f32 * key_hz / fs).fract() < 0.5) as i32 as f32)
        .collect()
}

pub fn minsps_from_env(default_msps: f32) -> f32 {
    std::env::var("ORION_SDR_THROUGHPUT_MINSPS")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(default_msps)
}

pub fn measure_throughput(mut f: impl FnMut() -> usize, samples_per_pass: usize, repeats: usize) -> (f32, f64) {
    let start = Instant::now();
    let mut sink = 0.0f64;
    for _ in 0..repeats {
        let n = f();
        sink = black_box(sink + (n as f64) * 1e-12);
    }
    let dt = start.elapsed().as_secs_f64();
    let total_samples = samples_per_pass as f64 * repeats as f64;
    let msps = (total_samples / dt) / 1.0e6;
    black_box(sink);
    (msps as f32, dt)
}

pub mod am;
pub mod bpsk;
pub mod cw;
pub mod fm;
pub mod pm;
pub mod qam;
pub mod qpsk;
pub mod ssb;
