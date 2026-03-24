
use num_complex::Complex32 as C32;
use crate::modulate::psk31::{Bpsk31Mod, Qpsk31Mod, psk31_sps};
use crate::demodulate::psk31::{Bpsk31Demod, Bpsk31Decider, Qpsk31Demod, Qpsk31Decider};
use crate::core::Block;
use std::hint::black_box;
use super::{minsps_from_env, measure_throughput};

#[test]
#[cfg(feature = "throughput")]
fn throughput_bpsk31_roundtrip() {
    let fs = 8000.0f32;
    let sps = psk31_sps(fs);
    let n_syms = 4096;
    let n_samples = n_syms * sps;
    let repeats = 20;

    let bits_in: Vec<u8> = (0..n_syms).map(|i| (i & 1) as u8).collect();

    let (msps, dt) = measure_throughput(
        || {
            let mut modulator = Bpsk31Mod::new(fs, 0.0, 1.0);
            let iq = modulator.modulate_bits(&bits_in);
            let mut demod = Bpsk31Demod::new(fs, 0.0, 1.0);
            let mut soft = vec![0.0f32; n_syms + 4];
            let wr = demod.process(&iq, &mut soft);
            let mut bits_out = vec![0u8; wr.out_written];
            Bpsk31Decider::new().process(&soft[..wr.out_written], &mut bits_out);
            black_box(bits_out[0]);
            n_samples
        },
        n_samples,
        repeats,
    );

    println!("[BPSK31] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.5);
    assert!(msps >= min_msps, "BPSK31 throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}

#[test]
#[cfg(feature = "throughput")]
fn throughput_qpsk31_roundtrip() {
    let fs = 8000.0f32;
    let sps = psk31_sps(fs);
    let n_syms = 4096;
    let n_samples = n_syms * sps;
    let repeats = 20;

    let bits_in: Vec<u8> = (0..n_syms).map(|i| (i & 1) as u8).collect();

    let (msps, dt) = measure_throughput(
        || {
            let mut modulator = Qpsk31Mod::new(fs, 0.0, 1.0);
            let iq = modulator.modulate_bits(&bits_in);
            let mut demod = Qpsk31Demod::new(fs, 0.0, 1.0);
            let mut soft = vec![0.0f32; n_syms * 2 + 4];
            let wr = demod.process(&iq, &mut soft);
            let mut decider = Qpsk31Decider::new();
            decider.process(&soft[..wr.out_written], &mut vec![]);
            let mut bits_out = Vec::new();
            decider.flush(&mut bits_out);
            black_box(bits_out.first().copied().unwrap_or(0));
            n_samples
        },
        n_samples,
        repeats,
    );

    println!("[QPSK31] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.3);
    assert!(msps >= min_msps, "QPSK31 throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}
