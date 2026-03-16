
use crate::core::{Block, IqToIqChain};
use crate::modulate::{BpskMapper, BpskMod};
use crate::demodulate::{BpskDemod, BpskDecider};
use std::hint::black_box;
use num_complex::Complex32 as C32;
use super::{minsps_from_env, measure_throughput};

#[test]
fn throughput_bpsk_roundtrip() {
    let n = 65_536;
    let repeats = 30;

    let bits_in: Vec<u8> = (0..n).map(|i| (i & 1) as u8).collect();
    let mut syms   = vec![C32::default(); n];
    let mut soft   = vec![C32::default(); n];
    let mut bits_out = vec![0u8; n];

    let mut mapper  = BpskMapper::new();
    let mut modstage = IqToIqChain::new(BpskMod::new(1.0, 0.0, 1.0));
    let mut demod   = IqToIqChain::new(BpskDemod::new(1.0));
    let mut decider = BpskDecider::new();

    let (msps, dt) = measure_throughput(
        || {
            mapper.process(&bits_in, &mut syms);
            let iq = modstage.process_ref(&syms);
            demod.process_into(&iq, &mut soft);
            decider.process(&soft, &mut bits_out);
            black_box(bits_out[0]);
            n
        },
        n,
        repeats,
    );

    println!("[BPSK] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.2);
    assert!(msps >= min_msps, "BPSK throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}
