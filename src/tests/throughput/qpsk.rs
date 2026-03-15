
use crate::core::{Block, IqToIqChain};
use crate::modulate::{QpskMapper, QpskMod};
use crate::demodulate::{QpskDemod, QpskDecider};
use std::hint::black_box;
use num_complex::Complex32 as C32;
use super::{minsps_from_env, measure_throughput};

#[test]
fn throughput_qpsk_roundtrip() {
    let n_syms = 65_536;
    let n_bits = n_syms * 2;
    let repeats = 30;

    let bits_in: Vec<u8> = (0..n_bits).map(|i| ((i / 2 + i) & 1) as u8).collect();
    let mut syms     = vec![C32::default(); n_syms];
    let mut soft     = vec![C32::default(); n_syms];
    let mut bits_out = vec![0u8; n_bits];

    let mut mapper   = QpskMapper::new();
    let mut modstage = IqToIqChain::new(QpskMod::new(1.0, 0.0, 1.0));
    let mut demod    = IqToIqChain::new(QpskDemod::new(1.0));
    let mut decider  = QpskDecider::new();

    let (msps, dt) = measure_throughput(
        || {
            mapper.process(&bits_in, &mut syms);
            let iq = modstage.process_ref(&syms);
            demod.process_into(&iq, &mut soft);
            decider.process(&soft, &mut bits_out);
            black_box(bits_out[0]);
            n_syms
        },
        n_syms,
        repeats,
    );

    println!("[QPSK] {:.2} Msps in {:.3}s", msps, dt);
    let min_msps = minsps_from_env(0.2);
    assert!(msps >= min_msps, "QPSK throughput {:.2} Msps < min {:.2} Msps", msps, min_msps);
}
