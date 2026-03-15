
use crate::core::{Block, IqToIqChain};
use crate::modulate::{QamMapper, QamMod};
use crate::demodulate::{QamDemod, QamDecider};
use std::hint::black_box;
use num_complex::Complex32 as C32;
use super::{minsps_from_env, measure_throughput};

fn throughput_qam_roundtrip<const BITS: usize>(label: &str) {
    let n_syms = 65_536;
    let n_bits = n_syms * BITS;
    let repeats = 30;

    let bits_in: Vec<u8> = (0..n_bits).map(|i| ((i / BITS + i % BITS) & 1) as u8).collect();
    let mut syms     = vec![C32::default(); n_syms];
    let mut soft     = vec![C32::default(); n_syms];
    let mut bits_out = vec![0u8; n_bits];

    let mut mapper   = QamMapper::<BITS>::new();
    let mut modstage = IqToIqChain::new(QamMod::new(1.0, 0.0, 1.0));
    let mut demod    = IqToIqChain::new(QamDemod::new(1.0));
    let mut decider  = QamDecider::<BITS>::new();

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

    println!("[{}] {:.2} Msps in {:.3}s", label, msps, dt);
    let min_msps = minsps_from_env(0.2);
    assert!(msps >= min_msps, "{} throughput {:.2} Msps < min {:.2} Msps", label, msps, min_msps);
}

#[test]
fn throughput_qam16_roundtrip()  { throughput_qam_roundtrip::<4>("QAM-16");  }

#[test]
fn throughput_qam64_roundtrip()  { throughput_qam_roundtrip::<6>("QAM-64");  }

#[test]
fn throughput_qam256_roundtrip() { throughput_qam_roundtrip::<8>("QAM-256"); }
