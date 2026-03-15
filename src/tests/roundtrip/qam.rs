
use crate::modulate::{QamMapper, QamMod};
use crate::demodulate::{QamDemod, QamDecider};
use crate::core::Block;
use num_complex::Complex32 as C32;

fn qam_roundtrip_noiseless<const BITS: usize>(n_syms: usize) {
    let n_bits = n_syms * BITS;
    let bits_in: Vec<u8> = (0..n_bits).map(|i| ((i / BITS + i % BITS) & 1) as u8).collect();
    let mut syms     = vec![C32::default(); n_syms];
    let mut iq       = vec![C32::default(); n_syms];
    let mut soft     = vec![C32::default(); n_syms];
    let mut bits_out = vec![0u8; n_bits];

    QamMapper::<BITS>::new().process(&bits_in, &mut syms);
    QamMod::new(1.0, 0.0, 1.0).process(&syms, &mut iq);
    QamDemod::new(1.0).process(&iq, &mut soft);
    QamDecider::<BITS>::new().process(&soft, &mut bits_out);
    assert_eq!(bits_in, bits_out, "QAM-{} noiseless roundtrip failed", 1 << BITS);
}

#[test]
fn roundtrip_qam16_noiseless() { qam_roundtrip_noiseless::<4>(256); }

#[test]
fn roundtrip_qam64_noiseless() { qam_roundtrip_noiseless::<6>(256); }

#[test]
fn roundtrip_qam256_noiseless() { qam_roundtrip_noiseless::<8>(256); }
