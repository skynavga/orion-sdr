
use crate::modulate::{QpskMapper, QpskMod};
use crate::demodulate::{QpskDemod, QpskDecider};
use crate::core::Block;
use num_complex::Complex32 as C32;

#[test]
fn roundtrip_qpsk_noiseless() {
    let n_syms = 256;
    let bits_in: Vec<u8> = (0..n_syms * 2).map(|i| ((i / 2 + i) & 1) as u8).collect();
    let mut syms     = vec![C32::default(); n_syms];
    let mut iq       = vec![C32::default(); n_syms];
    let mut soft     = vec![C32::default(); n_syms];
    let mut bits_out = vec![0u8; n_syms * 2];
    QpskMapper::new().process(&bits_in, &mut syms);
    QpskMod::new(1.0, 0.0, 1.0).process(&syms, &mut iq);
    QpskDemod::new(1.0).process(&iq, &mut soft);
    QpskDecider::new().process(&soft, &mut bits_out);
    assert_eq!(bits_in, bits_out);
}
