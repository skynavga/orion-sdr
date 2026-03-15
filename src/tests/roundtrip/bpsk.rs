
use crate::modulate::{BpskMapper, BpskMod};
use crate::demodulate::{BpskDemod, BpskDecider};
use crate::core::Block;
use num_complex::Complex32 as C32;

#[test]
fn roundtrip_bpsk_noiseless() {
    let n = 256;
    let bits_in: Vec<u8> = (0..n).map(|i| (i & 1) as u8).collect();
    let mut syms = vec![C32::default(); n];
    let mut iq   = vec![C32::default(); n];
    let mut soft = vec![C32::default(); n];
    let mut bits_out = vec![0u8; n];
    BpskMapper::new().process(&bits_in, &mut syms);
    BpskMod::new(1.0, 0.0, 1.0).process(&syms, &mut iq);
    BpskDemod::new(1.0).process(&iq, &mut soft);
    BpskDecider::new().process(&soft, &mut bits_out);
    assert_eq!(bits_in, bits_out);
}
