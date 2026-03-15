
use crate::modulate::BpskMapper;
use crate::core::Block;
use num_complex::Complex32 as C32;

#[test]
fn bpsk_mapper_symbols() {
    let bits = [0u8, 1, 0, 1, 1, 0];
    let mut out = [C32::new(0.0, 0.0); 6];
    BpskMapper::new().process(&bits, &mut out);
    assert_eq!(out[0], C32::new( 1.0, 0.0));
    assert_eq!(out[1], C32::new(-1.0, 0.0));
    assert_eq!(out[2], C32::new( 1.0, 0.0));
    assert_eq!(out[3], C32::new(-1.0, 0.0));
    assert_eq!(out[4], C32::new(-1.0, 0.0));
    assert_eq!(out[5], C32::new( 1.0, 0.0));
}
