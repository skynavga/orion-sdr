
use crate::modulate::QpskMapper;
use crate::core::Block;
use num_complex::Complex32 as C32;

#[test]
fn qpsk_mapper_symbols() {
    const S: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let bits = [0u8, 0,  0, 1,  1, 0,  1, 1];
    let mut out = [C32::default(); 4];
    QpskMapper::new().process(&bits, &mut out);
    assert_eq!(out[0], C32::new( S,  S));
    assert_eq!(out[1], C32::new( S, -S));
    assert_eq!(out[2], C32::new(-S,  S));
    assert_eq!(out[3], C32::new(-S, -S));
}
