
use crate::modulate::Qam16Mapper;
use crate::core::Block;
use num_complex::Complex32 as C32;

#[test]
fn qam16_mapper_symbols() {
    let scale = (1.0f32 / 10.0f32).sqrt();
    let bits: Vec<u8> = vec![
        0,0, 0,0,
        0,1, 0,0,
        1,1, 0,0,
        1,0, 0,0,
    ];
    let mut out = [C32::default(); 4];
    Qam16Mapper::new().process(&bits, &mut out);
    let eps = 1e-6f32;
    assert!((out[0].re - (-3.0 * scale)).abs() < eps, "QAM-16 I level mismatch sym 0: {:?}", out[0]);
    assert!((out[1].re - (-1.0 * scale)).abs() < eps, "QAM-16 I level mismatch sym 1: {:?}", out[1]);
    assert!((out[2].re - ( 1.0 * scale)).abs() < eps, "QAM-16 I level mismatch sym 2: {:?}", out[2]);
    assert!((out[3].re - ( 3.0 * scale)).abs() < eps, "QAM-16 I level mismatch sym 3: {:?}", out[3]);
    for s in &out { assert!((s.im - (-3.0 * scale)).abs() < eps, "QAM-16 Q level mismatch: {:?}", s); }
}
