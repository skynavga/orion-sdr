
pub fn real_tone(fs: f32, f_hz: f32, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|k| amp * (std::f32::consts::TAU * f_hz * (k as f32) / fs).sin())
        .collect()
}

pub fn snr_db_at(fs: f32, f0: f32, x: &[f32]) -> f32 {
    let n = x.len().max(1);
    let proj = |f: f32| {
        let w = -std::f32::consts::TAU * f / fs;
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (k, &s) in x.iter().enumerate() {
            let t = w * (k as f32);
            re += s * t.cos();
            im += s * t.sin();
        }
        (re * re + im * im) / ((n as f32) * (n as f32))
    };
    let p_sig = proj(f0);
    let p_off = proj(f0 * 0.73) + 1e-20;
    10.0 * (p_sig / p_off).log10()
}

pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() { return 0.0; }
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    (sum_sq / x.len() as f32).sqrt()
}

pub fn tail<T>(x: &[T]) -> &[T] {
    &x[x.len() / 4..]
}

pub fn add_awgn(iq: &mut Vec<num_complex::Complex32>, noise_power: f32, seed: u64) {
    let mut state = seed ^ 0xDEAD_BEEF_CAFE_0000;
    let scale = (noise_power / 2.0).sqrt();

    let next_f32 = |s: &mut u64| -> f32 {
        let mut sum = 0.0f32;
        for _ in 0..12 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            sum += (*s as f32) / (u64::MAX as f32) - 0.5;
        }
        sum
    };

    for sample in iq.iter_mut() {
        let n_i = next_f32(&mut state) * scale;
        let n_q = next_f32(&mut state) * scale;
        sample.re += n_i;
        sample.im += n_q;
    }
}

pub fn make_ft8_test_buffer(
    time_offset_samples: usize,
    base_hz: f32,
    noise_power: f32,
) -> (Vec<num_complex::Complex32>, crate::codec::ft8::Ft8Bits) {
    use crate::codec::ft8::{Ft8Codec, Ft8Bits};
    use crate::modulate::Ft8Mod;
    use crate::modulate::ft8::FT8_FRAME_LEN;

    let payload: Ft8Bits = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60];
    let frame = Ft8Codec::encode(&payload);
    let iq_frame = Ft8Mod::new(12_000.0, base_hz, 0.0, 1.0).modulate(&frame);

    let total_len = time_offset_samples + FT8_FRAME_LEN + FT8_FRAME_LEN / 4;
    let mut buf = vec![num_complex::Complex32::new(0.0, 0.0); total_len];

    for (i, s) in iq_frame.iter().enumerate() {
        buf[time_offset_samples + i] = *s;
    }

    if noise_power > 0.0 {
        add_awgn(&mut buf, noise_power, 0x1234_5678_ABCD_EF00);
    }

    (buf, payload)
}

pub fn make_ft4_test_buffer(
    time_offset_samples: usize,
    base_hz: f32,
    noise_power: f32,
) -> (Vec<num_complex::Complex32>, crate::codec::ft4::Ft4Bits) {
    use crate::codec::ft4::{Ft4Codec, Ft4Bits};
    use crate::modulate::Ft4Mod;
    use crate::modulate::ft4::FT4_FRAME_LEN;

    let payload: Ft4Bits = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60];
    let frame = Ft4Codec::encode(&payload);
    let iq_frame = Ft4Mod::new(12_000.0, base_hz, 0.0, 1.0).modulate(&frame);

    let total_len = time_offset_samples + FT4_FRAME_LEN + FT4_FRAME_LEN / 4;
    let mut buf = vec![num_complex::Complex32::new(0.0, 0.0); total_len];

    for (i, s) in iq_frame.iter().enumerate() {
        buf[time_offset_samples + i] = *s;
    }

    if noise_power > 0.0 {
        add_awgn(&mut buf, noise_power, 0x9876_5432_FEDC_BA98);
    }

    (buf, payload)
}

pub mod am;
pub mod bpsk;
pub mod cw;
pub mod fm;
pub mod ft4;
pub mod ft8;
pub mod ft8_snr;
pub mod message;
pub mod pm;
pub mod psk31;
pub mod qam;
pub mod qpsk;
pub mod ssb;
