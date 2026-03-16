
use crate::modulate::{Ft8Mod, Ft8Frame};
use crate::modulate::ft8::{FT8_FRAME_LEN, FT8_TOTAL_SYMS, FT8_DATA_SYMS};

#[test]
fn ft8_frame_length() {
    let tx = Ft8Mod::new(12_000.0, 1_000.0, 0.0, 1.0);
    let iq = tx.modulate(&Ft8Frame::zeros());
    assert_eq!(iq.len(), FT8_FRAME_LEN, "FT8 frame length mismatch");
}

#[test]
fn ft8_symbol_sequence_count() {
    let seq = Ft8Mod::build_symbol_sequence(&Ft8Frame::zeros());
    assert_eq!(seq.len(), FT8_TOTAL_SYMS);
    let sync_pos: [(usize, usize); 3] = [(0, 7), (36, 43), (72, 79)];
    let mut is_sync = [false; FT8_TOTAL_SYMS];
    for &(start, end) in &sync_pos { for p in start..end { is_sync[p] = true; } }
    let data_count = is_sync.iter().filter(|&&s| !s).count();
    assert_eq!(data_count, FT8_DATA_SYMS);
}

#[test]
fn ft8_costas_positions_correct() {
    let costas = [3u8, 1, 4, 0, 6, 5, 2];
    let sync_starts = [0usize, 36, 72];
    let seq = Ft8Mod::build_symbol_sequence(&Ft8Frame::zeros());
    for &start in &sync_starts {
        for i in 0..7 {
            assert_eq!(seq[start + i], costas[i],
                "FT8 Costas mismatch at sym {}: got {}, expected {}",
                start + i, seq[start + i], costas[i]);
        }
    }
}

#[test]
fn ft8_iq_power_unity() {
    let tx = Ft8Mod::new(12_000.0, 1_000.0, 0.0, 1.0);
    let iq = tx.modulate(&Ft8Frame::zeros());
    let power: f32 = iq.iter().map(|z| z.norm_sqr()).sum::<f32>() / (FT8_FRAME_LEN as f32);
    assert!((power - 1.0).abs() < 0.01, "FT8 IQ power deviates from 1.0: {}", power);
}
