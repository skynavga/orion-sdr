
use crate::modulate::{Ft4Mod, Ft4Frame};
use crate::modulate::ft4::{FT4_FRAME_LEN, FT4_TOTAL_SYMS, FT4_DATA_SYMS};

#[test]
fn ft4_frame_length() {
    let tx = Ft4Mod::new(12_000.0, 1_000.0, 0.0, 1.0);
    let iq = tx.modulate(&Ft4Frame::zeros());
    assert_eq!(iq.len(), FT4_FRAME_LEN, "FT4 frame length mismatch");
}

#[test]
fn ft4_symbol_sequence_count() {
    let seq = Ft4Mod::build_symbol_sequence(&Ft4Frame::zeros());
    assert_eq!(seq.len(), FT4_TOTAL_SYMS);
    let sync_pos: [(usize, usize); 4] = [(1, 5), (34, 38), (67, 71), (100, 104)];
    let mut is_reserved = [false; FT4_TOTAL_SYMS];
    is_reserved[0] = true;
    is_reserved[104] = true;
    for &(start, end) in &sync_pos { for p in start..end { is_reserved[p] = true; } }
    let data_count = is_reserved.iter().filter(|&&s| !s).count();
    assert_eq!(data_count, FT4_DATA_SYMS);
}

#[test]
fn ft4_costas_positions_correct() {
    let costas: [[u8; 4]; 4] = [[0,1,3,2],[1,0,2,3],[2,3,1,0],[3,2,0,1]];
    let sync_starts = [1usize, 34, 67, 100];
    let seq = Ft4Mod::build_symbol_sequence(&Ft4Frame::zeros());
    assert_eq!(seq[0], 0, "FT4 ramp at position 0 should be tone 0");
    assert_eq!(seq[104], 0, "FT4 ramp at position 104 should be tone 0");
    for (blk, &start) in sync_starts.iter().enumerate() {
        for i in 0..4 {
            assert_eq!(seq[start + i], costas[blk][i],
                "FT4 Costas mismatch blk {} sym {}: got {}, expected {}",
                blk, i, seq[start + i], costas[blk][i]);
        }
    }
}

#[test]
fn ft4_iq_power_unity() {
    let tx = Ft4Mod::new(12_000.0, 1_000.0, 0.0, 1.0);
    let iq = tx.modulate(&Ft4Frame::zeros());
    let power: f32 = iq.iter().map(|z| z.norm_sqr()).sum::<f32>() / (FT4_FRAME_LEN as f32);
    assert!((power - 1.0).abs() < 0.01, "FT4 IQ power deviates from 1.0: {}", power);
}
