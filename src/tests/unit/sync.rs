
use num_complex::Complex32 as C32;

// --- Waterfall ---

#[test]
fn waterfall_peak_bin_matches_tone_frequency() {
    use crate::sync::waterfall::compute_waterfall;
    use crate::modulate::ft8::{FT8_TONE_SPACING_HZ, FT8_SAMPLES_PER_SYM};

    let fs = 12_000.0f32;
    let base_hz = 1_000.0f32;
    let tone_idx = 3usize;
    let f_tone = base_hz + tone_idx as f32 * FT8_TONE_SPACING_HZ;

    let n = FT8_SAMPLES_PER_SYM;
    let iq: Vec<C32> = (0..n).map(|k| {
        let phi = std::f32::consts::TAU * f_tone * (k as f32) / fs;
        C32::new(phi.cos(), phi.sin())
    }).collect();

    let wf = compute_waterfall(&iq, fs, base_hz, FT8_TONE_SPACING_HZ, n, 1, 8, 0);

    let peak_bin = (0..8).max_by(|&a, &b| wf.get(0, a).partial_cmp(&wf.get(0, b)).unwrap()).unwrap();
    assert_eq!(peak_bin, tone_idx,
        "Peak waterfall bin {} does not match transmitted tone {}", peak_bin, tone_idx);
}

#[test]
fn waterfall_peak_bin_dominates_neighbours() {
    use crate::sync::waterfall::compute_waterfall;
    use crate::modulate::ft8::{FT8_TONE_SPACING_HZ, FT8_SAMPLES_PER_SYM};

    let fs = 12_000.0f32;
    let base_hz = 1_000.0f32;
    let tone_idx = 5usize;
    let f_tone = base_hz + tone_idx as f32 * FT8_TONE_SPACING_HZ;

    let n = FT8_SAMPLES_PER_SYM;
    let iq: Vec<C32> = (0..n).map(|k| {
        let phi = std::f32::consts::TAU * f_tone * (k as f32) / fs;
        C32::new(phi.cos(), phi.sin())
    }).collect();

    let wf = compute_waterfall(&iq, fs, base_hz, FT8_TONE_SPACING_HZ, n, 1, 8, 0);

    let peak_e = wf.get(0, tone_idx);
    let left_e  = if tone_idx > 0 { wf.get(0, tone_idx - 1) } else { f32::NEG_INFINITY };
    let right_e = if tone_idx < 7 { wf.get(0, tone_idx + 1) } else { f32::NEG_INFINITY };

    assert!(peak_e - left_e > 10.0,
        "Peak bin not dominant over left neighbour: peak={:.2} left={:.2}", peak_e, left_e);
    assert!(peak_e - right_e > 10.0,
        "Peak bin not dominant over right neighbour: peak={:.2} right={:.2}", peak_e, right_e);
}

#[test]
fn waterfall_time_offset_shifts_window() {
    use crate::sync::waterfall::compute_waterfall;
    use crate::modulate::ft8::{FT8_TONE_SPACING_HZ, FT8_SAMPLES_PER_SYM};

    let fs = 12_000.0f32;
    let base_hz = 1_000.0f32;
    let tone_idx = 2usize;
    let f_tone = base_hz + tone_idx as f32 * FT8_TONE_SPACING_HZ;
    let n = FT8_SAMPLES_PER_SYM;

    let mut buf = vec![C32::new(0.0, 0.0); 2 * n];
    for k in 0..n {
        let phi = std::f32::consts::TAU * f_tone * (k as f32) / fs;
        buf[n + k] = C32::new(phi.cos(), phi.sin());
    }

    let wf0 = compute_waterfall(&buf, fs, base_hz, FT8_TONE_SPACING_HZ, n, 1, 8, 0);
    let e_silence = wf0.get(0, tone_idx);

    let wf1 = compute_waterfall(&buf, fs, base_hz, FT8_TONE_SPACING_HZ, n, 1, 8, n);
    let e_tone = wf1.get(0, tone_idx);
    let peak_bin = (0..8).max_by(|&a, &b| wf1.get(0, a).partial_cmp(&wf1.get(0, b)).unwrap()).unwrap();

    assert!(e_tone > e_silence + 10.0,
        "Tone energy with correct offset ({:.2}) should dominate silence ({:.2})", e_tone, e_silence);
    assert_eq!(peak_bin, tone_idx,
        "Peak bin {} should be {} with correct time offset", peak_bin, tone_idx);
}

// --- Costas score ---

#[test]
fn costas_score_peaks_at_correct_location() {
    use crate::sync::waterfall::compute_waterfall;
    use crate::sync::costas::costas_score;
    use crate::modulate::ft8::{Ft8Mod, Ft8Frame, FT8_TONE_SPACING_HZ, FT8_SAMPLES_PER_SYM, FT8_TOTAL_SYMS};

    let fs = 12_000.0f32;
    let base_hz = 1_000.0f32;

    let iq = Ft8Mod::new(fs, base_hz, 0.0, 1.0).modulate(&Ft8Frame::zeros());
    let wf = compute_waterfall(&iq, fs, base_hz, FT8_TONE_SPACING_HZ,
        FT8_SAMPLES_PER_SYM, FT8_TOTAL_SYMS, 16, 0);

    let costas = [3u8, 1, 4, 0, 6, 5, 2];
    let sync_pos: Vec<i32> = vec![0, 36, 72];

    let score_correct = costas_score(&wf, &costas, &sync_pos, 0, 0);
    let score_wrong_freq = costas_score(&wf, &costas, &sync_pos, 0, 4);
    let score_wrong_time = costas_score(&wf, &costas, &sync_pos, 5, 0);

    assert!(score_correct > score_wrong_freq,
        "Correct location ({:.2}) should beat wrong freq ({:.2})", score_correct, score_wrong_freq);
    assert!(score_correct > score_wrong_time,
        "Correct location ({:.2}) should beat wrong time ({:.2})", score_correct, score_wrong_time);
}

// --- find_candidates ---

#[test]
fn find_candidates_top_hit_at_correct_location() {
    use crate::sync::waterfall::compute_waterfall;
    use crate::sync::costas::find_candidates;
    use crate::modulate::ft8::{Ft8Mod, Ft8Frame, FT8_TONE_SPACING_HZ, FT8_SAMPLES_PER_SYM, FT8_TOTAL_SYMS, FT8_TONES};

    let fs = 12_000.0f32;
    let base_hz = 1_000.0f32;

    let iq = Ft8Mod::new(fs, base_hz, 0.0, 1.0).modulate(&Ft8Frame::zeros());

    let num_bins = FT8_TONES + 8;
    let wf = compute_waterfall(&iq, fs, base_hz, FT8_TONE_SPACING_HZ,
        FT8_SAMPLES_PER_SYM, FT8_TOTAL_SYMS, num_bins, 0);

    let costas = [3u8, 1, 4, 0, 6, 5, 2];
    let sync_pos: Vec<i32> = vec![0, 36, 72];

    let candidates = find_candidates(&wf, &costas, &sync_pos, FT8_TONES, 0, 0, 5);

    assert!(!candidates.is_empty(), "find_candidates returned no results");
    let best = &candidates[0];
    assert_eq!(best.time_sym, 0, "Best candidate time_sym should be 0, got {}", best.time_sym);
    assert_eq!(best.freq_bin, 0, "Best candidate freq_bin should be 0, got {}", best.freq_bin);
}

// --- LDPC decoder regression ---

#[test]
fn ldpc_decode_soft_early_exit_on_valid_initial_hard() {
    use crate::codec::ldpc::{ldpc_encode, ldpc_decode_soft, K_BYTES, N_BYTES, N};
    use crate::codec::crc::ft8_add_crc;

    let payload = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x23, 0x45, 0x60u8];
    let mut a91 = [0u8; K_BYTES];
    ft8_add_crc(&payload, &mut a91);
    let mut codeword = [0u8; N_BYTES];
    ldpc_encode(&a91, &mut codeword);

    let llr: [f32; N] = core::array::from_fn(|i| {
        let bit = (codeword[i / 8] >> (7 - (i % 8))) & 1;
        if bit == 0 { 5.0 } else { -5.0 }
    });

    let mut plain = [0u8; N];
    let errors = ldpc_decode_soft(&llr, 20, &mut plain);
    assert_eq!(errors, 0,
        "LDPC soft decode failed with moderate-magnitude valid LLRs (got {errors} errors)");
}

#[test]
fn ldpc_decode_soft_returns_best_plain_on_divergence() {
    use crate::codec::ldpc::{ldpc_decode_soft, ldpc_count_errors, N};
    use crate::codec::ldpc::{ldpc_encode, K_BYTES, N_BYTES};
    use crate::codec::crc::ft8_add_crc;

    let payload = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x00u8];
    let mut a91 = [0u8; K_BYTES];
    ft8_add_crc(&payload, &mut a91);
    let mut codeword = [0u8; N_BYTES];
    ldpc_encode(&a91, &mut codeword);

    let llr: [f32; N] = core::array::from_fn(|i| {
        let bit = (codeword[i / 8] >> (7 - (i % 8))) & 1;
        if bit == 0 { 0.1 } else { -0.1 }
    });

    let mut plain = [0u8; N];
    let errors = ldpc_decode_soft(&llr, 20, &mut plain);
    assert_eq!(errors, 0,
        "LDPC decode failed with weak-but-correct-sign LLRs ({errors} errors)");
    assert_eq!(ldpc_count_errors(&plain), 0,
        "Returned plain fails syndrome check");
}
