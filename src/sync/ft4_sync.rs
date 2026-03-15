// src/sync/ft4_sync.rs
//
// FT4 frame synchronisation pipeline.
//
// Mirrors ft8_sync.rs but for FT4:
//   - 4-FSK, 2 bits/symbol, 87 data symbols, 105 total symbols.
//   - 4 Costas blocks of 4 symbols at frame positions 1–4, 34–37, 67–70, 100–103.
//   - Ramp symbols (tone 0) at positions 0 and 104 (not used for sync scoring).
//   - Tone spacing: 20.833 Hz, samples/symbol: 576.
//   - Gray code: [0,1,3,2] (2 bits/symbol).
//
// FT4 data symbol positions within the 105-symbol frame:
//   Reserved: 0 (ramp), 1–4 (Costas), 34–37 (Costas), 67–70 (Costas), 100–103 (Costas), 104 (ramp)
//   Data: 5..34, 38..67, 71..100  (29 + 29 + 29 = 87 data symbols)
//
// ft8_lib ft4 data symbol offsets (0-based within frame):
//   k <  29: sym = k + 5
//   k <  58: sym = k + 9
//   k <  87: sym = k + 13
//
// Soft LLR extraction (ft8_lib ft4_extract_symbol, Gray code [0,1,3,2]):
//   s2[j] = log-energy of tone FT4_GRAY4[j]  (j = binary index 0..3)
//   bit 0 (MSB): binary {2,3} → tones {gray4(2),gray4(3)} = {3,2}
//   bit 1 (LSB): binary {1,3} → tones {gray4(1),gray4(3)} = {1,2}
//
//   LLR[0] = max(s2[2],s2[3]) − max(s2[0],s2[1])   (bit 0, MSB)
//   LLR[1] = max(s2[1],s2[3]) − max(s2[0],s2[2])   (bit 1, LSB)

use num_complex::Complex32 as C32;
use crate::codec::ldpc::N;
use crate::modulate::ft4::{
    FT4_TONE_SPACING_HZ, FT4_SAMPLES_PER_SYM, FT4_TOTAL_SYMS, FT4_DATA_SYMS,
    FT4_TONES,
};
use crate::sync::waterfall::{Waterfall, compute_waterfall};
use crate::sync::costas::Candidate;

// FT4 Costas pattern (4 symbols each, 4 blocks).
// From ft8_lib kFT4_Costas_pattern: [0,1,3,2], [1,0,2,3], [2,3,1,0], [3,2,0,1]
// For sync scoring we use the first block; all four match the same template.
// The sync scorer evaluates all 4 blocks simultaneously by passing all 4 starts.
const FT4_COSTAS_BLK: [[u8; 4]; 4] = [
    [0, 1, 3, 2],
    [1, 0, 2, 3],
    [2, 3, 1, 0],
    [3, 2, 0, 1],
];

// Sync block starts within the 105-symbol frame (symbol indices of block[0..4]).
const FT4_SYNC_POS: [i32; 4] = [1, 34, 67, 100];

// FT4 Gray code: binary index → tone index.
const FT4_GRAY4: [usize; 4] = [0, 1, 3, 2];

/// Result of a successful FT4 sync operation.
pub struct Ft4SyncResult {
    /// Symbol offset of the frame start in the waterfall.
    pub time_sym: i32,
    /// Frequency bin of tone-0.
    pub freq_bin: usize,
    /// Costas match score.
    pub score: f32,
    /// 174 soft LLRs ready for `Ft4Codec::decode_soft`.
    pub llr: [f32; N],
}

/// Synchronise and extract soft LLRs from an FT4 IQ buffer.
pub fn ft4_sync(
    iq: &[C32],
    fs: f32,
    base_hz: f32,
    max_hz: f32,
    t_min: i32,
    t_max: i32,
    max_cand: usize,
) -> Vec<Ft4SyncResult> {
    let num_tones_frame = FT4_TONES;
    let freq_range = (max_hz - base_hz).max(0.0);
    let num_bins = (freq_range / FT4_TONE_SPACING_HZ).ceil() as usize + num_tones_frame + 1;

    let wf_syms = (t_max + FT4_TOTAL_SYMS as i32 - t_min).max(1) as usize;
    let wf_sample_start = if t_min >= 0 {
        t_min as usize * FT4_SAMPLES_PER_SYM
    } else {
        0
    };
    let sym_offset_adj = if t_min < 0 { -t_min } else { 0 };

    let wf = compute_waterfall(
        iq,
        fs,
        base_hz,
        FT4_TONE_SPACING_HZ,
        FT4_SAMPLES_PER_SYM,
        wf_syms,
        num_bins,
        wf_sample_start,
    );

    let wf_t_min = 0i32;
    let wf_t_max = (wf_syms as i32 - FT4_TOTAL_SYMS as i32).max(0);

    // For FT4 the Costas pattern differs per block, so we score all four blocks
    // simultaneously using a custom scorer.
    let candidates = find_ft4_candidates(&wf, num_tones_frame, wf_t_min, wf_t_max, max_cand);

    let mut results = Vec::with_capacity(candidates.len());
    for cand in candidates {
        let llr = extract_ft4_llr(&wf, &cand);
        let llr_norm = normalise_llr(llr);
        results.push(Ft4SyncResult {
            time_sym: cand.time_sym - sym_offset_adj,
            freq_bin: cand.freq_bin,
            score: cand.score,
            llr: llr_norm,
        });
    }

    results
}

/// FT4-specific candidate search.
///
/// Scores candidates using all four different Costas block patterns simultaneously.
fn find_ft4_candidates(
    wf: &Waterfall,
    num_tones_frame: usize,
    t_min: i32,
    t_max: i32,
    max_candidates: usize,
) -> Vec<Candidate> {
    let mut heap: Vec<Candidate> = Vec::with_capacity(max_candidates + 1);

    let max_freq_bin = if wf.num_tones > num_tones_frame {
        wf.num_tones - num_tones_frame
    } else {
        return vec![];
    };

    for time_sym in t_min..=t_max {
        for freq_bin in 0..=max_freq_bin {
            let score = ft4_costas_score(wf, time_sym, freq_bin);

            if heap.len() < max_candidates {
                heap.push(Candidate { time_sym, freq_bin, score });
                if heap.len() == max_candidates {
                    heap.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
                }
            } else if score > heap[0].score {
                heap[0] = Candidate { time_sym, freq_bin, score };
                heap.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
            }
        }
    }

    heap.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    heap
}

/// FT4 Costas difference-metric score for all 4 blocks with their distinct patterns.
fn ft4_costas_score(wf: &Waterfall, time_sym: i32, freq_bin: usize) -> f32 {
    let mut total = 0.0f32;

    for (blk, block_start) in FT4_SYNC_POS.iter().enumerate() {
        let costas = &FT4_COSTAS_BLK[blk];
        for ci in 0..4i32 {
            let sym = time_sym + block_start + ci;
            if sym < 0 || sym >= wf.num_syms as i32 {
                continue;
            }
            let sym = sym as usize;
            let expected_tone = costas[ci as usize] as usize;
            let bin = freq_bin + expected_tone;

            if bin >= wf.num_tones {
                continue;
            }

            let e_signal = wf.get(sym, bin);

            let e_freq = {
                let left  = if bin > 0 { wf.get(sym, bin - 1) } else { f32::NEG_INFINITY };
                let right = if bin + 1 < wf.num_tones { wf.get(sym, bin + 1) } else { f32::NEG_INFINITY };
                left.max(right)
            };

            let e_time = {
                let prev = if sym > 0 { wf.get(sym - 1, bin) } else { f32::NEG_INFINITY };
                let next = if sym + 1 < wf.num_syms { wf.get(sym + 1, bin) } else { f32::NEG_INFINITY };
                prev.max(next)
            };

            total += (e_signal - e_freq.max(e_time)).max(0.0);
        }
    }

    total
}

/// Extract 174 raw (unnormalised) LLRs from the waterfall for one FT4 candidate.
///
/// FT4 data symbol positions (0-based in frame):
///   k in 0..29:  sym = k + 5
///   k in 29..58: sym = k + 9
///   k in 58..87: sym = k + 13
fn extract_ft4_llr(wf: &Waterfall, cand: &Candidate) -> [f32; N] {
    let mut llr = [0.0f32; N];
    let mut llr_idx = 0usize;

    for k in 0..FT4_DATA_SYMS {
        // Compute frame-relative symbol position
        let frame_sym = if k < 29 {
            k + 5
        } else if k < 58 {
            k + 9
        } else {
            k + 13
        };

        let wf_sym = cand.time_sym + frame_sym as i32;
        if wf_sym < 0 || wf_sym >= wf.num_syms as i32 {
            llr_idx += 2;
            continue;
        }
        let wf_sym = wf_sym as usize;

        // Gather log-energies for each of the 4 tones.
        let mut s = [f32::NEG_INFINITY; 4];
        for j in 0..4 {
            let bin = cand.freq_bin + j;
            if bin < wf.num_tones {
                s[j] = wf.get(wf_sym, bin);
            }
        }

        // s2[j] = log-energy indexed by binary value (reordered via Gray decode).
        // s2[j] = energy of tone FT4_GRAY4[j].
        let s2 = [s[FT4_GRAY4[0]], s[FT4_GRAY4[1]], s[FT4_GRAY4[2]], s[FT4_GRAY4[3]]];

        // bit 0 (MSB): binary {2,3} have bit0=1
        let bit0 = (s2[2].max(s2[3])) - (s2[0].max(s2[1]));
        // bit 1 (LSB): binary {1,3} have bit1=1
        let bit1 = (s2[1].max(s2[3])) - (s2[0].max(s2[2]));

        // Negate to match our LDPC convention (LLR > 0 = bit likely 0)
        llr[llr_idx]     = -bit0;
        llr[llr_idx + 1] = -bit1;
        llr_idx += 2;
    }

    llr
}

/// Normalise LLRs by signal variance.
///
/// scale = sqrt(24 / variance)  (same formula as FT8; 24 is kept consistent).
fn normalise_llr(mut llr: [f32; N]) -> [f32; N] {
    let variance: f32 = llr.iter().map(|x| x * x).sum::<f32>() / N as f32;
    if variance > 1e-10 {
        let scale = (24.0 / variance).sqrt();
        for v in llr.iter_mut() {
            *v *= scale;
        }
    }
    llr
}
