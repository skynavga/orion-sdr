// src/sync/costas.rs
//
// Costas sync score and candidate search for FT8/FT4.
//
// Algorithm mirrors ft8_lib ft8_sync_score / ftx_find_candidates:
//
//   Score = for each Costas symbol at (sym_pos, expected_tone):
//             energy[expected_tone] − max(energy[expected_tone ± 1 bin])
//                                   − max(energy[expected_tone ± 1 symbol])
//             (clamped to 0 so only clear peaks contribute positively)
//
//   This is a difference metric rather than absolute energy.  It is insensitive
//   to overall signal level and penalises positions where the expected tone has
//   close neighbours with similar energy, which happens when the frame is absent
//   or poorly aligned.
//
//   Candidate search scans:
//   - `time_offset` ∈ [t_min, t_max] symbol offsets within the waterfall
//   - Up to `num_bins - num_costas_tones` frequency starting bins
//   and retains the top-N results in a fixed-size min-heap.

use crate::sync::waterfall::Waterfall;

/// A sync candidate: time and frequency location plus score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// Symbol index of the start of the first Costas block in the waterfall.
    pub time_sym: i32,
    /// Frequency bin index of tone-0 of the frame.
    pub freq_bin: usize,
    /// Costas correlation score (higher = better match).
    pub score: f32,
}

/// Compute the Costas difference-metric score for one candidate location.
///
/// Parameters:
/// - `wf`         — waterfall (num_syms × num_bins)
/// - `costas`     — Costas tone sequence (length = costas_len)
/// - `sync_pos`   — symbol offsets of each Costas block start within the frame
/// - `time_sym`   — symbol offset of the candidate frame start (may be negative)
/// - `freq_bin`   — frequency bin of candidate tone-0
///
/// Returns the sum of per-symbol difference scores (≥ 0 each).
pub fn costas_score(
    wf: &Waterfall,
    costas: &[u8],
    sync_pos: &[i32],
    time_sym: i32,
    freq_bin: usize,
) -> f32 {
    let costas_len = costas.len() as i32;
    let mut total = 0.0f32;

    for &block_start in sync_pos {
        for ci in 0..costas_len {
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

            // Frequency neighbour penalty: max energy in adjacent bins
            let e_freq = {
                let left  = if bin > 0 { wf.get(sym, bin - 1) } else { f32::NEG_INFINITY };
                let right = if bin + 1 < wf.num_tones { wf.get(sym, bin + 1) } else { f32::NEG_INFINITY };
                left.max(right)
            };

            // Time neighbour penalty: max energy in adjacent symbols (same bin)
            let e_time = {
                let prev = if sym > 0 { wf.get(sym - 1, bin) } else { f32::NEG_INFINITY };
                let next = if sym + 1 < wf.num_syms { wf.get(sym + 1, bin) } else { f32::NEG_INFINITY };
                prev.max(next)
            };

            // Difference score for this Costas symbol; clamped to zero
            let diff = e_signal - e_freq.max(e_time);
            total += diff.max(0.0);
        }
    }

    total
}

/// Search a waterfall for the best Costas-matching frame starts.
///
/// Parameters:
/// - `wf`          — waterfall (num_syms × num_bins)
/// - `costas`      — Costas tone sequence
/// - `sync_pos`    — symbol offsets of each Costas block start within a frame
/// - `num_tones`   — number of FSK tones (bins spanned by one frame)
/// - `t_min`       — earliest time_sym to test (may be negative)
/// - `t_max`       — latest time_sym to test (inclusive)
/// - `max_candidates` — maximum number of results to return
///
/// Returns candidates sorted by score, best first.
pub fn find_candidates(
    wf: &Waterfall,
    costas: &[u8],
    sync_pos: &[i32],
    num_tones: usize,
    t_min: i32,
    t_max: i32,
    max_candidates: usize,
) -> Vec<Candidate> {
    // min-heap (BinaryHeap is max by default; we store negative score for min-heap behaviour)
    // Using a Vec + manual pruning for simplicity — O(N·max_candidates) but max_candidates ≤ 20.
    let mut heap: Vec<Candidate> = Vec::with_capacity(max_candidates + 1);

    let max_freq_bin = if wf.num_tones > num_tones {
        wf.num_tones - num_tones
    } else {
        return vec![];
    };

    for time_sym in t_min..=t_max {
        for freq_bin in 0..=max_freq_bin {
            let score = costas_score(wf, costas, sync_pos, time_sym, freq_bin);

            // Keep track of top-N using a manual min-heap on score.
            if heap.len() < max_candidates {
                heap.push(Candidate { time_sym, freq_bin, score });
                if heap.len() == max_candidates {
                    // Build min-heap (smallest score at root)
                    heap.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
                }
            } else if score > heap[0].score {
                heap[0] = Candidate { time_sym, freq_bin, score };
                // Sift down to restore min-heap property (simple linear re-sort for small N)
                heap.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
            }
        }
    }

    // Return sorted best-first
    heap.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    heap
}
