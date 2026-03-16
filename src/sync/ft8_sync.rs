// src/sync/ft8_sync.rs
//
// FT8 frame synchronisation pipeline.
//
// Decodes a raw IQ buffer containing (possibly) one or more FT8 frames into
// soft LLR vectors ready for LDPC decoding.
//
// Pipeline:
//   1. Compute waterfall (symbol-rate magnitude spectrogram).
//   2. Search waterfall for Costas-matching candidates (costas.rs).
//   3. For each candidate: extract per-symbol log-energy vectors and compute
//      soft LLRs via the max-log approximation.
//   4. Normalise LLRs by signal variance before returning.
//
// The waterfall is computed once for the full IQ buffer.  Multiple candidates
// share the same waterfall, so the compute cost is paid once.
//
// Soft LLR extraction (ft8_lib ft8_extract_symbol):
//   FT8 Gray code: [0,1,3,2,5,6,4,7]
//   For bit position b (0=MSB, 2=LSB of a 3-bit symbol):
//     s2[j] = log_energy of tone j (reordered by Gray map)
//     LLR[b] = max(s2 where bit b of Gray(j)=1) − max(s2 where bit b of Gray(j)=0)
//
//   For FT8 (8-FSK, 3 bits/symbol):
//     bit 0 (MSB): 1 in tones 4,5,6,7  (Gray indices 3,5,6,4,7 → gray[3..7])
//     bit 1:       1 in tones 2,3,6,7
//     bit 2 (LSB): 1 in tones 1,3,5,7
//
// Normalisation (ft8_lib):
//   variance = mean(llr²) over all 174 values
//   scale    = sqrt(24.0 / variance)   [24 = 8 tones × 3 bits]
//   llr_out  = llr_raw × scale

use num_complex::Complex32 as C32;
use crate::codec::ldpc::N;
use crate::modulate::ft8::{
    FT8_TONE_SPACING_HZ, FT8_SAMPLES_PER_SYM, FT8_TOTAL_SYMS,
    FT8_TONES,
};
use crate::sync::waterfall::{Waterfall, compute_waterfall};
use crate::sync::costas::{Candidate, find_candidates};

// FT8 Costas pattern and sync block symbol positions within the 79-symbol frame.
const FT8_COSTAS: [u8; 7] = [3, 1, 4, 0, 6, 5, 2];
const FT8_SYNC_POS: [i32; 3] = [0, 36, 72];

// FT8 Gray code (ft8_lib kFT8_Gray_map): binary index → tone index.
//   [0, 1, 3, 2, 5, 6, 4, 7]
//
// For soft LLR extraction (ft8_lib ft8_extract_symbol):
//   s2[j] = log_energy[FT8_GRAY8[j]]  — energy indexed by binary value.
//   logl[0] = max(s2[4..7]) - max(s2[0..3])  — MSB of binary index (bit position 0)
//   logl[1] = max(s2[2],s2[3],s2[6],s2[7]) - max(s2[0],s2[1],s2[4],s2[5])  — bit 1
//   logl[2] = max(s2[1],s2[3],s2[5],s2[7]) - max(s2[0],s2[2],s2[4],s2[6])  — LSB (bit 2)
//
// ft8_lib convention: positive logl[b] → bit b more likely 1.
// Our LDPC decoder: positive LLR → bit more likely 0.
// → negate logl before feeding to our decoder.

// FT8_GRAY8[j] = tone for binary index j.  Matches FT8_GRAY in codec/gray.rs:
// binary 0→tone 0, 1→1, 2→3, 3→2, 4→5, 5→6, 6→4, 7→7.
// s2[j] = wf_energy[FT8_GRAY8[j]] gives energy indexed by binary value.
const FT8_GRAY8: [usize; 8] = [0, 1, 3, 2, 5, 6, 4, 7];

/// Result of a successful FT8 sync operation.
pub struct Ft8SyncResult {
    /// Symbol offset of the frame start in the waterfall.
    pub time_sym: i32,
    /// Frequency bin of tone-0.
    pub freq_bin: usize,
    /// Costas match score.
    pub score: f32,
    /// 174 soft LLRs ready for `Ft8Codec::decode_soft`.
    pub llr: [f32; N],
}

/// Synchronise and extract soft LLRs from an FT8 IQ buffer.
///
/// Parameters:
/// - `iq`       — arbitrary-length IQ buffer at `fs` Hz
/// - `fs`       — sample rate (should be 12 000 Hz)
/// - `base_hz`  — lowest frequency to search (tone-0 minimum, Hz)
/// - `max_hz`   — highest frequency to search (tone-0 maximum, Hz)
/// - `t_min`    — earliest symbol offset to search (negative = before start of buffer)
/// - `t_max`    — latest symbol offset to search
/// - `max_cand` — maximum candidates to score (top-N by Costas score)
///
/// Returns up to `max_cand` results, ordered by Costas score (best first).
/// Each result includes the soft LLR vector for LDPC decoding.
pub fn ft8_sync(
    iq: &[C32],
    fs: f32,
    base_hz: f32,
    max_hz: f32,
    t_min: i32,
    t_max: i32,
    max_cand: usize,
) -> Vec<Ft8SyncResult> {
    // Number of frequency bins to cover [base_hz, max_hz] with FT8_TONE_SPACING_HZ spacing.
    // Add extra so frames starting anywhere in [base_hz, max_hz - frame_bandwidth] are found.
    let num_tones_frame = FT8_TONES;
    let freq_range = (max_hz - base_hz).max(0.0);
    let num_bins = (freq_range / FT8_TONE_SPACING_HZ).ceil() as usize + num_tones_frame + 1;

    // We need enough symbol rows to cover [t_min, t_max + FT8_TOTAL_SYMS).
    let wf_syms = (t_max + FT8_TOTAL_SYMS as i32 - t_min).max(1) as usize;
    // Sample offset for symbol 0 of the waterfall (t_min can be negative).
    let wf_sample_start = if t_min >= 0 {
        t_min as usize * FT8_SAMPLES_PER_SYM
    } else {
        0 // clip: start of buffer
    };
    // Symbol offset adjustment when t_min < 0: wf symbol 0 corresponds to sample 0,
    // which is t_min symbols before the nominal frame start.
    let sym_offset_adj = if t_min < 0 { -t_min } else { 0 };

    let wf = compute_waterfall(
        iq,
        fs,
        base_hz,
        FT8_TONE_SPACING_HZ,
        FT8_SAMPLES_PER_SYM,
        wf_syms,
        num_bins,
        wf_sample_start,
    );

    // Find candidates in the waterfall.
    // t_min/t_max are expressed relative to the waterfall (wf symbol 0).
    let wf_t_min = 0i32;
    let wf_t_max = (wf_syms as i32 - FT8_TOTAL_SYMS as i32).max(0);

    let sync_pos_adjusted: Vec<i32> = FT8_SYNC_POS.iter().map(|&p| p).collect();

    let candidates = find_candidates(
        &wf,
        &FT8_COSTAS,
        &sync_pos_adjusted,
        num_tones_frame,
        wf_t_min,
        wf_t_max,
        max_cand,
    );

    // For each candidate extract soft LLRs.
    let mut results = Vec::with_capacity(candidates.len());
    for cand in candidates {
        let llr = extract_ft8_llr(&wf, &cand);
        let llr_norm = normalise_llr(llr);
        results.push(Ft8SyncResult {
            time_sym: cand.time_sym - sym_offset_adj,
            freq_bin: cand.freq_bin,
            score: cand.score,
            llr: llr_norm,
        });
    }

    results
}

/// Extract 174 raw (unnormalised) LLRs from the waterfall for one candidate.
///
/// Data symbol positions in the 79-symbol FT8 frame: all positions not occupied
/// by Costas blocks.  Costas blocks: [0,7), [36,43), [72,79).
/// Data positions in frame order: 7..36, 43..72, 79 (but frame has 79 symbols total,
/// so: 7..36 = 29 syms, 43..72 = 29 syms, same pattern for 3rd data segment but
/// positions 72..79 are Costas, so: 7..36 (29) + 43..72 (29) = 58, which matches
/// FT8_DATA_SYMS = 58).  Wait — frame is 79 syms: sync@0-6, data@7-35, sync@36-42,
/// data@43-71, sync@72-78.  So data is at positions [7,36) ∪ [43,72).
fn extract_ft8_llr(wf: &Waterfall, cand: &Candidate) -> [f32; N] {
    let mut llr = [0.0f32; N];
    let mut llr_idx = 0usize;

    // Data symbol positions within the frame (relative to frame start).
    // FT8: sync at [0,7), [36,43), [72,79); data at [7,36), [43,72).
    let data_ranges: &[(usize, usize)] = &[(7, 36), (43, 72)];

    for &(range_start, range_end) in data_ranges {
        for data_sym in range_start..range_end {
            let wf_sym = cand.time_sym + data_sym as i32;
            if wf_sym < 0 || wf_sym >= wf.num_syms as i32 {
                // Missing symbol: zero LLR (maximum uncertainty)
                llr_idx += 3;
                continue;
            }
            let wf_sym = wf_sym as usize;

            // Gather log-energies for each of the 8 tones in this symbol.
            // s[j] = log-energy of tone (freq_bin + j).
            let mut s = [f32::NEG_INFINITY; 8];
            for j in 0..8 {
                let bin = cand.freq_bin + j;
                if bin < wf.num_tones {
                    s[j] = wf.get(wf_sym, bin);
                }
            }

            // s2[j] = log-energy indexed by binary value (reordered via Gray decode).
            // s2[j] = energy of tone FT8_GRAY8[j].
            let s2 = [
                s[FT8_GRAY8[0]], s[FT8_GRAY8[1]], s[FT8_GRAY8[2]], s[FT8_GRAY8[3]],
                s[FT8_GRAY8[4]], s[FT8_GRAY8[5]], s[FT8_GRAY8[6]], s[FT8_GRAY8[7]],
            ];

            // Max-log LLR for each bit:
            // LLR > 0  → bit more likely 0  (positive = bit=0 in this convention).
            // bit 0 (MSB): binary value has bit 0 set ↔ binary 4..7 ↔ s2[4..7]
            llr[llr_idx]     = max4(s2[4], s2[5], s2[6], s2[7])
                              - max4(s2[0], s2[1], s2[2], s2[3]);
            // bit 1: binary value has bit 1 set ↔ binary {2,3,6,7}
            llr[llr_idx + 1] = max4(s2[2], s2[3], s2[6], s2[7])
                              - max4(s2[0], s2[1], s2[4], s2[5]);
            // bit 2 (LSB): binary value has bit 2 set ↔ binary {1,3,5,7}
            llr[llr_idx + 2] = max4(s2[1], s2[3], s2[5], s2[7])
                              - max4(s2[0], s2[2], s2[4], s2[6]);

            // Negate: ft8_lib convention is LLR > 0 = bit likely 1; our LDPC decoder
            // uses LLR > 0 = bit likely 0.  Flip to match our decoder.
            llr[llr_idx]     = -llr[llr_idx];
            llr[llr_idx + 1] = -llr[llr_idx + 1];
            llr[llr_idx + 2] = -llr[llr_idx + 2];

            llr_idx += 3;
        }
    }

    llr
}

/// Normalise LLRs by signal variance.
///
/// scale = sqrt(24 / variance)  where variance = mean(llr²).
/// 24 = 8 tones × 3 bits/symbol.  Prevents LDPC decoder saturation.
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

#[inline]
fn max4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    a.max(b).max(c.max(d))
}

