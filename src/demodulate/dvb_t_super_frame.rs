// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/demodulate/dvb_t_super_frame.rs
//
// The conformant DVB-T OFDM SUPER-FRAME demodulator (ETSI EN 300 744 §4.4/§4.6):
// the inverse of `modulate::dvb_t_super_frame`. It decodes the four consecutive
// frames, checks the super-frame structure (frame-number sequence 0..3 and the
// alternating TPS sync word, implied by the recovered frame numbers), reassembles
// the 16-bit cell identifier from its two byte halves, and concatenates the four
// frames' payloads.
//
// A per-standard ORCHESTRATOR over the single-frame `demodulate::dvb_t_frame`;
// it adds no new PHY, only the multi-frame sequencing and cross-frame checks.

use super::dvb_t_frame::{DvbTRxError, dvb_t_frame_demodulate};
use crate::modulate::dvb_t_super_frame::{DVB_T_FRAMES_PER_SUPER_FRAME, DvbTSuperFrameParams};
use crate::waveform::dvb_t::DVB_T_N_FFT;
use num_complex::Complex32 as C32;

/// The recovered contents of a DVB-T super-frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DvbTRxSuperFrame {
    /// The four frames' payloads, concatenated in order.
    pub payload: Vec<u8>,
    /// The 16-bit cell identifier, reassembled from the high byte (frames 1 & 3)
    /// and low byte (frames 2 & 4).
    pub cell_id: u16,
}

/// Errors from [`dvb_t_super_frame_demodulate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DvbTRxSuperFrameError {
    /// A constituent frame failed to acquire or decode. Carries the frame index
    /// (0..3) and the underlying single-frame error.
    #[error("super-frame: frame {frame} failed: {source}")]
    Frame {
        frame: usize,
        #[source]
        source: DvbTRxError,
    },
    /// The recovered frame numbers were not the expected 0,1,2,3 sequence (the
    /// super-frame is misaligned or a TPS word was mis-decoded).
    #[error("super-frame: frame numbers out of sequence (got {got:?}, expected 0..3)")]
    FrameSequence {
        got: [u8; DVB_T_FRAMES_PER_SUPER_FRAME],
    },
    /// The buffer is too short to hold the whole super-frame.
    #[error("super-frame: too few samples for four frames")]
    Incomplete,
}

/// Demodulates one conformant DVB-T super-frame from `iq`, starting at the first
/// frame's first sample (each frame GI-acquires from its own cyclic prefix, so no
/// preamble is needed). `params` supplies the cold-start MCS and is used to
/// re-derive each frame's expected number and cell-id byte; `symbols_per_frame`
/// and `frame_payload_lens` come from the paired modulator's [`DvbTSuperFrame`].
///
/// Verifies the frame-number sequence is `0,1,2,3` (which, via
/// [`DvbTSuperFrameParams::frame`], also implies the correct alternating TPS sync
/// word), reassembles the 16-bit cell id, and concatenates the payloads.
pub fn dvb_t_super_frame_demodulate(
    params: DvbTSuperFrameParams,
    iq: &[C32],
    symbols_per_frame: usize,
    frame_payload_lens: [usize; DVB_T_FRAMES_PER_SUPER_FRAME],
) -> Result<DvbTRxSuperFrame, DvbTRxSuperFrameError> {
    let cp_len = params.frame(0).config().carrier_plan.cp_len();
    let sps = DVB_T_N_FFT + cp_len;
    let frame_samples = symbols_per_frame * sps;

    let mut payload = Vec::new();
    let mut frame_numbers = [0u8; DVB_T_FRAMES_PER_SUPER_FRAME];
    let mut cell_hi = 0u8;
    let mut cell_lo = 0u8;

    for f in 0..DVB_T_FRAMES_PER_SUPER_FRAME {
        // Each frame carries lead-in room for its own GI search: give the RX the
        // sub-buffer from this frame's start to the end (the single-frame RX finds
        // the CP boundary within the first symbol period).
        let start = f * frame_samples;
        let sub = iq.get(start..).ok_or(DvbTRxSuperFrameError::Incomplete)?;
        let rx = dvb_t_frame_demodulate(
            params.frame(f as u8),
            sub,
            symbols_per_frame,
            frame_payload_lens[f],
        )
        .map_err(|source| DvbTRxSuperFrameError::Frame { frame: f, source })?;

        frame_numbers[f] = rx.tps.frame_number;
        // Reassemble the cell id from the byte the frame's parity carries.
        if f.is_multiple_of(2) {
            cell_hi = rx.tps.cell_id; // frames 1 & 3 → high byte
        } else {
            cell_lo = rx.tps.cell_id; // frames 2 & 4 → low byte
        }
        payload.extend_from_slice(&rx.payload);
    }

    // The super-frame's frame numbers must be exactly 0,1,2,3.
    if frame_numbers != [0, 1, 2, 3] {
        return Err(DvbTRxSuperFrameError::FrameSequence { got: frame_numbers });
    }

    let cell_id = ((cell_hi as u16) << 8) | cell_lo as u16;
    Ok(DvbTRxSuperFrame { payload, cell_id })
}
