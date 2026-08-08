// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/modulate/dvb_t_super_frame.rs
//
// The conformant DVB-T OFDM SUPER-FRAME modulator (ETSI EN 300 744 §4.4/§4.6):
// four consecutive 68-symbol frames (frame numbers 0..3) emitted back to back.
// This is the multi-frame driver over the single-frame `DvbTFrameMod`; it
// sequences the four frames so a receiver observes the standard's super-frame
// structure:
//   • the TPS synchronization word alternates every frame — frames 1 & 3 (frame
//     numbers 0, 2) use TPS_SYNC_WORD_13, frames 2 & 4 (1, 3) use
//     TPS_SYNC_WORD_24 (§4.6.2.2), which is what lets a receiver find the
//     super-frame boundary;
//   • the 16-bit cell identifier is split across the super-frame — its most
//     significant byte (b15..b8) is signalled in frames 1 & 3 and its least
//     significant byte (b7..b0) in frames 2 & 4 (§4.6.2.10);
//   • the frame number s23,s24 counts 0..3 across the super-frame.
//
// SCOPE. The payload is split into four contiguous parts, one per frame, each
// carried by a self-contained conformant frame (per-frame FEC, with null-packet
// stuffing where a part does not fill its frame — see `DvbTFrameMod`).
// This reproduces the frame-level super-frame structure a receiver locks onto.
// The standard's stronger byte-continuous stream (§4.7 Table 16: an integer
// number of RS packets per super-frame, with a packet free to straddle a frame
// boundary for rates whose per-frame count is fractional, e.g. QPSK r3/4 =
// 94.5 packets/frame) is a streaming refinement left to a future continuous-FEC
// super-frame path; here each frame codes its own part independently.

use super::dvb_t_frame::{DvbTFrame, DvbTFrameMod};
use crate::fec::PunctureRate;
use crate::modulate::ConstellationOrder;
use crate::waveform::dvb_t::{DvbTFrameParams, DvbTLinkParams, GuardInterval};
use num_complex::Complex32 as C32;

/// Number of OFDM frames in one DVB-T super-frame (§4.4).
pub const DVB_T_FRAMES_PER_SUPER_FRAME: usize = 4;

/// Transmission parameters for a conformant DVB-T super-frame: the shared link
/// parameters ([`DvbTLinkParams`]) plus the **full 16-bit** cell identifier
/// (§4.6.2.10), which the driver splits across the four frames. Unlike
/// [`DvbTFrameParams`] there is no per-frame number — the driver derives 0..3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DvbTSuperFrameParams {
    /// Guard interval, constellation, and code rate (constant across the link).
    pub link: DvbTLinkParams,
    /// The full 16-bit cell identifier: b15..b8 ride frames 1 & 3, b7..b0 frames
    /// 2 & 4.
    pub cell_id: u16,
}

impl DvbTSuperFrameParams {
    /// The guard interval (from the link parameters).
    pub fn guard(self) -> GuardInterval {
        self.link.guard
    }

    /// The constellation (from the link parameters).
    pub fn constellation(self) -> ConstellationOrder {
        self.link.constellation
    }

    /// The inner code rate (from the link parameters).
    pub fn code_rate(self) -> PunctureRate {
        self.link.code_rate
    }

    /// The single-frame parameters for frame `frame_number` (0..3): the shared
    /// link parameters, this frame's number, and the appropriate byte of the
    /// 16-bit cell id (MSB in the even frames 1 & 3, LSB in the odd frames 2 & 4).
    pub fn frame(self, frame_number: u8) -> DvbTFrameParams {
        let cell_id = if frame_number.is_multiple_of(2) {
            (self.cell_id >> 8) as u8 // frames 1 & 3 → high byte
        } else {
            (self.cell_id & 0xFF) as u8 // frames 2 & 4 → low byte
        };
        DvbTFrameParams {
            link: self.link,
            frame_number,
            cell_id,
        }
    }
}

/// A modulated DVB-T super-frame: the time-domain IQ of four consecutive frames
/// plus the numerology a receiver needs to acquire and re-slice it.
#[derive(Debug, Clone)]
pub struct DvbTSuperFrame {
    /// Time-domain baseband IQ of all four frames, concatenated (no preamble).
    pub iq: Vec<C32>,
    /// OFDM symbols per constituent frame (each frame is padded to ≥ 68).
    pub symbols_per_frame: usize,
    /// Samples per OFDM symbol (`n_fft + cp_len`).
    pub samples_per_symbol: usize,
    /// The payload byte count carried by each of the four frames, in order — the
    /// receiver trims each frame's recovered payload to its entry and
    /// concatenates them.
    pub frame_payload_lens: [usize; DVB_T_FRAMES_PER_SUPER_FRAME],
}

impl DvbTSuperFrame {
    /// OFDM symbols across the whole super-frame (`4 · symbols_per_frame`).
    pub fn n_symbols(&self) -> usize {
        DVB_T_FRAMES_PER_SUPER_FRAME * self.symbols_per_frame
    }
}

/// A conformant DVB-T super-frame modulator: the multi-frame driver over
/// [`DvbTFrameMod`]. Constructed with the super-frame parameters (guard interval,
/// constellation, rate, and the full 16-bit cell id); [`modulate`](Self::modulate)
/// emits one four-frame super-frame per call.
#[derive(Debug, Clone)]
pub struct DvbTSuperFrameMod {
    params: DvbTSuperFrameParams,
    window_roll_off: usize,
}

impl DvbTSuperFrameMod {
    /// Builds a super-frame modulator with the given parameters.
    pub fn new(params: DvbTSuperFrameParams) -> Self {
        Self {
            params,
            window_roll_off: 0,
        }
    }

    /// Enables TX symbol windowing for every constituent frame (see
    /// [`DvbTFrameMod::with_symbol_window`](crate::modulate::DvbTFrameMod::with_symbol_window)).
    /// `0` (the default) disables it.
    pub fn with_symbol_window(mut self, roll_off: usize) -> Self {
        self.window_roll_off = roll_off;
        self
    }

    /// The super-frame parameters this modulator was built with.
    pub fn params(&self) -> DvbTSuperFrameParams {
        self.params
    }

    /// Modulates `payload` into one conformant DVB-T super-frame: four consecutive
    /// frames (frame numbers 0..3) with the alternating TPS sync word and the
    /// 16-bit cell id split across them. The payload is divided into four
    /// contiguous parts (as even as possible), each carried by a self-contained
    /// conformant frame; the parts are zero-padded to a common length so all four
    /// frames share one symbol count and the super-frame is a clean
    /// `4 × symbols_per_frame` block.
    pub fn modulate(&self, payload: &[u8]) -> DvbTSuperFrame {
        let params = self.params;
        // Split the payload into four contiguous parts as evenly as possible: the
        // first `rem` parts get one extra byte. `frame_payload_lens` records each
        // real part length so the RX trims back to it.
        let n = DVB_T_FRAMES_PER_SUPER_FRAME;
        let base = payload.len() / n;
        let rem = payload.len() % n;
        let mut parts: Vec<Vec<u8>> = Vec::with_capacity(n);
        let mut off = 0usize;
        let mut frame_payload_lens = [0usize; DVB_T_FRAMES_PER_SUPER_FRAME];
        for (i, len_slot) in frame_payload_lens.iter_mut().enumerate() {
            let len = base + usize::from(i < rem);
            parts.push(payload[off..off + len].to_vec());
            *len_slot = len;
            off += len;
        }
        // Zero-pad every part to the longest so all four frames modulate to an
        // identical symbol count — the super-frame is then a uniform
        // `4 × symbols_per_frame` block the receiver re-slices by offset. Padding
        // bytes sit past each frame's real payload length, so the RX (which trims
        // to `frame_payload_lens`) discards them.
        let part_len = frame_payload_lens.iter().copied().max().unwrap_or(0);
        for part in &mut parts {
            part.resize(part_len, 0);
        }

        let frames: Vec<DvbTFrame> = (0..n)
            .map(|f| {
                DvbTFrameMod::new(params.frame(f as u8))
                    .with_symbol_window(self.window_roll_off)
                    .modulate(&parts[f])
            })
            .collect();
        let symbols_per_frame = frames[0].n_symbols;
        let samples_per_symbol = frames[0].samples_per_symbol;
        debug_assert!(
            frames.iter().all(|f| f.n_symbols == symbols_per_frame),
            "equal-length parts must modulate to equal symbol counts"
        );

        let mut iq = Vec::with_capacity(symbols_per_frame * samples_per_symbol * n);
        for frame in &frames {
            iq.extend_from_slice(&frame.iq);
        }

        DvbTSuperFrame {
            iq,
            symbols_per_frame,
            samples_per_symbol,
            frame_payload_lens,
        }
    }
}
