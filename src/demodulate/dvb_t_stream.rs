// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/demodulate/dvb_t_stream.rs
//
// Streaming DVB-T frame receiver: a `feed`/`flush` front end over the batch
// `DvbTFrameDemod`, mirroring `OfdmFrameStreamDemod`. A DVB-T signal is a
// continuous run of fixed-size frames with no preamble, so the receiver
// accumulates IQ, guard-interval-acquires the next frame at the front of the
// buffer, decodes it, drains its samples, and loops — holding a partially-arrived
// frame until a later `feed` completes it.
//
// The frame geometry (symbols per frame, payload length per frame) is fixed by
// the link and supplied at construction, exactly as the batch demod takes
// `n_symbols`/`payload_len` — a cold DVB-T receiver knows the mode from
// configuration (or an out-of-band TPS lock) and confirms it via the TPS word.

use super::dvb_t_frame::{DvbTFrameDemod, DvbTRxError, DvbTRxFrame};
use crate::sync::dvb_t_gi_sync;
use crate::waveform::dvb_t::{DVB_T_N_FFT, DvbTFrameParams};
use num_complex::Complex32 as C32;

/// A streaming DVB-T receiver. Push IQ with [`feed`](Self::feed); it
/// guard-interval-acquires and decodes each fixed-size frame as its samples
/// arrive, returning the completed ones. [`flush`](Self::flush) runs a final pass
/// over the residual buffer.
pub struct DvbTFrameStreamDemod {
    /// The per-frame demodulator (carries the params + the integer-CFO flag).
    demod: DvbTFrameDemod,
    /// OFDM symbols per frame (from the paired modulator's `DvbTFrame::n_symbols`).
    n_symbols: usize,
    /// Payload byte count per frame (for trimming), from the modulator.
    payload_len: usize,
    /// Samples per OFDM symbol (`n_fft + cp_len`).
    sps: usize,
    /// Accumulated, not-yet-consumed IQ.
    buf: Vec<C32>,
}

impl DvbTFrameStreamDemod {
    /// Builds a streaming receiver for a link whose frames are `n_symbols` OFDM
    /// symbols carrying `payload_len` payload bytes each, under `params`.
    pub fn new(params: DvbTFrameParams, n_symbols: usize, payload_len: usize) -> Self {
        let cp_len = params.config().carrier_plan.cp_len();
        Self {
            demod: DvbTFrameDemod::new(params),
            n_symbols,
            payload_len,
            sps: DVB_T_N_FFT + cp_len,
            buf: Vec::new(),
        }
    }

    /// Enables (or disables) internal integer-CFO correction on the underlying
    /// frame demod — a link-constant knob set once at construction (see
    /// [`DvbTFrameDemod::with_integer_cfo_correction`]). Each decoded frame then has
    /// its whole-subcarrier offset removed internally.
    pub fn with_integer_cfo_correction(mut self, on: bool) -> Self {
        self.demod = self.demod.with_integer_cfo_correction(on);
        self
    }

    /// Accumulated (not-yet-consumed) sample count.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Read-only view of the accumulated IQ buffer.
    pub fn view_buf(&self) -> &[C32] {
        &self.buf
    }

    /// Discards all accumulated samples.
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// Samples one frame occupies once acquired (`n_symbols · sps`).
    fn frame_samples(&self) -> usize {
        self.n_symbols * self.sps
    }

    /// Feeds IQ samples and returns any frames (or errors) that completed.
    pub fn feed(&mut self, iq: &[C32]) -> Vec<Result<DvbTRxFrame, DvbTRxError>> {
        self.buf.extend_from_slice(iq);
        self.drain()
    }

    /// Runs a final decode pass over the residual buffer (end of stream). Same
    /// semantics as [`feed`](Self::feed) with no new samples.
    pub fn flush(&mut self) -> Vec<Result<DvbTRxFrame, DvbTRxError>> {
        self.drain()
    }

    /// Repeatedly acquires and decodes frames from the front of the buffer,
    /// consuming their samples, until no further complete frame is present.
    fn drain(&mut self) -> Vec<Result<DvbTRxFrame, DvbTRxError>> {
        let mut out = Vec::new();
        while let FrameStep::Decoded(result, consume_to) = self.try_one_frame() {
            self.buf.drain(..consume_to);
            out.push(result);
        }
        out
    }

    /// Attempts to acquire and decode one frame at the front of the buffer.
    fn try_one_frame(&mut self) -> FrameStep {
        let n_fft = DVB_T_N_FFT;
        let cp_len = self.sps - n_fft;
        let fs = self.demod.params().config().fs;

        // The GI search spans one symbol period; the frame's data then needs
        // `n_symbols · sps` more samples past the acquired start. Wait until the
        // worst case (start near the end of the search window) can be present.
        let need = self.sps + self.frame_samples();
        if self.buf.len() < need {
            return FrameStep::NeedMore;
        }

        // Acquire the symbol boundary at the front (search one symbol period).
        let Some(acq) = dvb_t_gi_sync(&self.buf, n_fft, cp_len, fs, self.sps) else {
            return FrameStep::NeedMore;
        };
        let start = acq.start_sample;
        let consume_to = start + self.frame_samples();
        if consume_to > self.buf.len() {
            return FrameStep::NeedMore;
        }

        // Decode the frame from its acquired start (the batch demod re-locks the
        // CP at offset 0 of this slice, which is idempotent; with integer-CFO
        // correction enabled it also removes any whole-subcarrier offset there).
        match self
            .demod
            .decode(&self.buf[start..], self.n_symbols, self.payload_len)
        {
            Ok(frame) => FrameStep::Decoded(Ok(frame), consume_to),
            // A genuine decode failure on a fully-present frame: report it and
            // consume past this frame so the stream advances (no re-locking the
            // same corrupt occurrence forever).
            Err(e) => FrameStep::Decoded(Err(e), consume_to),
        }
    }
}

/// The outcome of one attempt to decode a frame from the front of the buffer.
enum FrameStep {
    /// A frame (or a decode error on a complete frame) plus how many samples to
    /// drain from the front.
    Decoded(Result<DvbTRxFrame, DvbTRxError>, usize),
    /// Not enough samples buffered yet — hold and retry on the next `feed`.
    NeedMore,
}
