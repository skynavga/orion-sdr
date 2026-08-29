// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/demodulate/dvb_t_frame.rs
//
// The conformant DVB-T on-air frame DEMODULATOR (ETSI EN 300 744): the exact
// inverse of `modulate::dvb_t_frame`. Acquires the symbol grid from the guard
// interval (no preamble), equalizes each symbol from the scattered/continual
// pilots, soft-demaps the Figure-9a constellation, recovers the TPS word from the
// 17 TPS carriers, runs the payload FEC decode, and undoes the TS energy
// dispersal — recovering both the payload and the TPS-signalled parameters.
//
// A per-standard ORCHESTRATOR over the shared RX stages (no new abstraction):
//   • acquisition     — `sync::dvb_t_gi_sync` (guard-interval ML timing/CFO);
//   • integer CFO     — `sync::dvb_t_integer_cfo` (optional, continual-pilot);
//   • equalize/extract— `OfdmEqualizer::set_pilot_bins` + `ScatteredPilotExtractor`;
//   • soft demap      — `dvb_t_soft_llr` (Figure-9a);
//   • TPS             — `dvb_t_tps::TpsDecoder`;
//   • payload FEC     — `demodulate::ofdm_frame::decode_chain`;
//   • energy dispersal— `waveform::dvb_t_ts`.

use super::dvb_t_probe::DvbTRxProbe;
use super::ofdm::{EqualizerMethod, OfdmEqualizer};
use super::ofdm_frame::{bit_error_rate, decode_chain};
use crate::core::Block;
use crate::dsp::Rotator;
use crate::fec::{CrcKind, DecodeRule, InterleaverKind, ScramblerKind, ScramblerPos};
use crate::modulate::ofdm_frame::{CodecCache, block_plan, encode_chain_stages, inner_encode};
use crate::multicarrier::SymbolFft;
use crate::sync::{dvb_t_gi_sync, dvb_t_integer_cfo};
use crate::waveform::dvb_t::{
    DVB_T_DATA_CARRIERS, DVB_T_FRAME_OUTER, DVB_T_FRAME_OUTER_IL, DVB_T_N_FFT, DvbTFrameParams,
    ScatteredPilotExtractor, dvb_t_frame_fill_with, dvb_t_map_symbol, dvb_t_soft_llr,
    tps_carrier_bins,
};
use crate::waveform::dvb_t_tps::{TPS_SYMBOLS_PER_FRAME, TpsDecoder, TpsWord};
use crate::waveform::dvb_t_ts::{TS_PACKET_LEN, ts_depacketize, ts_energy_disperse};
use num_complex::Complex32 as C32;

/// Number of aligned symbols whose continual-pilot energy the internal integer-CFO
/// estimator accumulates before deciding (the pilot peak is modest — 45 of 1705
/// carriers, boosted ~1.78× — so a few symbols firm up the estimate under noise).
const INTEGER_CFO_ACCUM_SYMBOLS: usize = 8;

/// Trial integer-CFO search span, in whole subcarriers (`±` this many bins). The
/// continual pilots span the active band, so shifts beyond the guard-band margin
/// slide pilots out of band; a few tens of subcarriers is a generous front-end
/// range.
const INTEGER_CFO_MAX_BINS: i32 = 32;

/// What the optional integer-CFO pre-correction produced.
///
/// The two fields go `None` independently, and the *estimate*'s `None` is not
/// the buffer's: a link already on frequency yields `bins: Some(0)` with nothing
/// to rotate, which is a measurement, whereas a disabled or failed estimator
/// yields `bins: None`, which is the absence of one.
struct IntegerCfo {
    /// The rotated buffer, or `None` to decode the caller's `iq` untouched.
    corrected: Option<Vec<C32>>,
    /// The estimated whole-subcarrier offset in bins, or `None` when no
    /// estimate was made — correction disabled, acquisition failed, or no
    /// continual-pilot peak resolved.
    bins: Option<i32>,
}

impl IntegerCfo {
    /// No estimate and no rotation: the estimator did not run or could not
    /// decide.
    const NOT_MEASURED: Self = Self {
        corrected: None,
        bins: None,
    };
}

/// Per-frame receive diagnostics for a DVB-T link — the measured quality ladder
/// an analyzer or a link-margin display reads.
///
/// **Every field is `Option`, and `None` never means zero.** A rung that goes
/// absent exactly when the link fails must stay distinguishable from one
/// reporting a perfect result, or a dead link renders as a flawless one. `None`
/// means "not measured" — the flag that gates it was off, or the stage that
/// produces it did not run.
///
/// # What is deliberately absent
///
/// **There is no `inner_fec_ok`.** DVB-T's inner code is always
/// `ConvCode::DvbK7`, and [`ChainOutcome::inner_ok`] is documented as always
/// `true` for the convolutional arm — its soft Viterbi has no per-block
/// convergence flag. Exposing it would put a permanently-green lock on a display
/// that no link condition could ever move. The meaningful post-inner measurement
/// is [`inner_ber`](Self::inner_ber), which is measured by re-encode and does
/// vary.
///
/// **There is no `crc_ok`.** DVB-T carries no CRC (`CrcKind::None`), so the
/// chain's `crc_ok` reports `true` because there was nothing to check. The outer
/// Reed–Solomon stage is the integrity check on this waveform.
///
/// [`ChainOutcome::inner_ok`]: crate::demodulate::ofdm_frame::ChainOutcome::inner_ok
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DvbTRxDiagnostics {
    /// **Total** carrier frequency offset in Hz — the fractional estimate from
    /// guard-interval acquisition plus any whole-subcarrier offset that was
    /// removed ahead of it (`bins · fs / n_fft`), matching what the generic
    /// COFDM receiver reports in [`OfdmRxFrame::cfo_hz`].
    ///
    /// With integer-CFO correction off this is the fractional term alone, and
    /// so is only unambiguous within ±½ a subcarrier — a larger real offset
    /// aliases into this range and the frame would not have demapped anyway.
    ///
    /// [`OfdmRxFrame::cfo_hz`]: crate::demodulate::OfdmRxFrame::cfo_hz
    pub cfo_hz: Option<f32>,
    /// Normalized guard-interval correlation score in `[0, 1]` at the winning
    /// offset — the receiver's confidence that it locked onto a real cyclic
    /// prefix rather than noise.
    pub sync_score: Option<f32>,
    /// Sample offset of the acquired symbol boundary within the buffer this
    /// frame was decoded from.
    pub timing_offset_samples: Option<usize>,
    /// The whole-subcarrier frequency offset the internal estimator found and
    /// removed, in bins.
    ///
    /// `Some(0)` and `None` mean different things and are kept apart on
    /// purpose: `Some(0)` is "the estimator ran and the link is on frequency",
    /// `None` is "no estimate exists" — correction is disabled (the default),
    /// or acquisition failed, or no pilot peak could be resolved. A diagnostic
    /// that folded them together could not tell a locked receiver from one that
    /// never looked.
    pub integer_cfo_bins: Option<i32>,
    /// Payload error vector magnitude in dB, measured against the payload's own
    /// hard decisions — so it needs no knowledge of the transmitted bits and
    /// works over the air. Lower is better.
    pub evm_db: Option<f32>,
    /// Bit error rate at the **channel's output** — the inner decoder's input.
    /// The classic pre-FEC BER, `CBER`.
    ///
    /// `None` unless the demodulator was built with
    /// [`with_error_rates`](DvbTFrameDemod::with_error_rates): it costs one
    /// encode chain per frame.
    pub channel_ber: Option<f32>,
    /// Bit error rate at the **inner decoder's output**, before Reed–Solomon —
    /// `IBER`, the rung the convolutional code's coding gain shows up in. Same
    /// provenance and same gate as [`channel_ber`](Self::channel_ber).
    pub inner_ber: Option<f32>,
    /// Whether every outer Reed–Solomon codeword decoded.
    ///
    /// **On a frame returned from [`decode`](DvbTFrameDemod::decode) this is
    /// always `Some(true)`**, because DVB-T has no CRC and so the RS stage is
    /// what `ChainOutcome::is_valid` consults — a frame whose outer code failed
    /// is returned as [`DvbTRxError::PayloadDecode`] and never reaches a caller.
    /// It is carried anyway so the ladder is complete and a consumer reading it
    /// generically is not surprised by a missing rung, but
    /// [`rs_corrected_bytes`](Self::rs_corrected_bytes) is the rung that
    /// actually moves with link quality.
    pub outer_fec_ok: Option<bool>,
    /// How many bytes the outer Reed–Solomon decoder corrected across this
    /// frame's codewords.
    ///
    /// The one rung here that degrades *gracefully*: a pass/fail flag saturates,
    /// but a rising correction count is a link approaching the cliff while still
    /// delivering every byte. It is what real DVB-T receivers report and the
    /// natural driver for a packet-granularity error display.
    pub rs_corrected_bytes: Option<u32>,
}

/// The recovered contents of a DVB-T frame: the TS payload, the TPS word read
/// off the carriers, and the frame's receive diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct DvbTRxFrame {
    /// The recovered TS payload bytes (depacketized, trimmed to `payload_len`).
    pub payload: Vec<u8>,
    /// The transmission parameters recovered from the TPS carriers.
    pub tps: TpsWord,
    /// This frame's measured receive quality — see [`DvbTRxDiagnostics`].
    pub diagnostics: DvbTRxDiagnostics,
}

/// Errors from [`DvbTFrameDemod::decode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DvbTRxError {
    #[error("guard-interval acquisition failed (buffer too short or no CP lock)")]
    Acquisition,
    #[error("too few samples for the expected frame")]
    Incomplete,
    #[error("TPS word failed to decode (BCH uncorrectable)")]
    TpsDecode,
    #[error("payload FEC/CRC decode failed")]
    PayloadDecode,
}

/// A conformant, preamble-less DVB-T frame demodulator — the exact inverse of
/// [`DvbTFrameMod`](crate::modulate::DvbTFrameMod). Constructed with the link's
/// cold-start transmission parameters (a real receiver acquires on assumptions,
/// then confirms them against the recovered TPS word); [`decode`](Self::decode)
/// recovers one frame per call.
///
/// # Integer-CFO correction
/// The guard-interval acquisition resolves only the *fractional* CFO (±½ a
/// subcarrier). A capture with a larger front-end offset is shifted by whole
/// subcarriers and will not demap. Enable
/// [`with_integer_cfo_correction`](Self::with_integer_cfo_correction) — a
/// link-constant, set-once builder — and the demod estimates and removes the
/// integer offset internally (from the 45 continual pilots) before decoding. It is
/// off by default: a clean link needs no correction, and the estimate/rotate is
/// skipped entirely.
#[derive(Debug, Clone)]
pub struct DvbTFrameDemod {
    params: DvbTFrameParams,
    integer_cfo: bool,
    rx_window_backoff: usize,
    /// Whether to measure the three rungs that cost extra work — EVM, and the
    /// two BERs. Off by default.
    measure_errors: bool,
}

impl DvbTFrameDemod {
    /// Builds a demodulator for a link with the given cold-start parameters.
    /// Integer-CFO correction is **off** by default (see
    /// [`with_integer_cfo_correction`](Self::with_integer_cfo_correction)).
    pub fn new(params: DvbTFrameParams) -> Self {
        Self {
            params,
            integer_cfo: false,
            rx_window_backoff: 0,
            measure_errors: false,
        }
    }

    /// Enables (or disables) internal integer-CFO correction — a link-constant
    /// knob set once at construction. When on, [`decode`](Self::decode) estimates
    /// the whole-subcarrier offset from the continual pilots and rotates it out
    /// before demapping; when off (the default), no estimate is made and a clean
    /// buffer is decoded unchanged.
    pub fn with_integer_cfo_correction(mut self, on: bool) -> Self {
        self.integer_cfo = on;
        self
    }

    /// Sets the receiver FFT-window back-off in samples (default 0). Pull the
    /// per-symbol FFT window earlier into the guard for multipath/pre-echo
    /// robustness, and to make a matching TX symbol-window taper transparent
    /// (`roll_off = back_off = cp_len/2` is the transparent operating point; see
    /// [`DvbTFrameMod::with_symbol_window`](crate::modulate::DvbTFrameMod::with_symbol_window)).
    /// The scattered-pilot channel estimate is measured at the same back-off, so
    /// the induced phase ramp is corrected — required for a clean decode.
    pub fn with_rx_window_backoff(mut self, backoff: usize) -> Self {
        self.rx_window_backoff = backoff;
        self
    }

    /// Enables (or disables) the three [`DvbTRxDiagnostics`] rungs that cost
    /// work the decode would not otherwise do — [`evm_db`], the channel BER
    /// ([`channel_ber`], CBER), and the post-inner BER ([`inner_ber`], IBER).
    /// **Off by default**; without it all three stay `None`.
    ///
    /// **What it costs.** EVM adds an ideal-point remap per data carrier —
    /// 1512 per OFDM symbol — to the demap loop. The two BERs share one encode
    /// chain per frame, so asking for both pays for it once. With the flag off
    /// the demap loop is the one that shipped before any of this existed: the
    /// branch is hoisted to once per symbol, so an unmeasured receiver pays
    /// nothing.
    ///
    /// **Only these three are gated.** Every other rung is read straight off
    /// work the decode already does — acquisition's CFO and score, the timing
    /// offset, the integer-CFO estimate, the outer-FEC verdict and its
    /// corrected-byte count — so they are always populated. An `Option` that is
    /// `None` only because a flag was off is a worse API than one that is
    /// always there.
    ///
    /// [`evm_db`]: DvbTRxDiagnostics::evm_db
    /// [`channel_ber`]: DvbTRxDiagnostics::channel_ber
    /// [`inner_ber`]: DvbTRxDiagnostics::inner_ber
    pub fn with_error_rates(mut self, on: bool) -> Self {
        self.measure_errors = on;
        self
    }

    /// The transmission parameters this demodulator was built with.
    pub fn params(&self) -> DvbTFrameParams {
        self.params
    }

    /// Whether the BER rungs are measured (see
    /// [`with_error_rates`](Self::with_error_rates)).
    pub fn error_rates(&self) -> bool {
        self.measure_errors
    }

    /// Whether internal integer-CFO correction is enabled.
    pub fn integer_cfo_correction(&self) -> bool {
        self.integer_cfo
    }

    /// The receiver FFT-window back-off in samples (`0` = the standard
    /// CP-boundary window).
    pub fn rx_window_backoff(&self) -> usize {
        self.rx_window_backoff
    }

    /// Removes any whole-subcarrier CFO from `iq`. Aligns to a symbol boundary,
    /// accumulates continual-pilot energy over several symbols, estimates the
    /// integer offset, and rotates the whole buffer by `−k·fs/n_fft`. A frequency
    /// rotation commutes with the timing offset, so the subsequent GI-sync still
    /// finds the same symbol boundary.
    ///
    /// Returns both the rotated buffer and the estimate itself — the estimate is
    /// a diagnostic in its own right, and the function is the only place it
    /// exists. See [`IntegerCfo`] for what its two `None`s distinguish.
    fn integer_cfo_correct(&self, iq: &[C32], n_fft: usize, cp_len: usize, fs: f32) -> IntegerCfo {
        if !self.integer_cfo {
            return IntegerCfo::NOT_MEASURED;
        }
        let sps = n_fft + cp_len;
        let Some(acq) = dvb_t_gi_sync(iq, n_fft, cp_len, fs, sps) else {
            return IntegerCfo::NOT_MEASURED;
        };
        // Integer-CFO estimation uses the standard CP-boundary window (no
        // back-off): it detects a whole-subcarrier *frequency* shift from
        // continual-pilot energy and does not equalize a channel, so the data
        // window back-off is deliberately not applied here.
        let mut symbol_fft = SymbolFft::new(n_fft, cp_len);
        let mut accum = vec![C32::default(); n_fft];
        for s in 0..INTEGER_CFO_ACCUM_SYMBOLS {
            let off = acq.start_sample + s * sps;
            // Accumulate only fully-present symbols. CP-remove reads
            // `iq[off+cp_len .. off+cp_len+n_fft]`, so a full symbol needs
            // `off + sps` samples; guarding on that, `demod_symbol` always
            // succeeds. (The earlier `off + n_fft` guard was too loose — a
            // symbol whose core fit but whose CP did not made CP-remove a
            // no-op, and the loop then double-counted the previous symbol's
            // spectrum into `accum`. Unreachable for a real frame, which always
            // carries far more than the few accumulated symbols, but a latent
            // correctness bug regardless.)
            if off + sps > iq.len() {
                break;
            }
            let Some(freq) = symbol_fft.demod_symbol(&iq[off..]) else {
                break;
            };
            for (a, &x) in accum.iter_mut().zip(freq.iter()) {
                *a += C32::new(x.norm_sqr(), 0.0);
            }
        }
        let Some(k) = dvb_t_integer_cfo(&accum, n_fft, INTEGER_CFO_MAX_BINS).map(|e| e.bins) else {
            return IntegerCfo::NOT_MEASURED;
        };
        if k == 0 {
            // The estimator ran and found the link on frequency — reported as
            // `Some(0)`, not `None`. Nothing to rotate.
            return IntegerCfo {
                corrected: None,
                bins: Some(0),
            };
        }
        let mut corrected = vec![C32::default(); iq.len()];
        Rotator::new(-(k as f32) * fs / n_fft as f32, fs).rotate_block(iq, &mut corrected);
        IntegerCfo {
            corrected: Some(corrected),
            bins: Some(k),
        }
    }

    /// Demodulates one conformant DVB-T frame from `iq`, acquiring the symbol grid
    /// from the guard interval (no preamble). `payload_len` is the original payload
    /// byte count for trimming; `n_symbols` is the frame's symbol count (from the
    /// paired [`DvbTFrameMod`](crate::modulate::DvbTFrameMod)'s [`DvbTFrame`]). The
    /// TPS word recovered from the carriers is returned (a caller can assert it
    /// matches the construction parameters).
    ///
    /// If integer-CFO correction is enabled (see
    /// [`with_integer_cfo_correction`](Self::with_integer_cfo_correction)), the
    /// whole-subcarrier offset is estimated and removed internally first.
    pub fn decode(
        &self,
        iq: &[C32],
        n_symbols: usize,
        payload_len: usize,
    ) -> Result<DvbTRxFrame, DvbTRxError> {
        self.decode_inner(iq, n_symbols, payload_len, None)
    }

    /// [`decode`](Self::decode), additionally filling `probe` with the frame's
    /// equalized data-carrier symbols and its per-coded-bit correction map.
    ///
    /// `pub(crate)`: the public gate is
    /// [`DvbTFrameStreamDemod::feed_probed`](crate::demodulate::DvbTFrameStreamDemod::feed_probed),
    /// matching the generic path, where probing is likewise offered on the
    /// streaming receiver only. A probe is a *stream* instrument — its buffers
    /// exist to be reused across the frames of a continuous broadcast — and the
    /// batch entry point has no reuse to offer.
    ///
    /// **Does not clear `probe`**: the stream receiver drains several frames per
    /// `feed`, and each appends to the same buffers. Clearing is the entry
    /// point's job.
    pub(crate) fn decode_probed(
        &self,
        iq: &[C32],
        n_symbols: usize,
        payload_len: usize,
        probe: &mut DvbTRxProbe,
    ) -> Result<DvbTRxFrame, DvbTRxError> {
        let mark = probe.mark();
        let out = self.decode_inner(iq, n_symbols, payload_len, Some(probe));
        // Roll back only when nothing was committed. A frame that demapped but
        // failed its RS check commits a symbols-only record on purpose — the
        // constellation is exactly what an operator wants when frames stop
        // decoding — and must survive its own error return.
        if out.is_err() && !probe.committed_since(mark) {
            probe.rollback(mark);
        }
        out
    }

    fn decode_inner(
        &self,
        iq: &[C32],
        n_symbols: usize,
        payload_len: usize,
        mut probe: Option<&mut DvbTRxProbe>,
    ) -> Result<DvbTRxFrame, DvbTRxError> {
        let params = self.params;
        let cache = CodecCache::new();
        // Carry the RX window back-off into the derived config so every
        // `SymbolFft` in the per-symbol loop reads at the configured window
        // position (the scattered-pilot estimate then corrects the phase ramp).
        let base = params
            .config()
            .with_rx_window_backoff(self.rx_window_backoff);
        let n_fft = DVB_T_N_FFT;
        let cp_len = base.carrier_plan.cp_len();
        let sps = n_fft + cp_len;
        let vbits = params.constellation().bits_per_symbol();

        // 0. Optional integer-CFO pre-correction (link-constant, set at
        //    construction). A no-op unless enabled AND a whole-subcarrier offset is
        //    present, so a clean link decodes the original buffer unchanged.
        let int_cfo = self.integer_cfo_correct(iq, n_fft, cp_len, base.fs);
        let iq: &[C32] = int_cfo.corrected.as_deref().unwrap_or(iq);

        // 1. Acquire the symbol boundary from the cyclic prefix.
        let acq = dvb_t_gi_sync(iq, n_fft, cp_len, base.fs, sps).ok_or(DvbTRxError::Acquisition)?;
        let start = acq.start_sample;
        if iq.len() < start + n_symbols * sps {
            return Err(DvbTRxError::Incomplete);
        }

        // Acquisition's own outputs are diagnostics the demod was already
        // computing and discarding — `start_sample` was the only field read.
        // The reported CFO is the TOTAL: acquisition resolves only the
        // fractional term, and it ran on the already-rotated buffer, so any
        // whole-subcarrier offset removed at step 0 has to be added back to
        // describe the link rather than the residual.
        let subcarrier_spacing = base.fs / n_fft as f32;
        let mut diagnostics = DvbTRxDiagnostics {
            cfo_hz: Some(acq.cfo_hz + int_cfo.bins.unwrap_or(0) as f32 * subcarrier_spacing),
            sync_score: Some(acq.score),
            timing_offset_samples: Some(start),
            integer_cfo_bins: int_cfo.bins,
            ..Default::default()
        };

        // 2. Per-symbol: CP-remove → FFT → equalize (scattered pilots) → extract
        //    data LLRs + TPS cells.
        let mut extractor = ScatteredPilotExtractor::new(params.guard());
        let mut eq = OfdmEqualizer::new(&base, EqualizerMethod::PerSymbolPilotInterp);
        let mut symbol_fft =
            SymbolFft::new(n_fft, cp_len).with_window_backoff(base.rx_window_backoff);
        let mut tps_dec = TpsDecoder::new();
        let tps_bins = tps_carrier_bins();

        let mut equalized = vec![C32::default(); n_fft];
        let mut data_syms = vec![C32::default(); DVB_T_DATA_CARRIERS];
        let bits_per_sym = DVB_T_DATA_CARRIERS * vbits;
        let mut llrs = vec![0.0f32; n_symbols * bits_per_sym];

        // EVM accumulators. Measured *incrementally*, inside the loop that
        // already has each symbol, rather than by retaining the whole
        // constellation and walking it again: `data_syms` is one symbol's worth
        // and is overwritten on every iteration, and keeping all of them would
        // cost 1512 · n_symbols complex values (~823 KB a frame) for a scalar.
        // Hard decisions come from the LLRs this loop is computing anyway, so
        // the reference point is one allocation-free `dvb_t_map_symbol` per
        // carrier and nothing else.
        let (mut evm_err, mut evm_ref) = (0.0f64, 0.0f64);
        // At most 6 bits per DVB-T symbol (64-QAM); a fixed array keeps the
        // hard-decision buffer off the heap in the innermost loop.
        let mut hard_bits = [0u8; 6];
        // Hoisted out of the per-carrier loop below: this is a link-constant
        // builder flag, and the loop runs 1512 times per symbol.
        let measure_errors = self.measure_errors;
        // Where this frame's symbols begin in the probe's shared buffer. Taken
        // before the demap loop appends anything, so the span recorded at the
        // end covers exactly this frame.
        let sym_start = probe.as_deref().map_or(0, |p| p.symbols.len());

        let mut tps_word: Option<TpsWord> = None;
        for s in 0..n_symbols {
            let off = start + s * sps;
            let freq = match symbol_fft.demod_symbol(&iq[off..]) {
                Some(f) => f,
                None => return Err(DvbTRxError::Incomplete),
            };
            // TPS cells from the raw (pre-equalization) bins — DBPSK is differential
            // and needs no channel estimate.
            let cells: Vec<C32> = tps_bins.iter().map(|&b| freq[b]).collect();
            tps_dec.feed_symbol(&cells);
            if (s + 1) % TPS_SYMBOLS_PER_FRAME == 0 && tps_word.is_none() {
                tps_word = tps_dec.word();
                tps_dec.reset();
            }
            // Equalize from this symbol's phase pilots, then extract the data.
            eq.set_pilot_bins(
                extractor.phase(),
                extractor.current_pilot_bins(),
                extractor.data_bins(),
            );
            eq.process(freq, &mut equalized);
            extractor.extract_symbol(&equalized, &mut data_syms);
            // The probe's constellation: appended as each symbol is extracted,
            // because `data_syms` is one symbol's worth and is overwritten on
            // the next iteration. Frames from one `feed` sit end to end in the
            // probe's buffer, delimited by the spans `push_*` records.
            if let Some(p) = probe.as_deref_mut() {
                p.symbols.extend_from_slice(&data_syms);
            }
            let sym_llrs = &mut llrs[s * bits_per_sym..(s + 1) * bits_per_sym];
            // Two spellings of the same demap loop, chosen once per symbol
            // rather than once per carrier. The unmeasured arm is byte-for-byte
            // the loop that shipped before EVM existed, so a receiver that did
            // not ask for diagnostics pays nothing — not even the branch.
            if measure_errors {
                for (c, &sym) in data_syms.iter().enumerate() {
                    let l = dvb_t_soft_llr(sym, vbits).expect("DVB-T order");
                    sym_llrs[c * vbits..(c + 1) * vbits].copy_from_slice(&l);
                    // EVM against this carrier's own hard decisions — no
                    // knowledge of the transmitted bits, so it works over the
                    // air. Crate-wide LLR convention: positive means bit 0.
                    let hard = &mut hard_bits[..vbits];
                    for (h, &x) in hard.iter_mut().zip(l.iter()) {
                        *h = u8::from(x <= 0.0);
                    }
                    let ideal = dvb_t_map_symbol(hard).expect("DVB-T order");
                    let e = sym - ideal;
                    evm_err += (e.re * e.re + e.im * e.im) as f64;
                    evm_ref += (ideal.re * ideal.re + ideal.im * ideal.im) as f64;
                }
            } else {
                for (c, &sym) in data_syms.iter().enumerate() {
                    let l = dvb_t_soft_llr(sym, vbits).expect("DVB-T order");
                    sym_llrs[c * vbits..(c + 1) * vbits].copy_from_slice(&l);
                }
            }
        }
        // Ratio of error energy to reference energy, in dB. `evm_ref` is zero
        // when EVM was not measured, or for a zero-length frame.
        if evm_ref > 0.0 {
            diagnostics.evm_db = Some((10.0 * (evm_err / evm_ref).log10()) as f32);
        }

        let tps = tps_word.ok_or(DvbTRxError::TpsDecode)?;

        // 3. Payload FEC decode (inverse of the modulator's encode_chain).
        //
        // The payload occupies a PREFIX of the frame. `DvbTFrameMod` stuffs null
        // TS packets until the coded stream reaches the frame's data carriers, so
        // the bits on air are the encode of a stream far longer than `payload_len`
        // implies — for a 68-symbol QPSK frame, 205632 coded bits against the
        // 39180 one 188-byte packet produces. A plain decode only needs the
        // prefix. (The stuffed stream ends on or BEFORE the last data carrier;
        // the few carriers past it repeat its head and are outside every plan
        // here — see `dvb_t_frame_fill`.)
        let payload_packets = payload_len.div_ceil(TS_PACKET_LEN - 1).max(1);
        let payload_ts_bytes = payload_packets * TS_PACKET_LEN;

        // **A measurement needs the whole frame, and a prefix will not do.**
        // Reproducing what the transmitter sent means re-encoding the same bytes
        // it encoded, and with a convolutional outer interleaver no prefix of the
        // re-encode matches: Forney(12,17) draws each output byte from twelve
        // branches at different depths, so the very first coded bits already
        // depend on TS bytes belonging to codewords a prefix decode never
        // recovers. Re-encoding the prefix puts zeros where the transmitter had
        // real data and yields a CBER around 0.25 — a number that looks like a
        // dead link on a noiseless one. (Measured, not assumed: 0.256 before this
        // was understood.)
        //
        // So when any measured rung is asked for, decode the whole frame. The
        // recovered TS is then byte-for-byte what was encoded and the re-encode
        // is exact. It costs roughly 5x the FEC work, which is precisely why it
        // sits behind the same gate as the rungs that need it — the default path
        // still decodes the prefix and is unchanged.
        //
        // `dvb_t_frame_fill` is the modulator's own rule, not a second copy of
        // it: the count it returns is the count that was encoded, and its coded
        // stream ends on or before the last data carrier, so the plan built from
        // it is never longer than the LLRs below.
        let want_truth = self.measure_errors || probe.is_some();
        let n_ts_packets = if want_truth {
            dvb_t_frame_fill_with(params, payload_packets, n_symbols, &cache).n_ts_packets
        } else {
            payload_packets
        };
        let ts_bytes_len = n_ts_packets * TS_PACKET_LEN;
        let plan = block_plan(
            ts_bytes_len,
            CrcKind::None,
            DVB_T_FRAME_OUTER,
            params.inner(),
            DVB_T_FRAME_OUTER_IL,
            InterleaverKind::None,
            &cache,
        );
        // A payload that reached the demapper but did not verify still has a
        // constellation, and a constellation is precisely what an operator looks
        // at when frames stop decoding — so a probing receiver records the
        // symbols with an empty map before reporting the failure.
        let fail = |probe: &mut Option<&mut DvbTRxProbe>| {
            if let Some(p) = probe.as_deref_mut() {
                p.push_undecoded(sym_start, params.constellation(), tps);
            }
            DvbTRxError::PayloadDecode
        };

        let outcome = match decode_chain(
            &llrs,
            &plan,
            CrcKind::None,
            DVB_T_FRAME_OUTER,
            params.inner(),
            DVB_T_FRAME_OUTER_IL,
            InterleaverKind::None,
            ScramblerKind::None,
            ScramblerPos::BeforeOuterFec,
            0,
            &cache,
            DecodeRule::SumProduct,
        ) {
            Ok(o) if o.is_valid() => o,
            _ => return Err(fail(&mut probe)),
        };
        diagnostics.outer_fec_ok = Some(outcome.outer_ok);
        diagnostics.rs_corrected_bytes = outcome.outer_corrected_bytes;
        let mut ts = outcome.bytes;

        // 4. Undo energy dispersal and depacketize.
        if ts.len() < ts_bytes_len {
            return Err(fail(&mut probe));
        }

        // 3b. Optional BER rungs, from a re-encode of what was just recovered.
        //
        // A frame whose outer code verified *is* the ground truth: re-running
        // the encode chain on it reconstructs what the transmitter sent, so
        // comparing that against what arrived at each stage gives a rate rather
        // than a pass/fail flag — and needs no prior knowledge of the payload,
        // which is what makes it work over the air.
        //
        // **The truth reference is `ts`, not `payload`.** DVB-T applies energy
        // dispersal OUTSIDE the coded chain — `decode_chain` is called with
        // `ScramblerKind::None` and step 4 below disperses afterwards — so the
        // bytes the transmitter fed into the chain are the still-dispersed TS.
        // Re-encoding the returned `payload` instead would compile, run, and
        // produce a CBER that is noise. Every argument here mirrors the
        // `decode_chain` call above exactly, for the same reason.
        //
        // Off unless asked for: it costs one encode chain per frame. The probe's
        // correction map is built from the same re-encode, so asking for both
        // pays for it once.
        let stages = want_truth.then(|| {
            encode_chain_stages(
                &ts[..ts_bytes_len],
                CrcKind::None,
                DVB_T_FRAME_OUTER,
                params.inner(),
                DVB_T_FRAME_OUTER_IL,
                InterleaverKind::None,
                ScramblerKind::None,
                ScramblerPos::BeforeOuterFec,
                0,
                &cache,
            )
        });
        // The plan's own coded length, unclamped. The shared frame-fill rule
        // stops the transmitter's coded stream on or before the last data
        // carrier, so `plan.coded_bits <= llrs.len()` holds structurally and
        // there is nothing here to clamp against.
        //
        // `<=`, not `==`: the two are DELIBERATELY unequal on every frame in
        // every mode, by exactly the filler the modulator repeats across the
        // remaining carriers. The clamp this replaced (`.min(llrs.len())`) was
        // load-bearing in the wrong direction — it silently shortened the
        // comparison when the plan overran the LLRs, which is how a frame with
        // 474 untransmitted bits came to report a plausible `inner_ber` of
        // 5.0e-5 on a noiseless link instead of failing visibly.
        //
        // Unclamped, a violation would slice `llrs` out of range below rather
        // than mis-measure. That is the intent — but it is also not reachable
        // from caller input. Overstating `payload_len` moves `payload_packets`
        // by whole 187-byte packets, so the smallest inconsistency already
        // overruns by a packet's coded step (~2176 bits at r3/4), which puts
        // enough erasures in the Forney tail to fail the outer code — and the
        // failure returns above, before this point. There is no band where the
        // plan overruns the LLRs and the frame still decodes.
        debug_assert!(
            !want_truth || plan.coded_bits <= llrs.len(),
            "frame fill must keep the coded stream within the data carriers"
        );
        let coded_bits = plan.coded_bits;
        if let Some(s) = stages.as_ref().filter(|_| self.measure_errors) {
            // CBER: the demapped LLRs hard-decided, against the coded bits as
            // transmitted. Compared in place — materializing the hard decisions
            // would cost a `plan.coded_bits` allocation for a scalar.
            let n = coded_bits.min(s.coded.len());
            if n > 0 {
                let errs = llrs[..n]
                    .iter()
                    .zip(s.coded[..n].iter())
                    .filter(|&(&l, &c)| u8::from(l <= 0.0) != c)
                    .count();
                diagnostics.channel_ber = Some(errs as f32 / n as f32);
            }
            // IBER: what the inner decoder produced, against what it should
            // have. `bit_error_rate` compares over the shorter of the two, so
            // the untrimmed tail of `inner_out_bits` is handled implicitly.
            diagnostics.inner_ber = bit_error_rate(&outcome.inner_out_bits, &s.outer_il_bits);
        }

        // 3c. The probe's correction map: the same comparison the channel BER
        //     collapses to a scalar, kept per bit, plus a third stream saying
        //     what the inner decoder made of each one. Nothing new is measured
        //     — the ground truth is the re-encode above.
        if let (Some(p), Some(s)) = (&mut probe, stages.as_ref()) {
            // What arrived at the demapper, one hard decision per coded bit.
            // The unprobed path never materializes these (it compares LLRs in
            // place); a per-bit map has to.
            p.hard.clear();
            p.hard
                .extend(llrs[..coded_bits].iter().map(|&l| u8::from(l <= 0.0)));
            // What the inner decoder itself decided, re-encoded back into the
            // coded-bit domain. DVB-T configures no inner interleaver and no
            // after-inner scramble, so the generic path's `reencode_inner_output`
            // reduces to the inner encode alone here — the two stages it would
            // otherwise re-apply are both `None` in the `decode_chain` call above.
            let re = inner_encode(params.inner(), &outcome.inner_out_bits, &cache);
            p.estimate.clear();
            p.estimate
                .extend_from_slice(&re[..coded_bits.min(re.len())]);
            p.push_decoded(
                sym_start,
                params.constellation(),
                tps,
                &s.coded[..coded_bits.min(s.coded.len())],
            );
        }
        // Back to the payload's own prefix: when the whole frame was decoded for
        // measurement, everything past this is stuffed null packets.
        ts.truncate(payload_ts_bytes);
        ts_energy_disperse(&mut ts);
        let payload = ts_depacketize(&ts).ok_or(DvbTRxError::PayloadDecode)?;
        let payload = payload
            .get(..payload_len)
            .map(|s| s.to_vec())
            .unwrap_or(payload);

        Ok(DvbTRxFrame {
            payload,
            tps,
            diagnostics,
        })
    }
}
