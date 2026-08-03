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

use super::ofdm::{EqualizerMethod, OfdmEqualizer};
use super::ofdm_frame::decode_chain;
use crate::core::Block;
use crate::dsp::Rotator;
use crate::fec::{CrcKind, DecodeRule, InterleaverKind, ScramblerKind, ScramblerPos};
use crate::modulate::ofdm_frame::{CodecCache, block_plan};
use crate::multicarrier::{CyclicPrefixRemove, FftBlock};
use crate::sync::{dvb_t_gi_sync, dvb_t_integer_cfo};
use crate::waveform::dvb_t::{
    DVB_T_DATA_CARRIERS, DVB_T_FRAME_OUTER, DVB_T_FRAME_OUTER_IL, DVB_T_N_FFT, DvbTFrameParams,
    ScatteredPilotExtractor, dvb_t_soft_llr, tps_carrier_bins,
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

/// The recovered contents of a DVB-T frame: the TS payload and the TPS word read
/// off the carriers.
#[derive(Debug, Clone, PartialEq)]
pub struct DvbTRxFrame {
    /// The recovered TS payload bytes (depacketized, trimmed to `payload_len`).
    pub payload: Vec<u8>,
    /// The transmission parameters recovered from the TPS carriers.
    pub tps: TpsWord,
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
}

impl DvbTFrameDemod {
    /// Builds a demodulator for a link with the given cold-start parameters.
    /// Integer-CFO correction is **off** by default (see
    /// [`with_integer_cfo_correction`](Self::with_integer_cfo_correction)).
    pub fn new(params: DvbTFrameParams) -> Self {
        Self {
            params,
            integer_cfo: false,
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

    /// The transmission parameters this demodulator was built with.
    pub fn params(&self) -> DvbTFrameParams {
        self.params
    }

    /// Whether internal integer-CFO correction is enabled.
    pub fn integer_cfo_correction(&self) -> bool {
        self.integer_cfo
    }

    /// Removes any whole-subcarrier CFO from `iq`, returning the corrected buffer,
    /// or `None` to leave `iq` untouched (correction disabled, acquisition failed,
    /// or the estimate is 0). Aligns to a symbol boundary, accumulates
    /// continual-pilot energy over several symbols, estimates the integer offset,
    /// and rotates the whole buffer by `−k·fs/n_fft`. A frequency rotation commutes
    /// with the timing offset, so the subsequent GI-sync still finds the same
    /// symbol boundary.
    fn integer_cfo_correct(
        &self,
        iq: &[C32],
        n_fft: usize,
        cp_len: usize,
        fs: f32,
    ) -> Option<Vec<C32>> {
        if !self.integer_cfo {
            return None;
        }
        let sps = n_fft + cp_len;
        let acq = dvb_t_gi_sync(iq, n_fft, cp_len, fs, sps)?;
        let mut cpr = CyclicPrefixRemove::new(n_fft, cp_len);
        let mut fft = FftBlock::new(n_fft);
        let mut time = vec![C32::default(); n_fft];
        let mut freq = vec![C32::default(); n_fft];
        let mut accum = vec![C32::default(); n_fft];
        for s in 0..INTEGER_CFO_ACCUM_SYMBOLS {
            let off = acq.start_sample + s * sps;
            if off + n_fft > iq.len() {
                break;
            }
            cpr.process(&iq[off..], &mut time);
            fft.process(&time, &mut freq);
            for (a, &x) in accum.iter_mut().zip(freq.iter()) {
                *a += C32::new(x.norm_sqr(), 0.0);
            }
        }
        let k = dvb_t_integer_cfo(&accum, n_fft, INTEGER_CFO_MAX_BINS).map(|e| e.bins)?;
        if k == 0 {
            return None;
        }
        let mut corrected = vec![C32::default(); iq.len()];
        Rotator::new(-(k as f32) * fs / n_fft as f32, fs).rotate_block(iq, &mut corrected);
        Some(corrected)
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
        let params = self.params;
        let cache = CodecCache::new();
        let base = params.config();
        let n_fft = DVB_T_N_FFT;
        let cp_len = base.carrier_plan.cp_len();
        let sps = n_fft + cp_len;
        let vbits = params.constellation().bits_per_symbol();

        // 0. Optional integer-CFO pre-correction (link-constant, set at
        //    construction). A no-op unless enabled AND a whole-subcarrier offset is
        //    present, so a clean link decodes the original buffer unchanged.
        let corrected = self.integer_cfo_correct(iq, n_fft, cp_len, base.fs);
        let iq: &[C32] = corrected.as_deref().unwrap_or(iq);

        // 1. Acquire the symbol boundary from the cyclic prefix.
        let acq = dvb_t_gi_sync(iq, n_fft, cp_len, base.fs, sps).ok_or(DvbTRxError::Acquisition)?;
        let start = acq.start_sample;
        if iq.len() < start + n_symbols * sps {
            return Err(DvbTRxError::Incomplete);
        }

        // 2. Per-symbol: CP-remove → FFT → equalize (scattered pilots) → extract
        //    data LLRs + TPS cells.
        let mut extractor = ScatteredPilotExtractor::new(params.guard());
        let mut eq = OfdmEqualizer::new(&base, EqualizerMethod::PerSymbolPilotInterp);
        let mut cp_remove = CyclicPrefixRemove::new(n_fft, cp_len);
        let mut fft = FftBlock::new(n_fft);
        let mut tps_dec = TpsDecoder::new();
        let tps_bins = tps_carrier_bins();

        let mut time = vec![C32::default(); n_fft];
        let mut freq = vec![C32::default(); n_fft];
        let mut equalized = vec![C32::default(); n_fft];
        let mut data_syms = vec![C32::default(); DVB_T_DATA_CARRIERS];
        let bits_per_sym = DVB_T_DATA_CARRIERS * vbits;
        let mut llrs = vec![0.0f32; n_symbols * bits_per_sym];

        let mut tps_word: Option<TpsWord> = None;
        for s in 0..n_symbols {
            let off = start + s * sps;
            if cp_remove.process(&iq[off..], &mut time).out_written != n_fft {
                return Err(DvbTRxError::Incomplete);
            }
            fft.process(&time, &mut freq);
            // TPS cells from the raw (pre-equalization) bins — DBPSK is differential
            // and needs no channel estimate.
            let cells: Vec<C32> = tps_bins.iter().map(|&b| freq[b]).collect();
            tps_dec.feed_symbol(&cells);
            if (s + 1) % TPS_SYMBOLS_PER_FRAME == 0 && tps_word.is_none() {
                tps_word = tps_dec.word();
                tps_dec.reset();
            }
            // Equalize from this symbol's phase pilots, then extract the data.
            let pilots = extractor.current_pilot_bins().to_vec();
            let data_bins = extractor.data_bins().to_vec();
            eq.set_pilot_bins(&pilots, &data_bins);
            eq.process(&freq, &mut equalized);
            extractor.extract_symbol(&equalized, &mut data_syms);
            let sym_llrs = &mut llrs[s * bits_per_sym..(s + 1) * bits_per_sym];
            for (c, &sym) in data_syms.iter().enumerate() {
                let l = dvb_t_soft_llr(sym, vbits).expect("DVB-T order");
                sym_llrs[c * vbits..(c + 1) * vbits].copy_from_slice(&l);
            }
        }

        let tps = tps_word.ok_or(DvbTRxError::TpsDecode)?;

        // 3. Payload FEC decode (inverse of the modulator's encode_chain).
        let n_ts_packets = payload_len.div_ceil(TS_PACKET_LEN - 1).max(1);
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
        let (mut ts, ok) = decode_chain(
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
        )
        .map_err(|_| DvbTRxError::PayloadDecode)?;
        if !ok {
            return Err(DvbTRxError::PayloadDecode);
        }

        // 4. Undo energy dispersal and depacketize.
        if ts.len() < ts_bytes_len {
            return Err(DvbTRxError::PayloadDecode);
        }
        ts.truncate(ts_bytes_len);
        ts_energy_disperse(&mut ts);
        let payload = ts_depacketize(&ts).ok_or(DvbTRxError::PayloadDecode)?;
        let payload = payload
            .get(..payload_len)
            .map(|s| s.to_vec())
            .unwrap_or(payload);

        Ok(DvbTRxFrame { payload, tps })
    }
}
