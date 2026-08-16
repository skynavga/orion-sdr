// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/demodulate/ofdm_frame.rs
//
// The OFDM frame (MAC-layer) demodulator: the exact inverse of
// `modulate::ofdm_frame`. The *batch* receiver [`OfdmFrameDemod`] decodes a frame
// at a KNOWN start, and [`OfdmFrameStreamDemod`] handles the unknown-start /
// streaming case. It runs the concatenated COFDM decode chain:
//
//   IQ → soft-demap (LLRs) → inner-deinterleave (LLR) → inner-decode →
//        outer-deinterleave (byte) → outer-decode → descramble → strip CRC
//
// The header is decoded first with the fixed built-in scheme (BPSK + rate-1/2
// LDPC) to recover `mcs_index`/`payload_len`/`sequence_num`/`flags`/seed, then
// the payload is decoded at the MCS the header selected.

use crate::core::Block;
use crate::demodulate::ofdm::{
    EqualizerMethod, OfdmDemod, OfdmEqualizer, OfdmRxFrame, OfdmSoftDemod,
};
use crate::demodulate::ofdm_probe::{OfdmRxProbe, ProbeMeta};
use crate::dsp::Rotator;
use crate::fec::{
    BlockInterleaver, ConvDeinterleaver, ConvInterleaver, CrcKind, DecodeRule, FrameMetadata,
    FramePacket, InnerFec, InterleaverKind, OuterFec, RxError, ScramblerKind, ScramblerPos,
    viterbi_decode_soft_with,
};
use crate::modulate::ofdm::{ConstellationOrder, OfdmConfig};
use crate::modulate::ofdm_frame::{
    BCH_INFO_BITS, BlockPlan, CodecCache, HEADER_CONSTELLATION, HEADER_FIELD_BYTES, HEADER_LDPC,
    McsTable, bits_to_bytes, block_plan, build_scrambler, bytes_to_bits, check_and_strip_crc,
    encode_chain_stages, inner_encode, interleave_bits, scramble_bits, scramble_bytes,
    symbol_config, symbols_for_coded_bits,
};
use crate::multicarrier::{CarrierGrid, GridExtract, SymbolFft};
use crate::sync::{OfdmPreamble, earliest_accepted, ofdm_sync};
use num_complex::Complex32 as C32;
use std::sync::Arc;

/// Soft-demaps `n_symbols` OFDM symbols starting at `iq[0]` into a flat LLR
/// vector (one `f32` per coded bit, `+ ⇒ bit 0`). Returns `None` if `iq` is
/// too short.
///
/// With `equalizer = None` this is the flat-channel path (`OfdmDemod →
/// OfdmSoftDemod`, no per-bin correction) used by the batch entry point. With
/// an equalizer whose channel estimate is already set (from a training
/// symbol), it runs the full `CyclicPrefixRemove → FftBlock → OfdmEqualizer →
/// GridExtract → OfdmSoftDemod` chain, correcting a frequency-selective
/// channel — the streaming receiver's path.
/// `symbol_sink`, when supplied, receives every equalized data-carrier symbol
/// in demap order. EVM is measured by comparing these against the ideal points
/// their own hard decisions map back to, so it needs the constellation-domain
/// symbols the LLRs are derived from — which are otherwise consumed in place.
fn soft_demap(
    base: &OfdmConfig,
    constellation: ConstellationOrder,
    iq: &[C32],
    n_symbols: usize,
    equalizer: Option<&mut OfdmEqualizer>,
    mut symbol_sink: Option<&mut Vec<C32>>,
) -> Option<Vec<f32>> {
    let cfg = symbol_config(base, constellation);
    let sps = cfg.samples_per_ofdm_symbol();
    if iq.len() < n_symbols * sps {
        return None;
    }
    let n_data = cfg.carrier_plan.data_carriers().len();
    let bps = cfg.bits_per_ofdm_symbol();
    let mut soft = OfdmSoftDemod::new(&cfg);
    let mut symbols = vec![C32::default(); n_data];
    let mut llrs = vec![0.0f32; n_symbols * bps];

    match equalizer {
        None => {
            let mut demod = OfdmDemod::new(&cfg);
            let mut in_off = 0;
            let mut out_off = 0;
            for _ in 0..n_symbols {
                let dw = demod.process(&iq[in_off..], &mut symbols);
                if dw.out_written != n_data {
                    return None;
                }
                if let Some(sink) = symbol_sink.as_deref_mut() {
                    sink.extend_from_slice(&symbols);
                }
                let sw = soft.process(&symbols, &mut llrs[out_off..out_off + bps]);
                if sw.out_written != bps {
                    return None;
                }
                in_off += sps;
                out_off += bps;
            }
        }
        Some(eq) => {
            let n_fft = cfg.carrier_plan.n_fft();
            let cp_len = cfg.carrier_plan.cp_len();
            let grid = CarrierGrid::from_plan(&cfg.carrier_plan);
            let mut symbol_fft =
                SymbolFft::new(n_fft, cp_len).with_window_backoff(base.rx_window_backoff);
            let mut grid_extract = GridExtract::new(grid);
            let mut equalized = vec![C32::default(); n_fft];
            // Every symbol is extracted before any is demapped: the phase
            // tracker below needs the whole payload to fit a ramp across it,
            // and the LLRs must come from the *corrected* symbols.
            let mut all = vec![C32::default(); n_symbols * n_data];
            let mut in_off = 0;
            for k in 0..n_symbols {
                let freq = symbol_fft.demod_symbol(&iq[in_off..])?;
                if eq.process(freq, &mut equalized).out_written != n_fft {
                    return None;
                }
                if grid_extract.process(&equalized, &mut symbols).out_written != n_data {
                    return None;
                }
                all[k * n_data..(k + 1) * n_data].copy_from_slice(&symbols);
                in_off += sps;
            }

            remove_common_phase_error(&cfg, &mut all, n_symbols);

            let mut out_off = 0;
            for k in 0..n_symbols {
                let block = &all[k * n_data..(k + 1) * n_data];
                if let Some(sink) = symbol_sink.as_deref_mut() {
                    sink.extend_from_slice(block);
                }
                let sw = soft.process(block, &mut llrs[out_off..out_off + bps]);
                if sw.out_written != bps {
                    return None;
                }
                out_off += bps;
            }
        }
    }
    Some(llrs)
}

/// Minimum payload length, in OFDM symbols, worth running phase tracking over.
///
/// Below this the accumulated rotation is negligible and a two-point ramp fit
/// is noise. The frame header (one or two symbols, immediately after the
/// training symbol the channel was estimated from) falls under it by design.
const CPE_MIN_SYMBOLS: usize = 4;

/// Removes the residual **common phase error** accumulated across a frame's
/// symbols, in place.
///
/// **Why this is not optional.** The Schmidl & Cox carrier estimate has
/// variance, and `TrainingSymbolHold` measures the channel once from the
/// training symbol and holds it for the whole frame — nothing revisits it. A
/// residual offset `e` therefore integrates to `2*pi*e*T` of constellation
/// rotation by the end of a frame of duration `T`, and the receiver never sees
/// it happen. A few Hz of estimation error is already tens of degrees by the
/// last symbol of a 50 ms frame, and QPSK's decision boundary is 45.
///
/// The concatenated FEC itself holds FER = 0 far below that (`snr::cofdm_fer`,
/// batch demodulator at a known start), so without this the streaming receiver
/// gives away link budget the FEC already paid for — and the resulting errors
/// look exactly like an FEC cliff.
///
/// **A better initial estimate is not the fix.** S&C's variance is set by the
/// preamble's total correlated energy, so correlating at lag `3L` instead of
/// `L` triples the phase-to-frequency scaling and correlates a third as many
/// samples — measured 10.95 Hz against 10.96 Hz, a gain of 1.0x. Nothing is won
/// without spending more preamble.
///
/// **How.** Two passes over the equalized data symbols:
///
/// 1. A decision-directed tracking loop. Each symbol is de-rotated by the
///    phase predicted from the symbols before it, demapped, and its hard
///    decisions remapped to ideal constellation points; the residual
///    `arg(sum(y * conj(y_ideal)))` drives a second-order loop. Prediction is
///    what makes this work at all: a decision-directed estimate is only valid
///    while decisions are, and they stop being valid past 45 degrees for QPSK
///    — the very rotation this exists to remove. Tracking incrementally keeps
///    the residual *at the point of decision* well under a degree.
/// 2. A least-squares fit of the accumulated phase against symbol index. A
///    carrier offset is a straight line by construction, so fitting one pools
///    every symbol's estimate into two parameters instead of trusting each
///    alone, and removes the loop's start-up transient from the correction
///    actually applied.
///
/// Correcting per symbol rather than per frame moves the rotation budget from
/// the frame duration to one symbol. Measured end to end on the standard COFDM
/// test plan (53.8 ms frame), frame error rate against the known transmitted
/// payload, 60 trials per point:
///
/// | In-band SNR | FER without | FER with |
/// | --- | --- | --- |
/// | 20 dB | 0.083 | **0.000** |
/// | 15 dB | 0.350 | **0.017** |
/// | 12 dB | 0.550 | **0.050** |
/// | 10 dB | 0.717 | **0.133** |
/// | 8 dB | 0.783 | **0.367** |
///
/// Error-free reception starts at 20 dB rather than 25, and every point below
/// it improves several-fold. Reproduce with `snr::cofdm_stream_fer`.
///
fn remove_common_phase_error(cfg: &OfdmConfig, symbols: &mut [C32], n_symbols: usize) {
    let n_data = cfg.carrier_plan.data_carriers().len();
    if n_symbols < CPE_MIN_SYMBOLS || n_data == 0 || symbols.len() < n_symbols * n_data {
        return;
    }
    // Loop gains: fast enough to lock inside a short payload, slow enough that
    // per-symbol estimator noise does not drive the prediction. The fit below
    // is what sets the accuracy of the applied correction, so these only have
    // to keep the decisions valid.
    const ALPHA: f32 = 0.5;
    const BETA: f32 = 0.05;

    let mut soft = OfdmSoftDemod::new(cfg);
    let mut mapper = crate::modulate::ofdm::ideal_symbol_mapper(cfg.constellation);
    let bps = cfg.bits_per_ofdm_symbol();
    let mut llrs = vec![0.0f32; bps];
    let mut bits = vec![0u8; bps];
    let mut ideal = vec![C32::default(); n_data];
    let mut rotated = vec![C32::default(); n_data];
    let mut measured = vec![0.0f32; n_symbols];

    let (mut phase, mut freq) = (0.0f32, 0.0f32);
    for k in 0..n_symbols {
        let block = &symbols[k * n_data..(k + 1) * n_data];
        let (sin, cos) = (-phase).sin_cos();
        let p = C32::new(cos, sin);
        for (dst, src) in rotated.iter_mut().zip(block) {
            *dst = src * p;
        }
        if soft.process(&rotated, &mut llrs).out_written != bps {
            return;
        }
        for (b, l) in bits.iter_mut().zip(llrs.iter()) {
            // Crate-wide LLR convention: positive means bit 0 is more likely.
            *b = u8::from(*l <= 0.0);
        }
        if mapper.process(&bits, &mut ideal).out_written != n_data {
            return;
        }
        let mut acc = C32::default();
        for (y, r) in rotated.iter().zip(ideal.iter()) {
            acc += y * r.conj();
        }
        let err = if acc.re == 0.0 && acc.im == 0.0 {
            0.0
        } else {
            acc.im.atan2(acc.re)
        };
        // The total rotation observed on this symbol, before the loop moves on.
        measured[k] = phase + err;
        phase += freq + ALPHA * err;
        freq += BETA * err;
    }

    // Least-squares line through (k, measured[k]).
    let n = n_symbols as f32;
    let sum_k = n * (n - 1.0) / 2.0;
    let sum_k2 = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;
    let sum_y: f32 = measured.iter().sum();
    let sum_ky: f32 = measured.iter().enumerate().map(|(k, y)| k as f32 * y).sum();
    let denom = n * sum_k2 - sum_k * sum_k;
    if denom.abs() <= f32::EPSILON {
        return;
    }
    let slope = (n * sum_ky - sum_k * sum_y) / denom;
    let intercept = (sum_y - slope * sum_k) / n;

    for k in 0..n_symbols {
        let (sin, cos) = (-(intercept + slope * k as f32)).sin_cos();
        let p = C32::new(cos, sin);
        for c in &mut symbols[k * n_data..(k + 1) * n_data] {
            *c *= p;
        }
    }
}

/// Scattered-pilot variant of [`soft_demap`] for DVB-T: demaps `n_symbols` OFDM
/// symbols through the four-phase grid rotation (`extractor`). For each symbol it
/// runs `CyclicPrefixRemove → FftBlock`, installs that symbol's phase-`l`
/// continual+scattered+TPS pilot set on a per-symbol-interpolating equalizer
/// (`OfdmEqualizer::set_pilot_bins` + `PerSymbolPilotInterp`), equalizes,
/// extracts the phase-`l` data bins, and soft-demaps. The `extractor`'s phase
/// counter carries across calls, so a frame's header-then-payload symbols form
/// one continuous rotation matching the TX (`l = 0` at the first symbol after
/// [`ScatteredPilotExtractor::reset`]).
///
/// This is the conformant DVB-T channel-estimation path (dense scattered pilots
/// per symbol), replacing the Phase-1 training-symbol hold. Returns `None` if
/// `iq` is too short.
fn soft_demap_scattered(
    base: &OfdmConfig,
    constellation: ConstellationOrder,
    iq: &[C32],
    n_symbols: usize,
    extractor: &mut crate::waveform::dvb_t::ScatteredPilotExtractor,
) -> Option<Vec<f32>> {
    let cfg = symbol_config(base, constellation);
    let sps = cfg.samples_per_ofdm_symbol();
    if iq.len() < n_symbols * sps {
        return None;
    }
    let n_fft = cfg.carrier_plan.n_fft();
    let cp_len = cfg.carrier_plan.cp_len();
    let n_data = extractor.num_data_carriers();
    let vbits = constellation.bits_per_symbol();
    let bps = n_data * vbits;

    // Payload symbols on a DVB-T constellation get the DVB-T-exact soft LLRs
    // (Figure-9a bit assignment); a BPSK header block uses the generic demapper.
    let dvb_t_llr = crate::waveform::dvb_t::is_dvb_t_constellation(constellation);
    let mut soft = OfdmSoftDemod::new(&cfg);
    // A per-symbol-interpolating equalizer; its pilot set is re-installed for
    // each symbol's phase before `process`.
    let mut eq = OfdmEqualizer::new(&cfg, EqualizerMethod::PerSymbolPilotInterp);
    let mut symbol_fft = SymbolFft::new(n_fft, cp_len).with_window_backoff(cfg.rx_window_backoff);
    let mut equalized = vec![C32::default(); n_fft];
    let mut symbols = vec![C32::default(); n_data];
    let mut llrs = vec![0.0f32; n_symbols * bps];

    let mut in_off = 0;
    let mut out_off = 0;
    for _ in 0..n_symbols {
        let freq = symbol_fft.demod_symbol(&iq[in_off..])?;
        // Install this symbol's phase-`l` pilots (bins + known TX values) and the
        // phase's data bins to interpolate across, then equalize from them.
        let pilots = extractor.current_pilot_bins().to_vec();
        let data_bins: Vec<usize> = extractor.data_bins().to_vec();
        eq.set_pilot_bins(&pilots, &data_bins);
        if eq.process(freq, &mut equalized).out_written != n_fft {
            return None;
        }
        extractor.extract_symbol(&equalized, &mut symbols);
        let sym_llrs = &mut llrs[out_off..out_off + bps];
        if dvb_t_llr {
            for (c, &sym) in symbols.iter().enumerate() {
                let l = crate::waveform::dvb_t::dvb_t_soft_llr(sym, vbits).expect("DVB-T order");
                sym_llrs[c * vbits..(c + 1) * vbits].copy_from_slice(&l);
            }
        } else {
            let sw = soft.process(&symbols, sym_llrs);
            if sw.out_written != bps {
                return None;
            }
        }
        in_off += sps;
        out_off += bps;
    }
    Some(llrs)
}

/// Inverse of the interleaver, in the LLR (`f32`) domain.
fn deinterleave_llrs(il: InterleaverKind, llrs: &[f32]) -> Vec<f32> {
    match il {
        InterleaverKind::None => llrs.to_vec(),
        InterleaverKind::Block { rows, cols } => {
            let block = rows * cols;
            let bi = BlockInterleaver::new(rows, cols);
            let mut out = Vec::with_capacity(llrs.len());
            let mut restored = vec![0.0f32; block]; // reused across full chunks
            for chunk in llrs.chunks(block) {
                if chunk.len() < block {
                    out.extend_from_slice(chunk);
                    continue;
                }
                bi.deinterleave(chunk, &mut restored);
                out.extend_from_slice(&restored);
            }
            out
        }
        // The Forney interleaver is byte-domain (DVB-T's *outer* interleaver); it
        // is never configured as the inner (LLR-domain) interleaver. Pass through
        // so a mis-configuration degrades gracefully; the TX side likewise only
        // applies it byte-domain.
        InterleaverKind::Convolutional { .. } => {
            debug_assert!(false, "Convolutional interleaver is byte-domain only");
            llrs.to_vec()
        }
    }
}

/// Inverse of the outer interleaver, in the hard-bit (`u8`) domain.
fn deinterleave_bits(il: InterleaverKind, bits: &[u8]) -> Vec<u8> {
    match il {
        InterleaverKind::None => bits.to_vec(),
        InterleaverKind::Block { rows, cols } => {
            let block = rows * cols;
            let bi = BlockInterleaver::new(rows, cols);
            let mut out = Vec::with_capacity(bits.len());
            let mut restored = vec![0u8; block]; // reused across full chunks
            for chunk in bits.chunks(block) {
                if chunk.len() < block {
                    out.extend_from_slice(chunk);
                    continue;
                }
                bi.deinterleave(chunk, &mut restored);
                out.extend_from_slice(&restored);
            }
            out
        }
        InterleaverKind::Convolutional { branches, depth } => {
            // Frame-mode inverse of `interleave_bits`'s Convolutional arm. The
            // interleaved bit stream is `(n_padded + D)` whole bytes, `D` =
            // round-trip delay. Deinterleave the whole thing; the recovered
            // original bytes start at output offset `D` (the deinterleaver's
            // startup delay) and run for `n_padded`.
            let d = ConvInterleaver::new(branches, depth).roundtrip_delay();
            let total = bits.len() / 8;
            if total <= d {
                return Vec::new();
            }
            let n_padded = total - d;
            let bytes = bits_to_bytes(&bits[..total * 8]);
            let mut di = ConvDeinterleaver::new(branches, depth);
            let deint = di.feed(&bytes);
            bytes_to_bits(&deint[d..d + n_padded])
        }
    }
}

/// Inner-decodes an LLR stream into hard info bits (mirroring `inner_encode`).
/// `info_len` is the number of information bits the inner code protects (needed
/// by the convolutional Viterbi, which is variable-rate). Returns the info bits
/// and whether every block converged.
fn inner_decode(
    inner: InnerFec,
    coded_llrs: &[f32],
    info_len: usize,
    cache: &CodecCache,
    ldpc_rule: DecodeRule,
) -> (Vec<u8>, bool) {
    match inner {
        InnerFec::None => {
            // Hard-decide the LLRs directly.
            (
                coded_llrs.iter().map(|&l| u8::from(l <= 0.0)).collect(),
                true,
            )
        }
        InnerFec::Ldpc(code) => {
            let ldpc = cache.ldpc(code);
            let n = ldpc.n();
            let mut info = Vec::new();
            let mut all_ok = true;
            for chunk in coded_llrs.chunks(n) {
                if chunk.len() < n {
                    all_ok = false;
                    break;
                }
                let (msg, unsat) = ldpc.decode_soft_with(chunk, 50, ldpc_rule);
                if unsat != 0 {
                    all_ok = false;
                }
                info.extend_from_slice(&msg);
            }
            (info, all_ok)
        }
        InnerFec::Convolutional { rate, code } => {
            // Soft Viterbi over the whole block; the outer code / CRC below
            // decides success, so no per-block convergence flag here.
            let info = viterbi_decode_soft_with(code, coded_llrs, info_len, rate);
            (info, true)
        }
    }
}

/// Outer-decodes hard bits into message bits, fragmenting into shortened-BCH
/// codeword blocks (mirroring `outer_encode`). Returns the message bits and
/// whether every block decoded.
fn outer_decode(outer: OuterFec, coded_bits: &[u8], cache: &CodecCache) -> (Vec<u8>, bool) {
    match outer {
        OuterFec::None => (coded_bits.to_vec(), true),
        OuterFec::Bch { t } => {
            let code = cache.bch(t, BCH_INFO_BITS);
            let n = code.n();
            let mut msg = Vec::new();
            let mut all_ok = true;
            for chunk in coded_bits.chunks(n) {
                if chunk.len() < n {
                    all_ok = false;
                    break;
                }
                match code.decode(chunk) {
                    Ok(block) => msg.extend_from_slice(&block),
                    Err(_) => {
                        all_ok = false;
                        // Fall back to the systematic prefix so downstream CRC
                        // can still run (and fail) rather than aborting here.
                        msg.extend_from_slice(&chunk[..code.k()]);
                    }
                }
            }
            (msg, all_ok)
        }
        OuterFec::ReedSolomon { n, n_parity } => {
            // Byte-domain: pack coded bits to bytes, decode each n-byte codeword.
            let rs = cache.rs(n, n_parity);
            let coded_bytes = bits_to_bytes(coded_bits);
            let mut msg_bytes = Vec::new();
            let mut all_ok = true;
            for chunk in coded_bytes.chunks(n) {
                if chunk.len() < n {
                    all_ok = false;
                    break;
                }
                match rs.decode(chunk) {
                    Ok(block) => msg_bytes.extend_from_slice(&block),
                    Err(_) => {
                        all_ok = false;
                        msg_bytes.extend_from_slice(&chunk[..rs.k()]);
                    }
                }
            }
            (bytes_to_bits(&msg_bytes), all_ok)
        }
    }
}

/// The outcome of decoding one logical block: the recovered bytes plus each
/// stage's success, reported **separately**.
///
/// The concatenated scheme has two independent decoders, and folding them into
/// one flag destroys the only signal that distinguishes a marginal link from a
/// failing one. Errors the inner code corrects never reach the outer code, so
/// `inner_ok == false` with `outer_ok == true` is a link running hot but still
/// delivering — precisely the state a pre-FEC error rate is meant to surface,
/// and indistinguishable from success once folded.
///
/// Use [`is_valid`](Self::is_valid) to decide whether to accept the block;
/// the individual flags are diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainOutcome {
    /// The recovered info bytes, CRC stripped.
    pub bytes: Vec<u8>,
    /// Every inner-FEC block converged. Always `true` for
    /// [`InnerFec::None`], and for the convolutional arm, whose soft Viterbi
    /// has no per-block convergence flag — the outer code and CRC decide.
    pub inner_ok: bool,
    /// Every outer-FEC block decoded. Always `true` for [`OuterFec::None`].
    pub outer_ok: bool,
    /// The block's CRC checked.
    pub crc_ok: bool,
    /// Whether the configuration provides a CRC at all. Without this,
    /// `crc_ok` is ambiguous: [`CrcKind::None`] reports `true` because there
    /// was nothing to fail, not because anything was verified.
    pub crc_present: bool,
    /// Whether the configuration provides an outer code at all, on the same
    /// reasoning as [`crc_present`](Self::crc_present).
    pub outer_present: bool,
    /// What the inner decoder produced, **untrimmed** — every bit it decided,
    /// including the zero-padding tail of the final codeword that the block
    /// plan discards before the outer decoder sees it.
    ///
    /// Kept rather than dropped so a caller can compare it against a re-encode
    /// of the recovered message and obtain a post-inner-FEC bit error *rate*
    /// instead of a pass/fail flag. Costs nothing: the vector is already
    /// allocated, and is moved out rather than copied.
    ///
    /// Untrimmed because re-encoding a *trimmed* copy zero-pads that tail back
    /// to zero, which silently asserts the decoder got the padding right. For a
    /// 184-byte payload on the default ladder that is 168 bits of 5120 — enough
    /// to paint a real decoder error `Clean` in a correction map, and far too
    /// few to notice in one. Consumers that want only the bits the outer
    /// decoder saw take `[..plan.outer_il_bits]`; [`bit_error_rate`] compares
    /// over the shorter of its two inputs, so the trim is implicit there.
    pub inner_out_bits: Vec<u8>,
}

impl ChainOutcome {
    /// Whether the recovered bytes can be trusted, judged by the **strongest
    /// end-to-end check the configuration actually provides**.
    ///
    /// `inner_ok` is deliberately not part of this. It reports whether the
    /// inner decoder's parity checks converged — how hard it worked, not
    /// whether the result is right. Requiring it discards frames whose payload
    /// is verifiably correct: measured across a noise sweep, a CRC-carrying
    /// link delivers byte-exact payloads with `inner_ok == false` over a wide
    /// band, and rejecting those costs real sensitivity for nothing.
    ///
    /// The precedence:
    ///
    /// - **A CRC decides on its own.** It is computed over the recovered
    ///   payload end to end, so passing it means the bytes are right whatever
    ///   the stages beneath did — including an outer decoder that reported a
    ///   block it could not correct.
    /// - **Otherwise the outer code decides.** DVB-T carries no CRC
    ///   ([`CrcKind::None`]), so its Reed–Solomon stage is the integrity
    ///   check; RS reports failure when it cannot correct a codeword.
    /// - **Otherwise `inner_ok` is all there is.** A link with neither a CRC
    ///   nor an outer code has only the inner decoder's convergence to go on,
    ///   and dropping it there would accept anything.
    pub fn is_valid(&self) -> bool {
        if self.crc_present {
            self.crc_ok
        } else if self.outer_present {
            self.outer_ok
        } else {
            self.inner_ok
        }
    }
}

/// Decodes one logical block's coded LLRs back to its info bytes, checking the
/// CRC — the exact inverse of `modulate::ofdm_frame::encode_chain`. Public so
/// per-standard frame assemblers (e.g. `waveform::dvb_t_frame`) reuse the shared
/// FEC decode rather than duplicating it.
#[allow(clippy::too_many_arguments)]
pub fn decode_chain(
    coded_llrs: &[f32],
    plan: &BlockPlan,
    crc: CrcKind,
    outer: OuterFec,
    inner: InnerFec,
    outer_il: InterleaverKind,
    inner_il: InterleaverKind,
    scrambler: ScramblerKind,
    scrambler_pos: ScramblerPos,
    per_frame_seed: u32,
    cache: &CodecCache,
    ldpc_rule: DecodeRule,
) -> Result<ChainOutcome, RxError> {
    // 1. Trim to the exact coded-bit count, then invert the after-inner
    //    scramble (bit domain) if configured.
    let mut llrs = coded_llrs.to_vec();
    llrs.truncate(plan.coded_bits);

    // After-inner scrambling was applied to hard bits; to invert in the LLR
    // domain we flip the LLR sign where the PN bit is 1 (XOR by 1 negates the
    // bit ⇒ negate the LLR).
    let sc = build_scrambler(scrambler, per_frame_seed);
    if scrambler_pos == ScramblerPos::AfterInnerFec
        && let Some(ref s) = sc
    {
        apply_pn_to_llrs(s, &mut llrs);
    }

    // 2. Inner deinterleave (LLR), then inner decode.
    let inner_de = deinterleave_llrs(inner_il, &llrs);
    let inner_de = &inner_de[..plan.inner_coded_bits.min(inner_de.len())];
    let (outer_il_bits, inner_ok) =
        inner_decode(inner, inner_de, plan.outer_il_bits, cache, ldpc_rule);

    // 3. Outer deinterleave (byte/bit domain), then outer decode. The plan trims
    //    the final codeword's zero padding here — as a borrow, so
    //    `ChainOutcome::inner_out_bits` below still carries every bit the inner
    //    decoder decided (see its docs for why that matters).
    let trimmed = &outer_il_bits[..plan.outer_il_bits.min(outer_il_bits.len())];
    let outer_de = deinterleave_bits(outer_il, trimmed);
    let outer_de = &outer_de[..plan.outer_coded_bits.min(outer_de.len())];
    let (mut framed_bits, outer_ok) = outer_decode(outer, outer_de, cache);
    framed_bits.truncate(plan.framed_bytes * 8);

    if framed_bits.len() < plan.framed_bytes * 8 {
        return Err(RxError::MalformedHeader);
    }
    let mut framed = bits_to_bytes(&framed_bits);

    // 4. Invert the before-outer scramble (byte domain — the whitener is
    //    self-inverse, so the same call descrambles; handles DVB-T energy
    //    dispersal and the generic additive LFSR).
    if scrambler_pos == ScramblerPos::BeforeOuterFec {
        scramble_bytes(scrambler, per_frame_seed, &mut framed);
    }

    // 5. Strip and check the CRC.
    let (bytes, crc_ok) = check_and_strip_crc(crc, &framed).ok_or(RxError::MalformedHeader)?;
    Ok(ChainOutcome {
        bytes,
        inner_ok,
        outer_ok,
        crc_ok,
        crc_present: crc != CrcKind::None,
        outer_present: outer != OuterFec::None,
        // `deinterleave_bits` only borrowed this, so it moves out here.
        inner_out_bits: outer_il_bits,
    })
}

/// Re-encodes the inner decoder's own output back into the coded-bit domain —
/// steps 4 and 5 of `encode_chain_stages`, run on what the decoder decided
/// rather than on the recovered payload — writing into `out` (cleared and
/// refilled, so its capacity is reused across frames).
///
/// This is the third stream a correction map needs: comparing it against the
/// re-encode of the CRC-verified payload says which bits the inner decoder got
/// right, in the same index space the received hard decisions live in.
///
/// **Why re-encode rather than read the decoder's internal codeword.**
/// `Ldpc::decode_soft_with` holds a full n-bit hard-decision vector and returns
/// only its systematic prefix; exposing the rest would be cheaper. But the
/// convolutional arm has no codeword to expose — soft Viterbi produces
/// information bits and nothing else — so an accessor-based map would render on
/// LDPC and come up blank on DVB-T, whose inner code is `ConvCode::DvbK7`. The
/// re-encode costs one inner encode per frame and works on both.
fn reencode_inner_output(
    cfg: &OfdmConfig,
    inner: InnerFec,
    inner_out: &[u8],
    coded_bits: usize,
    per_frame_seed: u32,
    cache: &CodecCache,
    out: &mut Vec<u8>,
) {
    let inner_bits = inner_encode(inner, inner_out, cache);
    out.clear();
    match cfg.inner_interleaver {
        InterleaverKind::None => out.extend_from_slice(&inner_bits),
        il => out.extend_from_slice(&interleave_bits(il, &inner_bits)),
    }
    if cfg.scrambler_pos == ScramblerPos::AfterInnerFec
        && let Some(ref s) = build_scrambler(cfg.scrambler, per_frame_seed)
    {
        scramble_bits(s, out);
    }
    out.truncate(coded_bits);
}

/// Applies a PN sequence to LLRs by negating each LLR whose PN bit is 1.
fn apply_pn_to_llrs(s: &crate::fec::PnScrambler, llrs: &mut [f32]) {
    // The scrambler XORs bits; recover the PN bit-stream by scrambling a
    // zeroed byte buffer of the right length, then negate LLRs at PN==1.
    let n_bytes = llrs.len().div_ceil(8);
    let mut pn = vec![0u8; n_bytes];
    s.scramble(&mut pn);
    let pn_bits = bytes_to_bits(&pn);
    for (l, &p) in llrs.iter_mut().zip(pn_bits.iter()) {
        if p != 0 {
            *l = -*l;
        }
    }
}

/// Distinguishes "waiting for more samples" from a genuine decode failure, so
/// the streaming receiver can hold a partial frame rather than mis-report it.
enum BodyError {
    /// Not enough buffered samples for the header or the (now-known-length)
    /// payload — hold and retry after more input.
    Incomplete,
    /// A real decode failure (bad header CRC, payload CRC, or FEC).
    Failed(RxError),
}

/// Fraction of positions where two bit-streams differ, over their common
/// length. `None` if either is empty.
fn bit_error_rate(a: &[u8], b: &[u8]) -> Option<f32> {
    let n = a.len().min(b.len());
    if n == 0 {
        return None;
    }
    let errs = a[..n]
        .iter()
        .zip(b[..n].iter())
        .filter(|(x, y)| x != y)
        .count();
    Some(errs as f32 / n as f32)
}

/// Per-frame working buffers reused across the frames one receiver decodes.
///
/// Both were allocated fresh on every call before, probing or not. Moving them
/// here is a tidy-up rather than a speed-up — `decode_chain` still copies the
/// LLRs, `deinterleave_llrs` allocates, `inner_decode` allocates, and the decode
/// path stays far from allocation-free. Its real purpose is that the probe's
/// symbol buffer has to be reused anyway, so having the EVM path share the same
/// sink is cheaper than maintaining two.
#[derive(Debug, Clone, Default)]
struct FrameScratch {
    /// Equalized payload symbols, in demap order — the sink `soft_demap` fills
    /// when nothing is probing.
    symbols: Vec<C32>,
    /// The payload LLRs' hard decisions. EVM needs one per coded bit; the
    /// channel BER and the correction map need the first `coded_bits` of them,
    /// which is the same vector rather than a second one.
    hard: Vec<u8>,
}

/// What one frame body yielded: the packet, how many samples it consumed, and
/// the per-stage measurements taken along the way.
struct DecodedBody {
    packet: FramePacket,
    /// IQ samples the header+payload occupied, so a streaming caller can
    /// advance its buffer.
    consumed: usize,
    /// Payload EVM, measured against its own hard decisions before the FEC
    /// stages consume the LLRs.
    evm_db: Option<f32>,
    /// The payload's inner- and outer-FEC convergence, kept apart — see
    /// [`ChainOutcome`].
    inner_ok: bool,
    outer_ok: bool,
    /// Bit error rate at the channel's output, i.e. the inner decoder's input.
    channel_ber: Option<f32>,
    /// Bit error rate at the inner decoder's output, before the outer decoder.
    inner_ber: Option<f32>,
}

/// Decodes a frame body (header + payload) from `iq[0]` — the first sample
/// AFTER the preamble+training, already CFO-corrected. When
/// `channel_estimate` is `Some(n_fft freq bins)` the soft-demap equalizes each
/// symbol against it (multipath); `None` is the flat-channel path.
///
/// Returns a [`DecodedBody`], or a [`BodyError`] distinguishing "incomplete"
/// from a genuine failure.
///
/// `scratch` holds the per-frame working buffers (see [`FrameScratch`]). When
/// `probe` is `Some`, the payload's equalized symbols are appended to it instead
/// of to the scratch, and a decoded frame also gets a per-coded-bit correction
/// map — an observation of this decode, never an input to it.
#[allow(clippy::too_many_arguments)]
fn decode_frame_body(
    cfg: &OfdmConfig,
    mcs_table: &McsTable,
    iq: &[C32],
    channel_estimate: Option<&[C32]>,
    cache: &CodecCache,
    measure_ber: bool,
    scratch: &mut FrameScratch,
    probe: Option<&mut OfdmRxProbe>,
) -> Result<DecodedBody, BodyError> {
    // The scattered-pilot path collects no symbols — DVB-T frames are decoded by
    // `waveform::dvb_t_frame`, which carries its own diagnostics — so a
    // scattered link reports *no* probe frames rather than symbol-less ones. A
    // record with an empty constellation would read as "nothing arrived" instead
    // of "not measured here".
    let mut probe = probe.filter(|_| !cfg.dvb_t_scattered);
    let mut cursor = 0usize;

    // Builds a fresh equalizer for `constellation` carrying the shared channel
    // estimate, or `None` for the flat path.
    let make_eq = |constellation: ConstellationOrder| -> Option<OfdmEqualizer> {
        channel_estimate.map(|est| {
            let symcfg = symbol_config(cfg, constellation);
            let mut eq = OfdmEqualizer::new(&symcfg, EqualizerMethod::TrainingSymbolHold);
            eq.estimate_from_training_symbol(est);
            eq
        })
    };

    // For a DVB-T scattered-pilot link, one grid-rotation extractor spans the
    // whole frame body (header then payload) so the RX symbol phase matches the
    // TX (`l = 0` at the first header symbol). `None` for every other link, which
    // takes the static-grid `soft_demap`.
    let mut scattered = cfg.dvb_t_scattered.then(|| {
        let guard =
            crate::waveform::dvb_t::GuardInterval::from_cp_len_2k(cfg.carrier_plan.cp_len())
                .expect("DVB-T scattered link requires a 2K guard interval");
        crate::waveform::dvb_t::ScatteredPilotExtractor::new(guard)
    });

    // Soft-demaps `n_sym` symbols at `iq[off..]` through either the rotating
    // scattered grid (DVB-T) or the static plan, whichever the config selects.
    // `eq` is only consulted on the static path.
    let mut demap = |constellation: ConstellationOrder,
                     iq: &[C32],
                     off: usize,
                     n_sym: usize,
                     eq: Option<&mut OfdmEqualizer>,
                     sink: Option<&mut Vec<C32>>|
     -> Option<Vec<f32>> {
        match scattered.as_mut() {
            // The scattered path does not collect symbols: DVB-T frames are
            // decoded by `waveform::dvb_t_frame`, which has its own diagnostics.
            Some(x) => soft_demap_scattered(cfg, constellation, &iq[off..], n_sym, x),
            None => soft_demap(cfg, constellation, &iq[off..], n_sym, eq, sink),
        }
    };

    // 1. Header (only OrionSdr prepends a decodable header block here).
    let (metadata, per_frame_seed, payload_len) = if cfg.header_format.has_header_block() {
        let hplan = block_plan(
            HEADER_FIELD_BYTES,
            cfg.header_crc,
            OuterFec::None,
            InnerFec::Ldpc(HEADER_LDPC),
            InterleaverKind::None,
            InterleaverKind::None,
            cache,
        );
        let n_sym = symbols_for_coded_bits(cfg, HEADER_CONSTELLATION, hplan.coded_bits);
        let mut eq = make_eq(HEADER_CONSTELLATION);
        // Too few samples for the header ⇒ incomplete, not malformed.
        let llrs = demap(HEADER_CONSTELLATION, iq, cursor, n_sym, eq.as_mut(), None)
            .ok_or(BodyError::Incomplete)?;
        let header = decode_chain(
            &llrs,
            &hplan,
            cfg.header_crc,
            OuterFec::None,
            InnerFec::Ldpc(HEADER_LDPC),
            InterleaverKind::None,
            InterleaverKind::None,
            ScramblerKind::None,
            ScramblerPos::BeforeOuterFec,
            0,
            cache,
            // The header is decoded first to learn the MCS and must be as robust
            // as possible, so it always uses exact sum-product regardless of the
            // payload's configured rule.
            DecodeRule::SumProduct,
        )
        .map_err(BodyError::Failed)?;
        if !header.is_valid() {
            return Err(BodyError::Failed(RxError::HeaderCrcMismatch));
        }
        let fields = header.bytes;
        if fields.len() < HEADER_FIELD_BYTES {
            return Err(BodyError::Failed(RxError::MalformedHeader));
        }
        let mcs_index = fields[0];
        let payload_len = u32::from_be_bytes([fields[1], fields[2], fields[3], fields[4]]) as usize;
        let sequence_num = u32::from_be_bytes([fields[5], fields[6], fields[7], fields[8]]);
        let flags = fields[9];
        let seed = u32::from_be_bytes([fields[10], fields[11], fields[12], fields[13]]);

        let sps = symbol_config(cfg, HEADER_CONSTELLATION).samples_per_ofdm_symbol();
        cursor += n_sym * sps;
        (
            FrameMetadata {
                sequence_num,
                mcs_index,
                flags,
            },
            seed,
            payload_len,
        )
    } else {
        // NoHeader / DvbTps: this generic entry point has no in-band header to
        // read the MCS/length from. DvbTps frames are decoded by the dedicated
        // `waveform::dvb_t_frame` assembler (TPS-signalled, preamble-less); a
        // NoHeader caller must convey MCS/length out-of-band. Not supported here.
        return Err(BodyError::Failed(RxError::MalformedHeader));
    };

    // 2. Payload, decoded per the MCS the header selected.
    let mcs = mcs_table
        .get(metadata.mcs_index)
        .ok_or(BodyError::Failed(RxError::MalformedHeader))?;
    let pplan = block_plan(
        payload_len,
        cfg.payload_crc,
        mcs.outer_fec,
        mcs.inner_fec,
        cfg.outer_interleaver,
        cfg.inner_interleaver,
        cache,
    );
    let n_sym = symbols_for_coded_bits(cfg, mcs.constellation, pplan.coded_bits);
    let mut eq = make_eq(mcs.constellation);
    // The equalized payload symbols go straight into whichever buffer will
    // outlive this call: the probe's (appended, so several frames from one
    // `feed` sit end to end) or the reused scratch. EVM reads the same span
    // back, so probing adds no second copy of the constellation.
    let sym_start = match probe.as_deref_mut() {
        Some(p) => p.symbols.len(),
        None => {
            scratch.symbols.clear();
            0
        }
    };
    // Too few samples for the (now-known-length) payload ⇒ incomplete.
    let llrs = {
        let sink: &mut Vec<C32> = match probe.as_deref_mut() {
            Some(p) => &mut p.symbols,
            None => &mut scratch.symbols,
        };
        demap(
            mcs.constellation,
            iq,
            cursor,
            n_sym,
            eq.as_mut(),
            Some(sink),
        )
        .ok_or(BodyError::Incomplete)?
    };
    // One hard decision per coded bit, taken once: EVM measures against these,
    // and the channel BER and correction map below are the first
    // `pplan.coded_bits` of the same vector.
    scratch.hard.clear();
    scratch
        .hard
        .extend(llrs.iter().map(|&l| u8::from(l <= 0.0)));
    // EVM against the payload's own hard decisions, measured before the FEC
    // stages consume the LLRs. `symbol_config` re-resolves the constellation so
    // the ideal-point mapper matches the one that produced these symbols.
    let evm_db = {
        let symbols: &[C32] = match probe.as_deref() {
            Some(p) => &p.symbols[sym_start..],
            None => &scratch.symbols,
        };
        crate::demodulate::ofdm::evm_db(
            &symbol_config(cfg, mcs.constellation),
            symbols,
            &scratch.hard,
            n_sym,
        )
    };
    let payload_outcome = match decode_chain(
        &llrs,
        &pplan,
        cfg.payload_crc,
        mcs.outer_fec,
        mcs.inner_fec,
        cfg.outer_interleaver,
        cfg.inner_interleaver,
        cfg.scrambler,
        cfg.scrambler_pos,
        per_frame_seed,
        cache,
        // The payload honors the configured LDPC decode rule (opt-in min-sum).
        cfg.ldpc_decode_rule,
    ) {
        Ok(outcome) if outcome.is_valid() => outcome,
        // The payload reached the demapper but did not verify, so there is no
        // ground truth and no map — but the symbols exist, and a constellation
        // is precisely what an operator looks at when frames stop decoding.
        rest => {
            if let Some(p) = probe.as_deref_mut() {
                p.push_undecoded(sym_start, mcs.constellation, Some(metadata.sequence_num));
            }
            return Err(BodyError::Failed(match rest {
                Err(e) => e,
                Ok(_) => RxError::CrcMismatch,
            }));
        }
    };
    // Take the flags before consuming the bytes, so the payload is moved rather
    // than cloned out of the outcome.
    let (inner_ok, outer_ok) = (payload_outcome.inner_ok, payload_outcome.outer_ok);

    // True bit error rates, from a re-encode of what we just recovered.
    //
    // A frame that passed its CRC *is* the ground truth: re-running the encode
    // chain on it reconstructs exactly what the transmitter sent, so comparing
    // that against what arrived at each stage gives a rate rather than a
    // pass/fail flag. Crucially this needs no prior knowledge of the payload —
    // which is what makes it work over the air, where nothing about the
    // transmitted bits is known in advance.
    //
    // Off unless asked for: it costs one encode per frame. The probe's
    // correction map is built from the same re-encode, so asking for both pays
    // for it once.
    let stages = (measure_ber || probe.is_some()).then(|| {
        encode_chain_stages(
            &payload_outcome.bytes,
            cfg.payload_crc,
            mcs.outer_fec,
            mcs.inner_fec,
            cfg.outer_interleaver,
            cfg.inner_interleaver,
            cfg.scrambler,
            cfg.scrambler_pos,
            per_frame_seed,
            cache,
        )
    });
    let coded_bits = pplan.coded_bits.min(scratch.hard.len());
    let (channel_ber, inner_ber) = match stages.as_ref().filter(|_| measure_ber) {
        // The channel's output is the demapped LLRs hard-decided, compared
        // before any descrambling — `stages.coded` carries the scramble too.
        Some(s) => (
            bit_error_rate(&scratch.hard[..coded_bits], &s.coded),
            bit_error_rate(&payload_outcome.inner_out_bits, &s.outer_il_bits),
        ),
        None => (None, None),
    };

    // The correction map: the same XOR the channel BER collapses to a scalar,
    // kept per bit, plus a third stream saying what the inner decoder made of
    // each one. Nothing new is measured or assumed — the ground truth is the
    // re-encode above, which a noise sweep and a regenerated payload already
    // vouch for.
    if let (Some(p), Some(s)) = (&mut probe, stages.as_ref()) {
        reencode_inner_output(
            cfg,
            mcs.inner_fec,
            &payload_outcome.inner_out_bits,
            pplan.coded_bits,
            per_frame_seed,
            cache,
            &mut p.estimate,
        );
        debug_assert_eq!(
            p.estimate.len(),
            pplan.coded_bits,
            "the re-encoded decoder estimate must span the whole coded block"
        );
        // A block code's `n`/`k` let a display draw codeword boundaries; a
        // convolutional code terminates once per frame and has none to draw.
        let (codeword_bits, codeword_info_bits) = match mcs.inner_fec {
            InnerFec::Ldpc(code) => (code.n(), code.k()),
            InnerFec::None | InnerFec::Convolutional { .. } => (0, 0),
        };
        p.push_decoded(
            sym_start,
            ProbeMeta {
                sequence_num: Some(metadata.sequence_num),
                constellation: mcs.constellation,
                codeword_bits,
                codeword_info_bits,
            },
            &s.coded[..coded_bits.min(s.coded.len())],
            &scratch.hard[..coded_bits],
        );
    }

    let bytes = payload_outcome.bytes;
    let payload_sps = symbol_config(cfg, mcs.constellation).samples_per_ofdm_symbol();
    cursor += n_sym * payload_sps;
    // Trim to the declared payload length (coding blocks are zero-padded).
    let payload = bytes
        .get(..payload_len)
        .map(|s| s.to_vec())
        .unwrap_or(bytes);

    Ok(DecodedBody {
        packet: FramePacket { metadata, payload },
        consumed: cursor,
        evm_db,
        inner_ok,
        outer_ok,
        channel_ber,
        inner_ber,
    })
}

/// The batch OFDM frame demodulator — decodes a frame at a KNOWN start (`iq[0]`
/// is the first sample AFTER the preamble+training; the caller has already
/// synchronized and, if needed, equalized). The exact counterpart of
/// [`OfdmFrameMod`](crate::modulate::OfdmFrameMod), constructed the same way.
///
/// This is the flat-channel, known-start receiver; the streaming
/// [`OfdmFrameStreamDemod`] runs `ofdm_sync`, CFO correction, and training-
/// symbol equalization for unknown start / CFO / multipath.
#[derive(Debug, Clone)]
pub struct OfdmFrameDemod {
    cfg: OfdmConfig,
    mcs_table: McsTable,
    /// FEC code cache, so a stream of frames builds each code once (see
    /// [`CodecCache`]). Held behind `Arc` so it can be shared with a paired
    /// modulator (TX and RX then reuse the same built codes).
    cache: Arc<CodecCache>,
}

impl OfdmFrameDemod {
    /// Creates a batch demodulator over `cfg` and an `mcs_table`. It owns a
    /// fresh, private [`CodecCache`] warmed across the frames it decodes; use
    /// [`with_cache`](Self::with_cache) to share one with a modulator.
    pub fn new(cfg: OfdmConfig, mcs_table: McsTable) -> Self {
        Self::with_cache(cfg, mcs_table, Arc::new(CodecCache::new()))
    }

    /// Like [`new`](Self::new), but reuses the caller-provided `cache` — share
    /// one `Arc<CodecCache>` across a modulator/demodulator pair (or several
    /// links on the same MCS) so each FEC code is constructed only once.
    pub fn with_cache(cfg: OfdmConfig, mcs_table: McsTable, cache: Arc<CodecCache>) -> Self {
        crate::modulate::ofdm_frame::assert_baseband(&cfg);
        Self {
            cfg,
            mcs_table,
            cache,
        }
    }

    pub fn config(&self) -> &OfdmConfig {
        &self.cfg
    }

    /// Decodes one frame whose IQ begins at the first post-preamble sample.
    /// Returns the recovered [`FramePacket`] or an [`RxError`]. The internal
    /// [`CodecCache`] is reused across calls, so decoding many frames on one
    /// `OfdmFrameDemod` builds each FEC code only once.
    pub fn decode(&self, iq: &[C32]) -> Result<FramePacket, RxError> {
        // `&self`, so the working buffers are function-local rather than carried
        // on the receiver: one frame, one set, dropped on return. Unchanged
        // behaviour and unchanged cost against the per-call vectors this
        // replaces.
        let mut scratch = FrameScratch::default();
        decode_frame_body(
            &self.cfg,
            &self.mcs_table,
            iq,
            None,
            &self.cache,
            false,
            &mut scratch,
            None,
        )
        .map(|body| body.packet)
        .map_err(|e| match e {
            // A batch caller has no "wait for more" option; a truncated buffer
            // is a malformed input here.
            BodyError::Incomplete => RxError::MalformedHeader,
            BodyError::Failed(err) => err,
        })
    }
}

// ── Streaming receiver ─────────────────────────────────────────────────────

/// A successfully received frame plus its per-frame RX diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct RxFrame {
    pub packet: FramePacket,
    /// Acquisition/quality diagnostics: `cfo_hz` and `timing_offset_samples`
    /// are populated by the streaming receiver; `evm_db`/`channel_mse` are
    /// left `None` here (measured by the per-symbol pipeline, not the frame
    /// layer).
    pub diagnostics: OfdmRxFrame,
}

/// Streaming OFDM frame receiver: push raw IQ with [`feed`](Self::feed), poll
/// completed frames (or typed errors). Mirrors `Ft8StreamDecoder`'s
/// accumulate-and-drain shape.
///
/// Each `feed` accumulates samples, searches the buffer for a preamble via
/// `ofdm_sync`, and — for a candidate with enough buffered samples — corrects
/// CFO (`Rotator`), estimates the channel from the training symbol
/// (`OfdmEqualizer`), decodes the frame, and drains its samples from the
/// buffer, looping to drain multiple frames. A frame whose payload has not
/// fully arrived is held until a later `feed` completes it.
pub struct OfdmFrameStreamDemod {
    cfg: OfdmConfig,
    mcs_table: McsTable,
    preamble: OfdmPreamble,
    fs: f32,
    buf: Vec<C32>,
    /// Minimum sync score to accept a candidate.
    score_threshold: f32,
    /// Whether to carry the per-bin channel estimate on each frame's
    /// diagnostics. Off by default: it costs an `n_fft`-sized allocation per
    /// frame and only diagnostic consumers want it.
    want_channel_estimate: bool,
    /// Whether to measure true bit error rates by re-encoding each decoded
    /// frame. Off by default: it costs one encode per frame.
    want_error_rates: bool,
    /// FEC code cache, warmed across the frames this receiver decodes (see
    /// [`CodecCache`]). Held behind `Arc` so it can be shared with a paired
    /// modulator.
    cache: Arc<CodecCache>,
    /// Per-frame working buffers, reused across every frame this receiver
    /// decodes (see [`FrameScratch`]).
    scratch: FrameScratch,
}

impl OfdmFrameStreamDemod {
    /// Creates a streaming receiver with a fresh, private [`CodecCache`]; use
    /// [`with_cache`](Self::with_cache) to share one with a modulator.
    pub fn new(cfg: OfdmConfig, mcs_table: McsTable, preamble: OfdmPreamble) -> Self {
        Self::with_cache(cfg, mcs_table, preamble, Arc::new(CodecCache::new()))
    }

    /// Like [`new`](Self::new), but reuses the caller-provided `cache` so a
    /// modulator/demodulator pair sharing one `Arc<CodecCache>` builds each FEC
    /// code only once between them.
    pub fn with_cache(
        cfg: OfdmConfig,
        mcs_table: McsTable,
        preamble: OfdmPreamble,
        cache: Arc<CodecCache>,
    ) -> Self {
        crate::modulate::ofdm_frame::assert_baseband(&cfg);
        let fs = cfg.fs;
        Self {
            cfg,
            mcs_table,
            preamble,
            fs,
            buf: Vec::new(),
            score_threshold: 0.5,
            want_channel_estimate: false,
            want_error_rates: false,
            cache,
            scratch: FrameScratch::default(),
        }
    }

    /// Overrides the sync-score acceptance threshold (default 0.5).
    pub fn with_score_threshold(mut self, t: f32) -> Self {
        self.score_threshold = t;
        self
    }

    /// Carries the per-bin channel estimate on each frame's
    /// [`diagnostics`](RxFrame::diagnostics). Off by default — it costs an
    /// `n_fft`-sized allocation per frame, which only a caller measuring
    /// channel response wants to pay.
    ///
    /// Requires the preamble to carry a training symbol; without one there is
    /// nothing to estimate from and the field stays `None`.
    pub fn with_channel_estimate(mut self, on: bool) -> Self {
        self.want_channel_estimate = on;
        self
    }

    /// Measures true bit error rates at the inner decoder's input and output,
    /// reported as [`channel_ber`](OfdmRxFrame::channel_ber) and
    /// [`inner_ber`](OfdmRxFrame::inner_ber). Off by default.
    ///
    /// Works by re-encoding each successfully decoded frame: a frame that
    /// passed its CRC is ground truth, so the re-encode reconstructs what the
    /// transmitter sent and the difference from what arrived is a real error
    /// rate. **No prior knowledge of the payload is required**, which is what
    /// makes this usable over the air rather than only against a known test
    /// vector.
    ///
    /// Only frames that decode are measured — an undecodable frame has no
    /// ground truth, so a rising error rate that suddenly stops reporting is
    /// itself the signal that the link has given up.
    ///
    /// Costs one encode per decoded frame.
    pub fn with_error_rates(mut self, on: bool) -> Self {
        self.want_error_rates = on;
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

    /// Feeds IQ samples and returns any frames (or errors) that completed.
    pub fn feed(&mut self, iq: &[C32]) -> Vec<Result<RxFrame, RxError>> {
        self.buf.extend_from_slice(iq);
        self.drain(None)
    }

    /// Runs a final decode pass over the residual buffer (e.g. at end of
    /// stream). Same semantics as `feed` with no new samples.
    pub fn flush(&mut self) -> Vec<Result<RxFrame, RxError>> {
        self.drain(None)
    }

    /// [`feed`](Self::feed), additionally filling `probe` with each frame's
    /// equalized payload symbols and per-coded-bit correction map — the two
    /// quantities a constellation / decoder display needs. See [`OfdmRxProbe`].
    ///
    /// `probe` is **cleared first**, then refilled with everything this call
    /// produced; its allocations are retained, so probing a steady stream does
    /// not reallocate.
    ///
    /// **The gate is the choice of method, not a flag.** There is no
    /// `want_probe` field, so the unprobed [`feed`](Self::feed) gains no runtime
    /// branch and no receiver state can disagree with what the caller believes
    /// it enabled. A viewer that toggles its pane simply calls the other method.
    ///
    /// Costs, per frame: one encode chain (the same one
    /// [`with_error_rates`](Self::with_error_rates) runs — asking for both pays
    /// for it once), one further inner encode to re-derive what the decoder
    /// decided, and two buffer fills. Frames that fail their payload CRC still
    /// contribute their symbols, with an empty correction map; frames whose
    /// *header* fails never reach the payload demapper and contribute nothing.
    ///
    /// A `dvb_t_scattered` link reports **no** probe frames at all: its demap
    /// path collects no symbols, because DVB-T frames are decoded by
    /// `waveform::dvb_t_frame`, which carries its own diagnostics.
    pub fn feed_probed(
        &mut self,
        iq: &[C32],
        probe: &mut OfdmRxProbe,
    ) -> Vec<Result<RxFrame, RxError>> {
        probe.clear();
        self.buf.extend_from_slice(iq);
        self.drain(Some(probe))
    }

    /// [`flush`](Self::flush) with the probe of [`feed_probed`](Self::feed_probed).
    pub fn flush_probed(&mut self, probe: &mut OfdmRxProbe) -> Vec<Result<RxFrame, RxError>> {
        probe.clear();
        self.drain(Some(probe))
    }

    /// Repeatedly locates and decodes frames from the front of the buffer,
    /// consuming their samples, until no further complete frame is present.
    fn drain(&mut self, mut probe: Option<&mut OfdmRxProbe>) -> Vec<Result<RxFrame, RxError>> {
        let mut out = Vec::new();
        while let FrameStep::Decoded(result, consume_to) = self.try_one_frame(probe.as_deref_mut())
        {
            self.buf.drain(..consume_to);
            out.push(result);
        }
        out
    }

    /// Attempts to decode one frame at the front of the buffer.
    fn try_one_frame(&mut self, mut probe: Option<&mut OfdmRxProbe>) -> FrameStep {
        let n_fft = self.cfg.carrier_plan.n_fft();
        let cp_len = self.cfg.carrier_plan.cp_len();
        let pre_len = self.preamble.total_len();

        // Need at least a full preamble plus one header's worth before a search
        // can yield a decodable frame.
        if self.buf.len() < pre_len + (n_fft + cp_len) {
            return FrameStep::NeedMore;
        }

        let sync = ofdm_sync(&self.buf, self.fs, &self.preamble, 0, self.buf.len());
        // The EARLIEST accepted candidate, not the best-ranked one: this drains
        // the buffer front-to-back, and locking onto a later frame discards
        // every frame before it with nothing reported. See `earliest_accepted`.
        let Some(best) = earliest_accepted(sync, self.score_threshold, pre_len) else {
            return FrameStep::NeedMore;
        };

        // Total CFO = fractional + integer·subcarrier-spacing.
        let subcarrier_spacing = self.fs / n_fft as f32;
        let total_cfo = best.cfo_hz + best.integer_cfo_bins as f32 * subcarrier_spacing;

        // CFO-correct from the preamble start onward into a scratch buffer.
        let region = &self.buf[best.start_sample..];
        let mut corrected = vec![C32::default(); region.len()];
        let mut rot = Rotator::new(-total_cfo, self.fs);
        rot.rotate_block(region, &mut corrected);

        // Channel estimate from the training symbol (if the preamble carries
        // one), located just after the S&C repeats.
        let channel_estimate = self.estimate_channel(&corrected);

        // The frame body begins right after the whole preamble (S&C + training).
        if corrected.len() < pre_len {
            return FrameStep::NeedMore;
        }
        let body = &corrected[pre_len..];

        // A probe frame is committed as one unit: record where the buffers
        // stand, so an attempt that turns out to be `Incomplete` — which
        // consumes nothing and will be re-run from the header on the next
        // `feed` — leaves nothing behind to be reported a second time.
        let mark = probe.as_deref().map(|p| p.mark());

        match decode_frame_body(
            &self.cfg,
            &self.mcs_table,
            body,
            channel_estimate.as_deref(),
            &self.cache,
            self.want_error_rates,
            &mut self.scratch,
            probe.as_deref_mut(),
        ) {
            Ok(body) => {
                let diagnostics = OfdmRxFrame {
                    bits: Vec::new(),
                    num_symbols: 0,
                    evm_db: body.evm_db,
                    cfo_hz: Some(total_cfo),
                    timing_offset_samples: Some(best.start_sample as i32),
                    // A scalar channel MSE needs a reference to measure
                    // against, and a single-shot training estimate has none —
                    // deriving one means separating channel from noise, which
                    // is an estimator rather than an exposure. The per-bin
                    // estimate below carries strictly more information.
                    channel_mse: None,
                    sync_score: Some(best.score),
                    channel_estimate: self
                        .want_channel_estimate
                        .then(|| channel_estimate.as_deref().map(channel_from_training))
                        .flatten(),
                    inner_fec_ok: Some(body.inner_ok),
                    outer_fec_ok: Some(body.outer_ok),
                    channel_ber: body.channel_ber,
                    inner_ber: body.inner_ber,
                };
                let consume_to = best.start_sample + pre_len + body.consumed;
                if consume_to > self.buf.len() {
                    // Shouldn't happen (decode succeeded), but guard the drain.
                    return FrameStep::NeedMore;
                }
                FrameStep::Decoded(
                    Ok(RxFrame {
                        packet: body.packet,
                        diagnostics,
                    }),
                    consume_to,
                )
            }
            // The header or payload has not fully arrived yet — hold and retry
            // when more samples are fed. No buffer is consumed, and neither is
            // any probe state.
            Err(BodyError::Incomplete) => {
                if let (Some(p), Some(m)) = (&mut probe, mark) {
                    p.rollback(m);
                }
                FrameStep::NeedMore
            }
            // A genuine decode failure on a fully-present frame: report it and
            // advance just past this preamble so the search continues past it
            // (avoids re-locking the same corrupt occurrence forever).
            Err(BodyError::Failed(e)) => {
                let skip = (best.start_sample + pre_len).min(self.buf.len());
                FrameStep::Decoded(Err(e), skip)
            }
        }
    }

    /// Estimates the per-bin channel from the training symbol in `corrected`
    /// (CFO-corrected, preamble-start-relative). Returns `None` if the preamble
    /// carries no training symbol.
    fn estimate_channel(&self, corrected: &[C32]) -> Option<Vec<C32>> {
        let training = self.preamble.training_symbol?;
        let n_fft = training.n_fft;
        let cp_len = training.cp_len;
        let training_start = self.preamble.num_repeats * self.preamble.repeat_len;
        let end = training_start + n_fft + cp_len;
        if corrected.len() < end {
            return None;
        }
        // Estimate the channel at the same window position the data symbols use
        // (see `soft_demap`), or the held estimate would be applied at a
        // different window than it was measured at.
        let mut symbol_fft =
            SymbolFft::new(n_fft, cp_len).with_window_backoff(self.cfg.rx_window_backoff);
        let freq = symbol_fft.demod_symbol(&corrected[training_start..end])?;
        Some(freq.to_vec())
    }
}

/// Converts a received training symbol's frequency bins to the channel
/// `H[k] = received[k] / known[k]`, matching what `OfdmEqualizer` does
/// internally. Exposed on the diagnostics because the known pattern is
/// crate-internal, so a caller cannot perform this division itself.
fn channel_from_training(received_freq: &[C32]) -> Vec<C32> {
    let known = crate::sync::ofdm_sync::training_symbol_freq_pattern(received_freq.len());
    received_freq
        .iter()
        .zip(known.iter())
        .map(|(&rx, &k)| rx / k)
        .collect()
}

/// One step of the streaming drain loop.
enum FrameStep {
    /// A frame (or error) decoded; consume the buffer up to this index.
    Decoded(Result<RxFrame, RxError>, usize),
    /// Not enough buffered samples yet; wait for more.
    NeedMore,
}
