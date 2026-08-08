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
    scramble_bytes, symbol_config, symbols_for_coded_bits,
};
use crate::multicarrier::{CarrierGrid, GridExtract, SymbolFft};
use crate::sync::{OfdmPreamble, ofdm_sync};
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
fn soft_demap(
    base: &OfdmConfig,
    constellation: ConstellationOrder,
    iq: &[C32],
    n_symbols: usize,
    equalizer: Option<&mut OfdmEqualizer>,
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
            let mut symbol_fft = SymbolFft::new(n_fft, cp_len);
            let mut grid_extract = GridExtract::new(grid);
            let mut equalized = vec![C32::default(); n_fft];
            let mut in_off = 0;
            let mut out_off = 0;
            for _ in 0..n_symbols {
                let freq = symbol_fft.demod_symbol(&iq[in_off..])?;
                if eq.process(freq, &mut equalized).out_written != n_fft {
                    return None;
                }
                if grid_extract.process(&equalized, &mut symbols).out_written != n_data {
                    return None;
                }
                let sw = soft.process(&symbols, &mut llrs[out_off..out_off + bps]);
                if sw.out_written != bps {
                    return None;
                }
                in_off += sps;
                out_off += bps;
            }
        }
    }
    Some(llrs)
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
    let mut symbol_fft = SymbolFft::new(n_fft, cp_len);
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

/// Decodes one logical block's coded LLRs back to its info bytes, checking the
/// CRC. Returns `Ok((bytes, crc_ok))` or an error if the structure is invalid.
/// Decodes one logical block's coded LLRs back to its info bytes, checking the
/// CRC — the exact inverse of `modulate::ofdm_frame::encode_chain`. Public so
/// per-standard frame assemblers (e.g. `waveform::dvb_t_frame`) reuse the shared
/// FEC decode rather than duplicating it. Returns `(bytes, all_ok)` where
/// `all_ok` folds in the CRC and every FEC block's convergence.
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
) -> Result<(Vec<u8>, bool), RxError> {
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
    let (mut outer_il_bits, inner_ok) =
        inner_decode(inner, inner_de, plan.outer_il_bits, cache, ldpc_rule);
    outer_il_bits.truncate(plan.outer_il_bits);

    // 3. Outer deinterleave (byte/bit domain), then outer decode.
    let outer_de = deinterleave_bits(outer_il, &outer_il_bits);
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
    Ok((bytes, crc_ok && inner_ok && outer_ok))
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

/// Decodes a frame body (header + payload) from `iq[0]` — the first sample
/// AFTER the preamble+training, already CFO-corrected. When
/// `channel_estimate` is `Some(n_fft freq bins)` the soft-demap equalizes each
/// symbol against it (multipath); `None` is the flat-channel path.
///
/// Returns the recovered [`FramePacket`] and the number of IQ samples the
/// header+payload occupied (so a streaming caller can advance its buffer), or a
/// [`BodyError`] distinguishing "incomplete" from a genuine failure.
fn decode_frame_body(
    cfg: &OfdmConfig,
    mcs_table: &McsTable,
    iq: &[C32],
    channel_estimate: Option<&[C32]>,
    cache: &CodecCache,
) -> Result<(FramePacket, usize), BodyError> {
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
                     eq: Option<&mut OfdmEqualizer>|
     -> Option<Vec<f32>> {
        match scattered.as_mut() {
            Some(x) => soft_demap_scattered(cfg, constellation, &iq[off..], n_sym, x),
            None => soft_demap(cfg, constellation, &iq[off..], n_sym, eq),
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
        let llrs = demap(HEADER_CONSTELLATION, iq, cursor, n_sym, eq.as_mut())
            .ok_or(BodyError::Incomplete)?;
        let (fields, ok) = decode_chain(
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
        if !ok {
            return Err(BodyError::Failed(RxError::HeaderCrcMismatch));
        }
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
    // Too few samples for the (now-known-length) payload ⇒ incomplete.
    let llrs =
        demap(mcs.constellation, iq, cursor, n_sym, eq.as_mut()).ok_or(BodyError::Incomplete)?;
    let (bytes, ok) = decode_chain(
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
    )
    .map_err(BodyError::Failed)?;
    if !ok {
        return Err(BodyError::Failed(RxError::CrcMismatch));
    }
    let payload_sps = symbol_config(cfg, mcs.constellation).samples_per_ofdm_symbol();
    cursor += n_sym * payload_sps;
    // Trim to the declared payload length (coding blocks are zero-padded).
    let payload = bytes
        .get(..payload_len)
        .map(|s| s.to_vec())
        .unwrap_or(bytes);

    Ok((FramePacket { metadata, payload }, cursor))
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
        decode_frame_body(&self.cfg, &self.mcs_table, iq, None, &self.cache)
            .map(|(frame, _)| frame)
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
    /// FEC code cache, warmed across the frames this receiver decodes (see
    /// [`CodecCache`]). Held behind `Arc` so it can be shared with a paired
    /// modulator.
    cache: Arc<CodecCache>,
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
        let fs = cfg.fs;
        Self {
            cfg,
            mcs_table,
            preamble,
            fs,
            buf: Vec::new(),
            score_threshold: 0.5,
            cache,
        }
    }

    /// Overrides the sync-score acceptance threshold (default 0.5).
    pub fn with_score_threshold(mut self, t: f32) -> Self {
        self.score_threshold = t;
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
        self.drain()
    }

    /// Runs a final decode pass over the residual buffer (e.g. at end of
    /// stream). Same semantics as `feed` with no new samples.
    pub fn flush(&mut self) -> Vec<Result<RxFrame, RxError>> {
        self.drain()
    }

    /// Repeatedly locates and decodes frames from the front of the buffer,
    /// consuming their samples, until no further complete frame is present.
    fn drain(&mut self) -> Vec<Result<RxFrame, RxError>> {
        let mut out = Vec::new();
        while let FrameStep::Decoded(result, consume_to) = self.try_one_frame() {
            self.buf.drain(..consume_to);
            out.push(result);
        }
        out
    }

    /// Attempts to decode one frame at the front of the buffer.
    fn try_one_frame(&mut self) -> FrameStep {
        let n_fft = self.cfg.carrier_plan.n_fft();
        let cp_len = self.cfg.carrier_plan.cp_len();
        let pre_len = self.preamble.total_len();

        // Need at least a full preamble plus one header's worth before a search
        // can yield a decodable frame.
        if self.buf.len() < pre_len + (n_fft + cp_len) {
            return FrameStep::NeedMore;
        }

        let sync = ofdm_sync(&self.buf, self.fs, &self.preamble, 0, self.buf.len());
        let Some(best) = sync.into_iter().find(|r| r.score >= self.score_threshold) else {
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

        match decode_frame_body(
            &self.cfg,
            &self.mcs_table,
            body,
            channel_estimate.as_deref(),
            &self.cache,
        ) {
            Ok((packet, body_samples)) => {
                let diagnostics = OfdmRxFrame {
                    bits: Vec::new(),
                    num_symbols: 0,
                    evm_db: None,
                    cfo_hz: Some(total_cfo),
                    timing_offset_samples: Some(best.start_sample as i32),
                    channel_mse: None,
                };
                let consume_to = best.start_sample + pre_len + body_samples;
                if consume_to > self.buf.len() {
                    // Shouldn't happen (decode succeeded), but guard the drain.
                    return FrameStep::NeedMore;
                }
                FrameStep::Decoded(
                    Ok(RxFrame {
                        packet,
                        diagnostics,
                    }),
                    consume_to,
                )
            }
            // The header or payload has not fully arrived yet — hold and retry
            // when more samples are fed. No buffer is consumed.
            Err(BodyError::Incomplete) => FrameStep::NeedMore,
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
        let mut symbol_fft = SymbolFft::new(n_fft, cp_len);
        let freq = symbol_fft.demod_symbol(&corrected[training_start..end])?;
        Some(freq.to_vec())
    }
}

/// One step of the streaming drain loop.
enum FrameStep {
    /// A frame (or error) decoded; consume the buffer up to this index.
    Decoded(Result<RxFrame, RxError>, usize),
    /// Not enough buffered samples yet; wait for more.
    NeedMore,
}
