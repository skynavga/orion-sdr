// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/demodulate/ofdm_frame.rs
//
// The OFDM frame (MAC-layer) demodulator: the exact inverse of
// `modulate::ofdm_frame`. This release provides the *batch* path
// [`demodulate_frame`] — a frame at a KNOWN start (no acquisition/streaming
// yet; that is the next release). It runs the concatenated COFDM decode chain:
//
//   IQ → soft-demap (LLRs) → inner-deinterleave (LLR) → inner-decode →
//        outer-deinterleave (byte) → outer-decode → descramble → strip CRC
//
// The header is decoded first with the fixed built-in scheme (BPSK + rate-1/2
// LDPC) to recover `mcs_index`/`payload_len`/`sequence_num`/`flags`/seed, then
// the payload is decoded at the MCS the header selected.

use crate::core::Block;
use crate::demodulate::ofdm::{OfdmDemod, OfdmSoftDemod};
use crate::fec::{
    BlockInterleaver, CrcKind, FrameMetadata, FramePacket, HeaderFormat, InnerFec, InterleaverKind,
    Ldpc, OuterFec, RxError, ScramblerKind, ScramblerPos,
};
use crate::modulate::ofdm::{ConstellationOrder, OfdmConfig};
use crate::modulate::ofdm_frame::{
    BCH_INFO_BITS, BlockPlan, HEADER_CONSTELLATION, HEADER_FIELD_BYTES, HEADER_LDPC, McsTable,
    bits_to_bytes, block_plan, build_scrambler, bytes_to_bits, check_and_strip_crc,
    shortened_bch_for, symbol_config, symbols_for_coded_bits,
};
use num_complex::Complex32 as C32;

/// Soft-demaps `n_symbols` OFDM symbols starting at `iq[0]` into a flat LLR
/// vector (one `f32` per coded bit, `+ ⇒ bit 0`). Returns `None` if `iq` is
/// too short.
fn soft_demap(
    base: &OfdmConfig,
    constellation: ConstellationOrder,
    iq: &[C32],
    n_symbols: usize,
) -> Option<Vec<f32>> {
    let cfg = symbol_config(base, constellation);
    let sps = cfg.samples_per_ofdm_symbol();
    if iq.len() < n_symbols * sps {
        return None;
    }
    // Two stages: OfdmDemod turns time-domain IQ into per-carrier soft symbols
    // (CP-remove → FFT → grid-extract), then OfdmSoftDemod turns those soft
    // symbols into per-bit LLRs. (This flat-channel path has no equalizer; the
    // streaming receiver adds sync/CFO/equalization in the next release.)
    let mut demod = OfdmDemod::new(&cfg);
    let mut soft = OfdmSoftDemod::new(&cfg);
    let n_data = cfg.carrier_plan.data_carriers().len();
    let bps = cfg.bits_per_ofdm_symbol();
    let mut symbols = vec![C32::default(); n_data];
    let mut llrs = vec![0.0f32; n_symbols * bps];
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
    Some(llrs)
}

/// Inverse of the block interleaver, in the LLR (`f32`) domain.
fn deinterleave_llrs(il: InterleaverKind, llrs: &[f32]) -> Vec<f32> {
    match il {
        InterleaverKind::None => llrs.to_vec(),
        InterleaverKind::Block { rows, cols } => {
            let block = rows * cols;
            let bi = BlockInterleaver::new(rows, cols);
            let mut out = Vec::with_capacity(llrs.len());
            for chunk in llrs.chunks(block) {
                if chunk.len() < block {
                    out.extend_from_slice(chunk);
                    continue;
                }
                let mut restored = vec![0.0f32; block];
                bi.deinterleave(chunk, &mut restored);
                out.extend_from_slice(&restored);
            }
            out
        }
    }
}

/// Inverse of the block interleaver, in the hard-bit (`u8`) domain.
fn deinterleave_bits(il: InterleaverKind, bits: &[u8]) -> Vec<u8> {
    match il {
        InterleaverKind::None => bits.to_vec(),
        InterleaverKind::Block { rows, cols } => {
            let block = rows * cols;
            let bi = BlockInterleaver::new(rows, cols);
            let mut out = Vec::with_capacity(bits.len());
            for chunk in bits.chunks(block) {
                if chunk.len() < block {
                    out.extend_from_slice(chunk);
                    continue;
                }
                let mut restored = vec![0u8; block];
                bi.deinterleave(chunk, &mut restored);
                out.extend_from_slice(&restored);
            }
            out
        }
    }
}

/// Inner-decodes an LLR stream into hard info bits, fragmenting into N-sized
/// codeword blocks (mirroring `inner_encode`). Returns the info bits and
/// whether every block converged.
fn inner_decode(inner: InnerFec, coded_llrs: &[f32]) -> (Vec<u8>, bool) {
    match inner {
        InnerFec::None => {
            // Hard-decide the LLRs directly.
            (
                coded_llrs.iter().map(|&l| u8::from(l <= 0.0)).collect(),
                true,
            )
        }
        InnerFec::Ldpc(code) => {
            let ldpc = Ldpc::new(code);
            let n = ldpc.n();
            let mut info = Vec::new();
            let mut all_ok = true;
            for chunk in coded_llrs.chunks(n) {
                if chunk.len() < n {
                    all_ok = false;
                    break;
                }
                let (msg, unsat) = ldpc.decode_soft(chunk, 50);
                if unsat != 0 {
                    all_ok = false;
                }
                info.extend_from_slice(&msg);
            }
            (info, all_ok)
        }
    }
}

/// Outer-decodes hard bits into message bits, fragmenting into shortened-BCH
/// codeword blocks (mirroring `outer_encode`). Returns the message bits and
/// whether every block decoded.
fn outer_decode(outer: OuterFec, coded_bits: &[u8]) -> (Vec<u8>, bool) {
    match outer {
        OuterFec::None => (coded_bits.to_vec(), true),
        OuterFec::Bch { t } => {
            let code = shortened_bch_for(t, BCH_INFO_BITS);
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
    }
}

/// Decodes one logical block's coded LLRs back to its info bytes, checking the
/// CRC. Returns `Ok((bytes, crc_ok))` or an error if the structure is invalid.
#[allow(clippy::too_many_arguments)]
fn decode_chain(
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
    let (mut outer_il_bits, inner_ok) = inner_decode(inner, inner_de);
    outer_il_bits.truncate(plan.outer_il_bits);

    // 3. Outer deinterleave (byte/bit domain), then outer decode.
    let outer_de = deinterleave_bits(outer_il, &outer_il_bits);
    let outer_de = &outer_de[..plan.outer_coded_bits.min(outer_de.len())];
    let (mut framed_bits, outer_ok) = outer_decode(outer, outer_de);
    framed_bits.truncate(plan.framed_bytes * 8);

    if framed_bits.len() < plan.framed_bytes * 8 {
        return Err(RxError::MalformedHeader);
    }
    let mut framed = bits_to_bytes(&framed_bits);

    // 4. Invert the before-outer scramble (byte domain).
    if scrambler_pos == ScramblerPos::BeforeOuterFec
        && let Some(ref s) = sc
    {
        s.scramble(&mut framed);
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

/// Batch-demodulates a frame at a KNOWN start (`iq[0]` is the first sample
/// AFTER the preamble+training — the caller has already synchronized and, if
/// needed, equalized). Returns the recovered [`FramePacket`] or an [`RxError`].
///
/// This is the non-streaming path for this release; the streaming
/// `feed`/`flush` receiver that runs `ofdm_sync` and handles unknown start,
/// CFO, and multipath is the next release.
pub fn demodulate_frame(
    cfg: &OfdmConfig,
    mcs_table: &McsTable,
    iq: &[C32],
) -> Result<FramePacket, RxError> {
    let mut cursor = 0usize;

    // 1. Header (unless NoHeader).
    let (metadata, per_frame_seed, payload_len) = if cfg.header_format == HeaderFormat::OrionSdr {
        let hplan = block_plan(
            HEADER_FIELD_BYTES,
            cfg.header_crc,
            OuterFec::None,
            InnerFec::Ldpc(HEADER_LDPC),
            InterleaverKind::None,
            InterleaverKind::None,
        );
        let n_sym = symbols_for_coded_bits(cfg, HEADER_CONSTELLATION, hplan.coded_bits);
        let llrs = soft_demap(cfg, HEADER_CONSTELLATION, &iq[cursor..], n_sym)
            .ok_or(RxError::MalformedHeader)?;
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
        )?;
        if !ok {
            return Err(RxError::HeaderCrcMismatch);
        }
        if fields.len() < HEADER_FIELD_BYTES {
            return Err(RxError::MalformedHeader);
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
        // NoHeader: the caller must convey MCS/length out-of-band. Not
        // supported by this batch entry point yet.
        return Err(RxError::MalformedHeader);
    };

    // 2. Payload, decoded per the MCS the header selected.
    let mcs = mcs_table
        .get(metadata.mcs_index)
        .ok_or(RxError::MalformedHeader)?;
    let pplan = block_plan(
        payload_len,
        cfg.payload_crc,
        mcs.outer_fec,
        mcs.inner_fec,
        cfg.outer_interleaver,
        cfg.inner_interleaver,
    );
    let n_sym = symbols_for_coded_bits(cfg, mcs.constellation, pplan.coded_bits);
    let llrs =
        soft_demap(cfg, mcs.constellation, &iq[cursor..], n_sym).ok_or(RxError::MalformedHeader)?;
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
    )?;
    if !ok {
        return Err(RxError::CrcMismatch);
    }
    // Trim to the declared payload length (coding blocks are zero-padded).
    let payload = bytes
        .get(..payload_len)
        .map(|s| s.to_vec())
        .unwrap_or(bytes);

    Ok(FramePacket { metadata, payload })
}
