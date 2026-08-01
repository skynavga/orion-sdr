// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/modulate/ofdm_frame.rs
//
// The OFDM frame (MAC-layer) modulator: turns a `FramePacket` into a flat
// stream of time-domain IQ, applying the concatenated COFDM coding chain and
// prepending the acquisition preamble + a fixed, MCS-independent header.
//
// On-air layout (transmit order):
//   [ S&C preamble + training symbol ][ header symbols ][ payload symbols ]
//
// The header is coded with a fixed built-in scheme (BPSK + rate-1/2 LDPC, no
// interleaver, no scrambler) so the receiver can always decode it before it
// knows the payload MCS. Its byte layout is `HEADER_FIELD_BYTES` of fields
// followed by the configured `header_crc`. The payload is coded per the MCS
// selected by `metadata.mcs_index` (constellation + inner/outer FEC from the
// MCS table) plus the link-wide interleavers/scrambler/`payload_crc` from
// `OfdmConfig`.
//
// This module owns the shared bit-domain coding chain (`encode_chain`,
// `pack_*`) that the frame demodulator inverts; the demodulator imports these
// so the two are exact mirrors.

use super::ofdm::{ConstellationOrder, OfdmConfig, OfdmMod};
use crate::codec::{crc16, crc32};
use crate::fec::{
    Bch, CrcKind, FramePacket, HeaderFormat, InnerFec, InterleaverKind, Ldpc, LdpcCode, OuterFec,
    PnScrambler, ReedSolomon, ScramblerKind, ScramblerPos, SeedMode, conv_encode_punctured,
    punctured_coded_len,
};
use crate::multicarrier::CarrierPlan;
use crate::sync::{OfdmPreamble, generate_ofdm_preamble};
use num_complex::Complex32 as C32;
use std::sync::{Arc, Mutex};

/// Memoizes the constructed FEC code objects a link reuses frame after frame.
///
/// Constructing a code — especially [`Ldpc::new`], whose sparse parity-check
/// build with its 4-cycle guard costs milliseconds — is a pure function of the
/// code's parameters, so the object is identical every frame. Without this the
/// concatenated-FEC chain rebuilt its `Ldpc`/`Bch`/`ReedSolomon` on *every*
/// frame (encode and decode); the cache builds each once per link and hands out
/// shared references thereafter.
///
/// The key spaces are tiny (a link uses one header LDPC plus the handful of
/// codes in its MCS table), so linear-scan association lists beat a hash map
/// here. A `Mutex` gives lazy population behind the `&self` encode/decode entry
/// points; the cached objects are handed out as `Arc`s so callers hold them
/// without keeping the lock across the (potentially long) encode/decode call.
///
/// `Send + Sync` (via `Arc`/`Mutex`) so it can live inside an `OfdmFrameMod` /
/// `OfdmFrameStreamDemod` exposed to the PyO3 bindings, which require their
/// pyclasses to be thread-safe. Access is a handful of uncontended lookups per
/// frame, so the lock is effectively free. The produced codes are bit-identical
/// to freshly constructed ones — this changes speed, never output.
///
/// A tiny memo map keyed by `K`, holding shared code objects `V`.
type CodeMemo<K, V> = Mutex<Vec<(K, Arc<V>)>>;

#[derive(Debug, Default)]
pub struct CodecCache {
    ldpc: CodeMemo<LdpcCode, Ldpc>,
    /// Shortened-BCH keyed by `(t, msg_bits)`.
    bch: CodeMemo<(usize, usize), Bch>,
    /// Reed–Solomon keyed by `(n, n_parity)`.
    rs: CodeMemo<(usize, usize), ReedSolomon>,
}

impl Clone for CodecCache {
    /// A cloned cache starts empty rather than copying entries — cache contents
    /// are pure derivations of the codes used, rebuilt on demand, and this keeps
    /// `Clone` free of a lock acquisition. (Only `OfdmFrameMod` derives `Clone`;
    /// it is not exercised on a hot path.)
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl CodecCache {
    /// A fresh, empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the [`Ldpc`] for `code`, building and caching it on first use.
    pub fn ldpc(&self, code: LdpcCode) -> Arc<Ldpc> {
        let mut table = self.ldpc.lock().unwrap();
        if let Some((_, c)) = table.iter().find(|(k, _)| *k == code) {
            return Arc::clone(c);
        }
        let built = Arc::new(Ldpc::new(code));
        table.push((code, Arc::clone(&built)));
        built
    }

    /// Returns the shortened [`Bch`] correcting `t` errors with a `msg_bits`
    /// message part, building and caching it on first use.
    pub fn bch(&self, t: usize, msg_bits: usize) -> Arc<Bch> {
        let key = (t, msg_bits);
        let mut table = self.bch.lock().unwrap();
        if let Some((_, c)) = table.iter().find(|(k, _)| *k == key) {
            return Arc::clone(c);
        }
        let built = Arc::new(shortened_bch_for(t, msg_bits));
        table.push((key, Arc::clone(&built)));
        built
    }

    /// Returns the [`ReedSolomon`] code `(n, n_parity)`, building and caching it
    /// on first use.
    pub fn rs(&self, n: usize, n_parity: usize) -> Arc<ReedSolomon> {
        let key = (n, n_parity);
        let mut table = self.rs.lock().unwrap();
        if let Some((_, c)) = table.iter().find(|(k, _)| *k == key) {
            return Arc::clone(c);
        }
        let built = Arc::new(ReedSolomon::new(n, n_parity).expect("valid RS config"));
        table.push((key, Arc::clone(&built)));
        built
    }
}

/// Number of header field bytes before the header CRC: `mcs_index` (1) +
/// `payload_len` (4, big-endian) + `sequence_num` (4) + `flags` (1) +
/// `scrambler_seed` (4) = 14 bytes.
pub const HEADER_FIELD_BYTES: usize = 14;

/// The fixed constellation used for header symbols (most robust).
pub const HEADER_CONSTELLATION: ConstellationOrder = ConstellationOrder::Bpsk;

/// The fixed inner code protecting the header — a rate-1/2 LDPC, independent of
/// the payload MCS.
pub const HEADER_LDPC: LdpcCode = LdpcCode::N512R12;

/// A modulation-and-coding scheme: the payload's constellation plus its inner
/// and outer FEC. Selected per-frame by `FrameMetadata::mcs_index` via an
/// [`McsTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mcs {
    pub constellation: ConstellationOrder,
    pub inner_fec: InnerFec,
    pub outer_fec: OuterFec,
}

impl Mcs {
    pub const fn new(
        constellation: ConstellationOrder,
        inner_fec: InnerFec,
        outer_fec: OuterFec,
    ) -> Self {
        Self {
            constellation,
            inner_fec,
            outer_fec,
        }
    }
}

/// Maps an 8-bit `mcs_index` to an [`Mcs`]. The sender and receiver must share
/// the same table.
#[derive(Debug, Clone)]
pub struct McsTable {
    entries: Vec<Mcs>,
}

impl McsTable {
    pub fn new(entries: Vec<Mcs>) -> Self {
        assert!(
            !entries.is_empty(),
            "MCS table must have at least one entry"
        );
        Self { entries }
    }

    /// A small default ladder: increasing constellation order, all with a
    /// rate-1/2 LDPC inner code and a BCH(t=8) outer code — the concatenated
    /// COFDM baseline.
    pub fn default_ladder() -> Self {
        let inner = InnerFec::Ldpc(LdpcCode::N512R12);
        let outer = OuterFec::Bch { t: 8 };
        Self::new(vec![
            Mcs::new(ConstellationOrder::Bpsk, inner, outer),
            Mcs::new(ConstellationOrder::Qpsk, inner, outer),
            Mcs::new(ConstellationOrder::Qam16, inner, outer),
            Mcs::new(ConstellationOrder::Qam64, inner, outer),
        ])
    }

    pub fn get(&self, mcs_index: u8) -> Option<Mcs> {
        self.entries.get(mcs_index as usize).copied()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ── Shared bit/byte helpers ────────────────────────────────────────────────

/// Unpacks bytes into a bit vector, MSB-first per byte.
pub fn bytes_to_bits(bytes: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(bytes.len() * 8);
    for &b in bytes {
        for i in (0..8).rev() {
            bits.push((b >> i) & 1);
        }
    }
    bits
}

/// Packs a bit slice (MSB-first per byte) back into bytes. The bit count must
/// be a multiple of 8.
pub fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    assert_eq!(
        bits.len() % 8,
        0,
        "bit count must be a whole number of bytes"
    );
    let mut bytes = Vec::with_capacity(bits.len() / 8);
    for chunk in bits.chunks(8) {
        let mut b = 0u8;
        for &bit in chunk {
            b = (b << 1) | (bit & 1);
        }
        bytes.push(b);
    }
    bytes
}

/// Appends the selected CRC (over `data`) to `data`, big-endian.
pub fn append_crc(crc: CrcKind, data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    match crc {
        CrcKind::None => {}
        CrcKind::Crc16 => out.extend_from_slice(&crc16(data).to_be_bytes()),
        CrcKind::Crc32 => out.extend_from_slice(&crc32(data).to_be_bytes()),
    }
    out
}

/// Splits `data` into (payload, crc-ok). Returns `None` if `data` is too short
/// to hold the CRC field. With [`CrcKind::None`] the check is vacuously true.
pub fn check_and_strip_crc(crc: CrcKind, data: &[u8]) -> Option<(Vec<u8>, bool)> {
    let clen = crc.len_bytes();
    if data.len() < clen {
        return None;
    }
    let (payload, tail) = data.split_at(data.len() - clen);
    let ok = match crc {
        CrcKind::None => true,
        CrcKind::Crc16 => crc16(payload).to_be_bytes()[..] == *tail,
        CrcKind::Crc32 => crc32(payload).to_be_bytes()[..] == *tail,
    };
    Some((payload.to_vec(), ok))
}

/// Builds a [`PnScrambler`] from a [`ScramblerKind`] and an explicit seed value
/// (for `PerFrameRandom`, the caller supplies the drawn seed). Returns `None`
/// for [`ScramblerKind::None`].
pub fn build_scrambler(kind: ScramblerKind, per_frame_seed: u32) -> Option<PnScrambler> {
    match kind {
        ScramblerKind::None => None,
        ScramblerKind::Additive { poly, width, seed } => {
            let raw = match seed {
                SeedMode::Fixed(v) => v,
                SeedMode::PerFrameRandom => per_frame_seed,
            };
            // Reduce the seed into the register width, and avoid the all-zero
            // fixed point (an all-zero additive LFSR never advances). The
            // receiver derives the same value from the header field, so this
            // reduction must be deterministic.
            let mask = if width >= 32 {
                u32::MAX
            } else {
                (1u32 << width) - 1
            };
            let s = {
                let m = raw & mask;
                if m == 0 { 1 } else { m }
            };
            Some(PnScrambler::new(poly, width as u32, s))
        }
    }
}

// ── Block-size bookkeeping (shared TX/RX) ──────────────────────────────────

/// Deterministic size accounting for one logical block's coding chain, so the
/// transmitter and receiver agree on every intermediate length (needed to trim
/// interleaver/fragmentation zero-padding on receive) and on how many OFDM
/// symbols the coded bits occupy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockPlan {
    /// Raw payload/field byte count (before CRC).
    pub info_bytes: usize,
    /// Bytes after appending the CRC.
    pub framed_bytes: usize,
    /// Bits after the outer code (before outer interleave).
    pub outer_coded_bits: usize,
    /// Bits after outer interleave.
    pub outer_il_bits: usize,
    /// Bits after the inner code (before inner interleave).
    pub inner_coded_bits: usize,
    /// Final coded-bit count (after inner interleave) = symbols · bits/symbol.
    pub coded_bits: usize,
}

/// Rounds `n` up to a whole number of `block`-sized units (identity if
/// `block == 0`).
fn round_up(n: usize, block: usize) -> usize {
    if block == 0 {
        n
    } else {
        n.div_ceil(block) * block
    }
}

/// Bit count after the frame-mode streaming Forney interleaver: pack `n_bits`
/// to whole bytes, round the byte count up to a multiple of `branches` (the
/// feed alignment), add the round-trip delay `branches·(branches−1)·depth`
/// (the flush drain), then back to bits. Mirrors the length growth in
/// [`interleave_bits`]'s `Convolutional` arm so the deinterleaver sees the exact
/// length and can trim the delay offset.
fn conv_il_bits(n_bits: usize, branches: usize, depth: usize) -> usize {
    let bytes =
        round_up(n_bits.div_ceil(8), branches) + crate::fec::conv_roundtrip_delay(branches, depth);
    bytes * 8
}

/// Computes the [`BlockPlan`] for `info_bytes` under the given coding chain,
/// reusing constructed code objects from `cache` (their dimensions are all this
/// needs, but sharing the cache avoids rebuilding them here and in the
/// encode/decode passes).
pub fn block_plan(
    info_bytes: usize,
    crc: CrcKind,
    outer: OuterFec,
    inner: InnerFec,
    outer_il: InterleaverKind,
    inner_il: InterleaverKind,
    cache: &CodecCache,
) -> BlockPlan {
    let framed_bytes = info_bytes + crc.len_bytes();
    let framed_bits = framed_bytes * 8;

    let outer_coded_bits = match outer {
        OuterFec::None => framed_bits,
        OuterFec::Bch { t } => {
            let code = cache.bch(t, BCH_INFO_BITS);
            let n_blocks = framed_bits.div_ceil(BCH_INFO_BITS);
            n_blocks * code.n()
        }
        OuterFec::ReedSolomon { n, n_parity } => {
            // Byte-domain: whole k-byte info blocks → n-byte codewords.
            let rs = cache.rs(n, n_parity);
            let n_blocks = framed_bytes.div_ceil(rs.k());
            n_blocks * rs.n() * 8
        }
    };

    let outer_il_bits = match outer_il {
        InterleaverKind::None => outer_coded_bits,
        InterleaverKind::Block { rows, cols } => round_up(outer_coded_bits, rows * cols),
        InterleaverKind::Convolutional { branches, depth } => {
            conv_il_bits(outer_coded_bits, branches, depth)
        }
    };

    let inner_coded_bits = match inner {
        InnerFec::None => outer_il_bits,
        InnerFec::Ldpc(code) => {
            // LDPC dimensions come straight off the code point (no construction
            // needed), but touch the cache so the object is warm for encode.
            let ldpc = cache.ldpc(code);
            let n_blocks = outer_il_bits.div_ceil(ldpc.k());
            n_blocks * ldpc.n()
        }
        InnerFec::Convolutional { rate } => punctured_coded_len(outer_il_bits, rate),
    };

    let coded_bits = match inner_il {
        InterleaverKind::None => inner_coded_bits,
        InterleaverKind::Block { rows, cols } => round_up(inner_coded_bits, rows * cols),
        InterleaverKind::Convolutional { branches, depth } => {
            conv_il_bits(inner_coded_bits, branches, depth)
        }
    };

    BlockPlan {
        info_bytes,
        framed_bytes,
        outer_coded_bits,
        outer_il_bits,
        inner_coded_bits,
        coded_bits,
    }
}

/// Number of OFDM symbols a logical block occupies for a given constellation
/// over the base plan.
pub fn symbols_for_coded_bits(
    base: &OfdmConfig,
    constellation: ConstellationOrder,
    bits: usize,
) -> usize {
    let bps = base.carrier_plan.data_carriers().len() * constellation.bits_per_symbol();
    bits.div_ceil(bps)
}

// ── Coding chain (encode side) ─────────────────────────────────────────────

/// Applies a block interleaver to `bits` in place-by-value: writes bit `i` of
/// each padded block row-major and reads column-major. Returns the interleaved
/// bits plus the block size used, so the deinterleaver can trim padding.
pub fn interleave_bits(il: InterleaverKind, bits: &[u8]) -> Vec<u8> {
    match il {
        InterleaverKind::None => bits.to_vec(),
        InterleaverKind::Block { rows, cols } => {
            let block = rows * cols;
            let bi = crate::fec::BlockInterleaver::new(rows, cols);
            let mut out = Vec::with_capacity(bits.len().div_ceil(block) * block);
            // Reused across chunks: the interleaver and both scratch buffers are
            // built once instead of per chunk.
            let mut padded = vec![0u8; block];
            let mut permuted = vec![0u8; block];
            for chunk in bits.chunks(block) {
                padded[..chunk.len()].copy_from_slice(chunk);
                padded[chunk.len()..].fill(0);
                bi.interleave(&padded, &mut permuted);
                out.extend_from_slice(&permuted);
            }
            out
        }
        InterleaverKind::Convolutional { branches, depth } => {
            // Byte-domain streaming Forney interleaver, driven in FRAME mode:
            // reset, feed the (byte-packed, `branches`-aligned) payload, then
            // flush the delay lines. The output grows by the round-trip delay
            // `branches·(branches−1)·depth`, which `block_plan`'s `conv_il_bits`
            // mirrors so the deinterleaver knows the length and trims it.
            let mut ci = crate::fec::ConvInterleaver::new(branches, depth);
            let bytes = pack_bits_padded(bits);
            let n = round_up(bytes.len(), branches);
            let mut padded = bytes;
            padded.resize(n, 0);
            let mut out_bytes = ci.feed(&padded);
            out_bytes.extend_from_slice(&ci.flush());
            bytes_to_bits(&out_bytes)
        }
    }
}

/// Fixed information-bit block size for the outer BCH code (per shortened
/// codeword). Chosen so one codeword fits comfortably within GF(2^8)'s length
/// bound (n = k + parity ≤ 255) for the t values used here.
pub const BCH_INFO_BITS: usize = 120;

/// Encodes `message_bytes` through the outer code (byte domain), returning the
/// coded bits (MSB-first). The message bit stream is fragmented into
/// [`BCH_INFO_BITS`]-bit blocks, each encoded into one shortened BCH codeword;
/// the final block is zero-padded. `None` outer code passes the bytes through
/// as bits.
pub fn outer_encode(outer: OuterFec, message_bytes: &[u8], cache: &CodecCache) -> Vec<u8> {
    match outer {
        OuterFec::None => bytes_to_bits(message_bytes),
        OuterFec::Bch { t } => {
            let msg_bits = bytes_to_bits(message_bytes);
            let code = cache.bch(t, BCH_INFO_BITS);
            let mut out = Vec::new();
            for chunk in msg_bits.chunks(BCH_INFO_BITS) {
                let mut block = chunk.to_vec();
                block.resize(BCH_INFO_BITS, 0);
                out.extend_from_slice(&code.encode(&block));
            }
            out
        }
        OuterFec::ReedSolomon { n, n_parity } => {
            // RS is a byte-domain code: fragment into k-byte blocks, encode each
            // into an n-byte codeword (final block zero-padded), then emit bits.
            let rs = cache.rs(n, n_parity);
            let k = rs.k();
            let mut out_bytes = Vec::new();
            for chunk in message_bytes.chunks(k) {
                let mut block = chunk.to_vec();
                block.resize(k, 0);
                out_bytes.extend_from_slice(&rs.encode(&block));
            }
            bytes_to_bits(&out_bytes)
        }
    }
}

/// Encodes `info_bits` through the inner code, returning coded bits. The info
/// bit stream is fragmented into K-bit blocks, each encoded into one N-bit
/// codeword (final block zero-padded). `None` passes through.
pub fn inner_encode(inner: InnerFec, info_bits: &[u8], cache: &CodecCache) -> Vec<u8> {
    match inner {
        InnerFec::None => info_bits.to_vec(),
        InnerFec::Ldpc(code) => {
            let ldpc = cache.ldpc(code);
            let k = ldpc.k();
            let mut out = Vec::new();
            for chunk in info_bits.chunks(k) {
                let mut msg = chunk.to_vec();
                msg.resize(k, 0);
                out.extend_from_slice(&ldpc.encode(&msg));
            }
            out
        }
        // The convolutional code terminates once per block (whole info stream +
        // tail bits), not per fixed-size fragment.
        InnerFec::Convolutional { rate } => conv_encode_punctured(info_bits, rate),
    }
}

/// Constructs a BCH code correcting `t` errors, shortened so its message part
/// holds exactly `msg_bits` information bits.
pub fn shortened_bch_for(t: usize, msg_bits: usize) -> Bch {
    // Parity length is fixed by t; choose n = msg_bits + parity_bits.
    let full = Bch::new(t).expect("valid BCH t");
    let parity = full.parity_bits();
    Bch::shortened(msg_bits + parity, t).expect("valid shortened BCH")
}

/// Runs the full encode chain for one logical block (header or payload):
/// `bytes → CRC → [scramble if before] → outer → outer-interleave → inner →
/// inner-interleave → [scramble if after]`, returning coded bits ready to map.
#[allow(clippy::too_many_arguments)]
pub fn encode_chain(
    bytes: &[u8],
    crc: CrcKind,
    outer: OuterFec,
    inner: InnerFec,
    outer_il: InterleaverKind,
    inner_il: InterleaverKind,
    scrambler: ScramblerKind,
    scrambler_pos: ScramblerPos,
    per_frame_seed: u32,
    cache: &CodecCache,
) -> Vec<u8> {
    // 1. CRC over the raw bytes.
    let mut framed = append_crc(crc, bytes);

    // 2. Optional scramble before the outer code.
    let sc = build_scrambler(scrambler, per_frame_seed);
    if scrambler_pos == ScramblerPos::BeforeOuterFec
        && let Some(ref s) = sc
    {
        s.scramble(&mut framed);
    }

    // 3. Outer FEC (byte → coded bits), then outer interleave (byte-domain,
    //    but we operate on bits here for a single generic interleaver).
    let outer_bits = outer_encode(outer, &framed, cache);
    let outer_il_bits = interleave_bits(outer_il, &outer_bits);

    // 4. Inner FEC (bits → coded bits), then inner interleave.
    let inner_bits = inner_encode(inner, &outer_il_bits, cache);
    let mut coded = interleave_bits(inner_il, &inner_bits);

    // 5. Optional scramble after the inner code (bit domain).
    if scrambler_pos == ScramblerPos::AfterInnerFec
        && let Some(ref s) = sc
    {
        // Scramble whole bytes; pad to a byte boundary, scramble, trim.
        scramble_bits(s, &mut coded);
    }

    coded
}

/// Scrambles a bit vector by packing to bytes (zero-padded), XORing the PN
/// sequence, and unpacking — used for the after-inner-FEC bit-domain position.
pub fn scramble_bits(s: &PnScrambler, bits: &mut [u8]) {
    let mut bytes = pack_bits_padded(bits);
    s.scramble(&mut bytes);
    let unpacked = bytes_to_bits(&bytes);
    bits.copy_from_slice(&unpacked[..bits.len()]);
}

/// Packs bits to bytes, zero-padding the final partial byte.
fn pack_bits_padded(bits: &[u8]) -> Vec<u8> {
    let mut padded = bits.to_vec();
    let rem = padded.len() % 8;
    if rem != 0 {
        padded.resize(padded.len() + (8 - rem), 0);
    }
    bits_to_bytes(&padded)
}

/// Serializes the 14 header field bytes (before CRC), big-endian.
pub fn pack_header_fields(
    mcs_index: u8,
    payload_len: u32,
    sequence_num: u32,
    flags: u8,
    scrambler_seed: u32,
) -> [u8; HEADER_FIELD_BYTES] {
    let mut out = [0u8; HEADER_FIELD_BYTES];
    out[0] = mcs_index;
    out[1..5].copy_from_slice(&payload_len.to_be_bytes());
    out[5..9].copy_from_slice(&sequence_num.to_be_bytes());
    out[9] = flags;
    out[10..14].copy_from_slice(&scrambler_seed.to_be_bytes());
    out
}

/// Maps coded bits to IQ symbols by running `OfdmMod::modulate` with the given
/// constellation over the shared carrier plan. Zero-pads the final partial
/// OFDM symbol (as `OfdmMod::modulate` does).
fn map_bits_to_iq(base: &OfdmConfig, constellation: ConstellationOrder, bits: &[u8]) -> Vec<C32> {
    let cfg = symbol_config(base, constellation);
    let mut modstage = OfdmMod::new(&cfg);
    modstage.modulate(bits)
}

/// Builds a bare per-symbol `OfdmConfig` (no frame fields) for a given
/// constellation, sharing the base plan/fs/rf/gain. Used to drive `OfdmMod`
/// for the header (BPSK) and payload (MCS) symbol streams.
pub fn symbol_config(base: &OfdmConfig, constellation: ConstellationOrder) -> OfdmConfig {
    OfdmConfig::new(
        base.carrier_plan.clone(),
        base.fs,
        base.rf_hz,
        base.gain,
        constellation,
    )
}

/// The OFDM frame modulator.
#[derive(Debug, Clone)]
pub struct OfdmFrameMod {
    cfg: OfdmConfig,
    mcs_table: McsTable,
    preamble: OfdmPreamble,
    /// FEC code cache, so a stream of frames builds each code once (see
    /// [`CodecCache`]). Held behind `Arc` so it can be shared with a paired
    /// demodulator (TX and RX then reuse the same built codes).
    cache: Arc<CodecCache>,
}

impl OfdmFrameMod {
    /// Creates a frame modulator over `cfg`, an `mcs_table`, and the
    /// acquisition `preamble` (which should carry a training symbol sized to
    /// the plan for the receiver's channel estimation). The modulator owns a
    /// fresh, private [`CodecCache`]; use [`with_cache`](Self::with_cache) to
    /// share one with a demodulator.
    pub fn new(cfg: OfdmConfig, mcs_table: McsTable, preamble: OfdmPreamble) -> Self {
        Self::with_cache(cfg, mcs_table, preamble, Arc::new(CodecCache::new()))
    }

    /// Like [`new`](Self::new), but reuses the caller-provided `cache` — share
    /// one `Arc<CodecCache>` across a modulator/demodulator pair (or several
    /// links on the same MCS) so each FEC code is constructed only once.
    pub fn with_cache(
        cfg: OfdmConfig,
        mcs_table: McsTable,
        preamble: OfdmPreamble,
        cache: Arc<CodecCache>,
    ) -> Self {
        Self {
            cfg,
            mcs_table,
            preamble,
            cache,
        }
    }

    pub fn config(&self) -> &OfdmConfig {
        &self.cfg
    }

    /// The training-symbol-carrying preamble prepended to every frame.
    pub fn preamble(&self) -> &OfdmPreamble {
        &self.preamble
    }

    /// Modulates a whole `FramePacket` into a flat IQ stream:
    /// `[preamble+training][header][payload]`.
    ///
    /// `per_frame_seed` supplies the scrambler seed for a `PerFrameRandom`
    /// configuration (ignored otherwise); it is recorded in the header so the
    /// receiver can rebuild the descrambler.
    pub fn modulate_frame(&self, frame: &FramePacket, per_frame_seed: u32) -> Vec<C32> {
        let mut out = Vec::new();

        // 1. Preamble + training symbol.
        out.extend_from_slice(&generate_ofdm_preamble(&self.preamble, &self.cfg));

        // 2. Header (present unless NoHeader).
        if self.cfg.header_format == HeaderFormat::OrionSdr {
            let fields = pack_header_fields(
                frame.metadata.mcs_index,
                frame.payload.len() as u32,
                frame.metadata.sequence_num,
                frame.metadata.flags,
                per_frame_seed,
            );
            let header_bits = encode_chain(
                &fields,
                self.cfg.header_crc,
                OuterFec::None,
                InnerFec::Ldpc(HEADER_LDPC),
                InterleaverKind::None,
                InterleaverKind::None,
                ScramblerKind::None,
                ScramblerPos::BeforeOuterFec,
                0,
                &self.cache,
            );
            out.extend_from_slice(&map_bits_to_iq(
                &self.cfg,
                HEADER_CONSTELLATION,
                &header_bits,
            ));
        }

        // 3. Payload, coded per the selected MCS.
        let mcs = self
            .mcs_table
            .get(frame.metadata.mcs_index)
            .expect("mcs_index must be in the MCS table");
        let payload_bits = encode_chain(
            &frame.payload,
            self.cfg.payload_crc,
            mcs.outer_fec,
            mcs.inner_fec,
            self.cfg.outer_interleaver,
            self.cfg.inner_interleaver,
            self.cfg.scrambler,
            self.cfg.scrambler_pos,
            per_frame_seed,
            &self.cache,
        );
        out.extend_from_slice(&map_bits_to_iq(&self.cfg, mcs.constellation, &payload_bits));

        out
    }
}

/// Convenience: the carrier plan cloned from a config (used by the demodulator).
pub fn plan_of(cfg: &OfdmConfig) -> CarrierPlan {
    cfg.carrier_plan.clone()
}
