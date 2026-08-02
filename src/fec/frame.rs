// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/fec/frame.rs
//
// Waveform-agnostic frame/packet types and the FEC/interleaver/CRC/scrambler
// scheme descriptors shared by the OFDM frame layer. Nothing here depends on
// OFDM: `FramePacket`/`FrameMetadata`/`RxError` are a byte payload plus
// metadata plus a typed error, and the scheme enums are lightweight parameter
// descriptors (not the runtime codec objects) so a config that embeds them
// stays cheaply `Clone`/`PartialEq`.
//
// The concatenated coding chain these descriptors parameterize is
//   payload → CRC → [scramble] → outer FEC → outer interleave →
//              inner FEC → inner interleave → [scramble] → symbol map
// (reversed on receive), with the scrambler position selected by
// [`ScramblerPos`]. See the OFDM frame modulator/demodulator for the wiring.

use super::{ConvCode, LdpcCode, PunctureRate};

/// A high-level transport unit (MAC-layer view): metadata plus an opaque byte
/// payload. The frame modulator serializes this to IQ; the frame demodulator
/// recovers it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FramePacket {
    pub metadata: FrameMetadata,
    pub payload: Vec<u8>,
}

impl FramePacket {
    pub fn new(metadata: FrameMetadata, payload: Vec<u8>) -> Self {
        Self { metadata, payload }
    }
}

/// Per-frame metadata carried in (or alongside) the frame header.
///
/// `sequence_num` lets a receiver order/deduplicate frames; `mcs_index`
/// selects the payload's modulation-and-coding scheme from the sender's MCS
/// table; `flags` is reserved for future use (e.g. a last-fragment marker).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameMetadata {
    pub sequence_num: u32,
    pub mcs_index: u8,
    pub flags: u8,
}

impl FrameMetadata {
    pub fn new(sequence_num: u32, mcs_index: u8) -> Self {
        Self {
            sequence_num,
            mcs_index,
            flags: 0,
        }
    }
}

/// Why a received frame could not be delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RxError {
    /// No preamble was found within the searched buffer (streaming RX only).
    #[error("no preamble found")]
    PreambleTimeout,
    /// The header symbols could not be decoded into a well-formed header
    /// (e.g. an implausible payload length).
    #[error("malformed frame header")]
    MalformedHeader,
    /// The header decoded but its CRC did not check.
    #[error("header CRC mismatch")]
    HeaderCrcMismatch,
    /// The payload decoded but its CRC did not check.
    #[error("payload CRC mismatch")]
    CrcMismatch,
    /// The inner/outer FEC could not converge to a valid codeword.
    #[error("FEC uncorrectable")]
    FecUncorrectable,
}

// ── Scheme descriptors ─────────────────────────────────────────────────────

/// Outer (algebraic, hard-decision) FEC selection. The outer code runs on the
/// byte/symbol domain after the inner decoder, correcting residual errors the
/// inner code left behind — the DVB-style concatenation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OuterFec {
    #[default]
    None,
    /// Binary BCH shortened to fit the frame, correcting up to `t` errors per
    /// codeword.
    Bch { t: usize },
    /// Reed–Solomon over GF(2^8): `n` codeword bytes, `n_parity = 2t` parity
    /// bytes (so `k = n − n_parity`), correcting up to `t` symbol errors per
    /// codeword. The DVB-T outer code is `ReedSolomon { n: 204, n_parity: 16 }`.
    ReedSolomon { n: usize, n_parity: usize },
}

/// Inner (soft-decision) FEC selection. The inner code consumes the
/// demapper's LLRs directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InnerFec {
    #[default]
    None,
    /// One of the fixed-family LDPC codes.
    Ldpc(LdpcCode),
    /// Zero-tail-terminated, punctured, soft-Viterbi-decoded convolutional code
    /// at the given puncture rate. `code` selects the mother code
    /// ([`ConvCode::K5`], the crate default, or [`ConvCode::DvbK7`], DVB-T's
    /// constraint-length-7 inner code).
    Convolutional { rate: PunctureRate, code: ConvCode },
}

/// Interleaver selection for either the inner (LLR-domain) or outer
/// (byte-domain) stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterleaverKind {
    #[default]
    None,
    /// Rectangular block interleaver over `rows × cols` elements.
    Block { rows: usize, cols: usize },
    /// DVB-T Forney convolutional interleaver, `branches` (`I`) branches each a
    /// `depth` (`M`)-cell delay line, driven as a streaming block (see
    /// [`crate::fec::ConvInterleaver`]). DVB-T's outer interleaver is
    /// `Convolutional { branches: 12, depth: 17 }`. Frame orchestrators drive it
    /// in reset-per-frame mode; a stream orchestrator carries its state across
    /// the whole stream.
    Convolutional { branches: usize, depth: usize },
}

/// Cyclic-redundancy-check selection (presence and width), used independently
/// for the header and the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CrcKind {
    /// No integrity check (accept/reject rests on FEC alone).
    None,
    /// CRC-16/CCITT-FALSE.
    Crc16,
    /// CRC-32/ISO-HDLC.
    #[default]
    Crc32,
}

impl CrcKind {
    /// Width of the appended CRC field in bytes.
    pub fn len_bytes(self) -> usize {
        match self {
            CrcKind::None => 0,
            CrcKind::Crc16 => 2,
            CrcKind::Crc32 => 4,
        }
    }
}

/// How an additive scrambler is seeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedMode {
    /// Fixed initial register value, known to both ends (DVB energy dispersal).
    Fixed(u32),
    /// A per-frame-random seed, signaled to the receiver in the header
    /// (802.11-style). Requires a header (`HeaderFormat != NoHeader`).
    PerFrameRandom,
}

/// Additive PN scrambler (energy-dispersal whitener) selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScramblerKind {
    #[default]
    None,
    /// A parameterized additive LFSR: `poly` feedback taps over a `width`-bit
    /// register, seeded per `seed`.
    Additive {
        poly: u32,
        width: u8,
        seed: SeedMode,
    },
    /// DVB-T energy dispersal: the exact standard PRBS (1 + X^14 + X^15, init
    /// `100101010000000`, MSB-first — `waveform::dvb_t::DvbTEnergyDispersal`),
    /// applied byte-domain. Distinct from `Additive` because DVB-T's whitener is
    /// MSB-first with fixed parameters, not a generic LSB-first LFSR.
    DvbTEnergyDispersal,
    // A self-synchronizing multiplicative scrambler is a later addition.
}

/// Where the scrambler sits in the coding chain (standard-dependent).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScramblerPos {
    /// Whiten the payload bytes before the outer FEC (DVB energy-dispersal
    /// position), so a decode failure stays CRC-detectable.
    #[default]
    BeforeOuterFec,
    /// Whiten the final coded bits just before symbol mapping.
    AfterInnerFec,
}

/// On-air header format selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HeaderFormat {
    /// The bespoke `orion-sdr` header (see the OFDM frame modulator): a fixed,
    /// BPSK + rate-1/2-coded symbol block carrying `mcs_index`,
    /// `payload_len`, `sequence_num`, `flags`, a scrambler seed, and a header
    /// CRC. Not conformant to any particular standard.
    #[default]
    OrionSdr,
    /// No in-band header symbol; the receiver takes MCS/length from
    /// configuration or out-of-band signaling.
    NoHeader,
}
