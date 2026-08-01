// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/fec/mod.rs

//! Waveform-agnostic forward-error-correction and framing building blocks:
//! Galois-field arithmetic, block codes (BCH now, LDPC/Reed–Solomon as they
//! land), interleavers, a PN scrambler, and the frame/packet types shared by
//! the OFDM frame layer. Nothing here depends on OFDM — these are generic
//! channel-coding primitives reusable across waveforms.

pub mod bch;
pub use bch::{Bch, BchError};

pub mod conv;
pub use conv::{PunctureRate, conv_encode_punctured, punctured_coded_len, viterbi_decode_soft};

pub mod frame;
pub use frame::{
    CrcKind, FrameMetadata, FramePacket, HeaderFormat, InnerFec, InterleaverKind, OuterFec,
    RxError, ScramblerKind, ScramblerPos, SeedMode,
};

pub mod gf;
pub use gf::Gf256;

pub mod interleaver;
pub use interleaver::{BlockInterleaver, ConvDeinterleaver, ConvInterleaver, conv_roundtrip_delay};

pub mod ldpc_codes;
pub use ldpc_codes::{DecodeRule, Ldpc, LdpcCode};

pub mod reed_solomon;
pub use reed_solomon::{ReedSolomon, RsError};

pub mod scrambler;
pub use scrambler::{PnScrambler, PnScramblerStream};
