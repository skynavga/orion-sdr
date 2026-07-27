// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/fec/mod.rs

//! Waveform-agnostic forward-error-correction and framing building blocks:
//! Galois-field arithmetic, block codes (BCH now, LDPC/Reed–Solomon as they
//! land), interleavers, a PN scrambler, and the frame/packet types shared by
//! the OFDM frame layer. Nothing here depends on OFDM — these are generic
//! channel-coding primitives reusable across waveforms.

pub mod gf;
pub use gf::Gf256;

pub mod interleaver;
pub use interleaver::BlockInterleaver;

pub mod scrambler;
pub use scrambler::PnScrambler;
