// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

//! orion-sdr — toplevel exports

pub mod codec;
pub mod core;
pub mod demodulate;
pub mod dsp;
pub mod message;
pub mod modulate;
#[cfg(feature = "extension-module")]
pub mod python;
pub mod sync;
pub mod util;

pub use core::{AudioToIqChain, Block, IqToAudioChain, IqToIqChain, WorkReport};
