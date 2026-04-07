// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

//! orion-sdr — toplevel exports

pub mod core;
pub mod dsp;
pub mod demodulate;
pub mod modulate;
pub mod codec;
pub mod sync;
pub mod message;
pub mod util;
#[cfg(feature = "extension-module")]
pub mod python;

pub use core::{Block, WorkReport, AudioToIqChain, IqToAudioChain, IqToIqChain};
