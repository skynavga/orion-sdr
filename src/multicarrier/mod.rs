// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/multicarrier/mod.rs

//! Waveform-agnostic FFT-domain primitives shared by OFDM and, in future
//! releases, other multicarrier waveforms (DFT-s-OFDM/SC-FDMA, OTFS).

pub mod config;
pub use config::{CarrierPlan, CarrierPlanError, SubcarrierRole};

pub mod cyclic_prefix;
pub use cyclic_prefix::{CyclicPrefixInsert, CyclicPrefixRemove};

pub mod fft;
pub use fft::{FftBlock, IfftBlock};

pub mod grid;
pub use grid::{CarrierGrid, GridExtract, GridMap};
