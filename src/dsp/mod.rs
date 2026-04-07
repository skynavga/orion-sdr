// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/dsp/mod.rs
#![allow(dead_code)]

pub mod agc;
pub use agc::{AgcRms, AgcRmsIq};

pub mod dc;
pub use dc::DcBlocker;

pub mod decim;
pub use decim::FirDecimator;

pub mod fir;
pub use fir::{FirLowpass, HalfCosineMf};

pub mod iir;
pub use iir::{Biquad, LpCascade, LpDcCascade};

pub mod nco;
pub use nco::{Nco, mix_with_nco};

pub mod rotator;
pub use rotator::Rotator;
