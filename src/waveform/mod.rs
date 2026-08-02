// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/waveform/mod.rs

//! Standard-specific waveform assemblies built from the generic `fec/`,
//! `multicarrier/`, `modulate/`, and `demodulate/` primitives. Unlike those
//! modules — which bake in no standard's numerology or coding — a `waveform`
//! submodule encodes one concrete standard's parameters (carrier maps, FEC
//! chains, scramblers) and produces ready-to-use configs.
//!
//! Currently: `dvbt` (DVB-T / narrowband DVB-T for amateur DATV).

pub mod dvbt;
