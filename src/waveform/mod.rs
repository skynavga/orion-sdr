// Copyright (c) 2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/waveform/mod.rs

//! Standard-specific waveform assemblies built from the generic `fec/`,
//! `multicarrier/`, `modulate/`, and `demodulate/` primitives. Unlike those
//! modules — which bake in no standard's numerology or coding — a `waveform`
//! submodule encodes one concrete standard's parameters (carrier maps, FEC
//! chains, scramblers) and produces ready-to-use configs.
//!
//! Currently: `dvb_t` (DVB-T / narrowband DVB-T for amateur DATV), its
//! `dvb_t_tps` TPS-signalling submodule, and `dvb_t_ts` (MPEG-2 transport-stream
//! packet adaptation + energy dispersal).

pub mod dvb_t;
pub mod dvb_t_tps;
pub mod dvb_t_ts;
