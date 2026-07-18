// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/modulate/ofdm.rs
use crate::multicarrier::CarrierPlan;

/// OFDM waveform configuration. Fields are added as later releases need
/// them; `carrier_plan` is the only field Release B's resource-grid mapping
/// requires.
#[derive(Debug, Clone, PartialEq)]
pub struct OfdmConfig {
    pub carrier_plan: CarrierPlan,
}

impl OfdmConfig {
    pub fn new(carrier_plan: CarrierPlan) -> Self {
        Self { carrier_plan }
    }
}
