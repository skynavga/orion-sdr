// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/multicarrier/config.rs
use num_complex::Complex32 as C32;

/// Role assigned to a single OFDM subcarrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubcarrierRole {
    Data,
    Pilot,
    Null,
}

/// Errors returned by [`CarrierPlan::validate`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CarrierPlanError {
    #[error("carrier index {0} is out of range for n_fft={1} (valid: -(n_fft/2)..=((n_fft-1)/2))")]
    OutOfRange(i32, usize),
    #[error("carrier index {0} is assigned more than one role (data/pilot overlap)")]
    Overlap(i32),
    #[error("no data carriers specified")]
    EmptyDataSet,
}

/// Resource-grid description: FFT size, cyclic-prefix length, and the
/// data/pilot subcarrier assignment. Bakes in no standard's numerology —
/// the caller chooses `n_fft`, `cp_len`, and carrier layout to match their
/// link's delay spread and Doppler spread.
///
/// Carrier indices are **signed**, following the convention that bin 0 is
/// DC and negative frequencies count down from it (e.g. `-26..=26`). Bin 0
/// (DC) is implicitly null unless explicitly included in `data_carriers` or
/// `pilot_carriers` — callers must opt in to using DC.
#[derive(Debug, Clone, PartialEq)]
pub struct CarrierPlan {
    n_fft: usize,
    cp_len: usize,
    data_carriers: Vec<i32>,
    pilot_carriers: Vec<(i32, C32)>,
}

impl CarrierPlan {
    pub fn new(n_fft: usize, cp_len: usize) -> Self {
        Self {
            n_fft,
            cp_len,
            data_carriers: Vec::new(),
            pilot_carriers: Vec::new(),
        }
    }

    pub fn with_data_carriers(mut self, carriers: impl IntoIterator<Item = i32>) -> Self {
        self.data_carriers.extend(carriers);
        self
    }

    pub fn with_pilot_carriers(mut self, carriers: impl IntoIterator<Item = (i32, C32)>) -> Self {
        self.pilot_carriers.extend(carriers);
        self
    }

    pub fn n_fft(&self) -> usize {
        self.n_fft
    }

    pub fn cp_len(&self) -> usize {
        self.cp_len
    }

    pub fn data_carriers(&self) -> &[i32] {
        &self.data_carriers
    }

    pub fn pilot_carriers(&self) -> &[(i32, C32)] {
        &self.pilot_carriers
    }

    /// Signed carrier-index bounds representable by `n_fft`: negative
    /// frequencies down to `-(n_fft/2)`, positive up to `(n_fft-1)/2`.
    fn index_bounds(&self) -> (i32, i32) {
        let n = self.n_fft as i32;
        (-(n / 2), (n - 1) / 2)
    }

    fn in_range(&self, idx: i32) -> bool {
        let (lo, hi) = self.index_bounds();
        idx >= lo && idx <= hi
    }

    /// Validate carrier-index range and data/pilot overlap, and confirm at
    /// least one data carrier is specified.
    pub fn validate(&self) -> Result<(), CarrierPlanError> {
        if self.data_carriers.is_empty() {
            return Err(CarrierPlanError::EmptyDataSet);
        }

        for &idx in &self.data_carriers {
            if !self.in_range(idx) {
                return Err(CarrierPlanError::OutOfRange(idx, self.n_fft));
            }
        }
        for &(idx, _) in &self.pilot_carriers {
            if !self.in_range(idx) {
                return Err(CarrierPlanError::OutOfRange(idx, self.n_fft));
            }
        }

        let mut seen = std::collections::HashSet::new();
        for &idx in &self.data_carriers {
            if !seen.insert(idx) {
                return Err(CarrierPlanError::Overlap(idx));
            }
        }
        for &(idx, _) in &self.pilot_carriers {
            if !seen.insert(idx) {
                return Err(CarrierPlanError::Overlap(idx));
            }
        }

        Ok(())
    }
}
