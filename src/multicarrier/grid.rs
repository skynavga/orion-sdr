// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/multicarrier/grid.rs
use super::config::{CarrierPlan, SubcarrierRole};
use crate::core::{Block, WorkReport};
use num_complex::Complex32 as C32;

/// Resolved carrier-index → rustfft-bin mapping for one `CarrierPlan`,
/// built once via [`CarrierGrid::from_plan`].
///
/// Natural rustfft bin order is used internally (bin 0 = DC, negative
/// frequencies wrap into `n_fft/2..n_fft`); the signed-index → bin
/// translation (`bin = carrier_idx.rem_euclid(n_fft)`) happens once here at
/// construction, never per-sample.
#[derive(Debug, Clone)]
pub struct CarrierGrid {
    n_fft: usize,
    role: Vec<SubcarrierRole>,
    data_bins: Vec<usize>,
    pilot_bins: Vec<(usize, C32)>,
}

impl CarrierGrid {
    /// Builds the grid from `plan`, resolving each signed carrier index to its
    /// rustfft bin once.
    ///
    /// # Panics
    ///
    /// Panics if `plan` fails [`CarrierPlan::validate`] (out-of-range index,
    /// data/pilot overlap, or empty data set). An invalid plan is a caller
    /// programming error: without this check an overlapping data/pilot carrier
    /// would land in both `data_bins` and `pilot_bins`, and [`GridMap`] would
    /// silently overwrite the data symbol with the pilot value — a wrong-result
    /// bug with no other signal. Every OFDM construction path (`OfdmMod::new`,
    /// `OfdmDemod::new`, `OfdmEqualizer::new`) routes through here, so this one
    /// check guards the whole Rust pipeline; the Python bindings additionally
    /// surface the same `validate()` error as a `ValueError` before reaching
    /// this point.
    pub fn from_plan(plan: &CarrierPlan) -> Self {
        plan.validate()
            .expect("CarrierGrid::from_plan: invalid CarrierPlan");
        let n_fft = plan.n_fft();
        let mut role = vec![SubcarrierRole::Null; n_fft];

        let mut data_bins = Vec::with_capacity(plan.data_carriers().len());
        for &idx in plan.data_carriers() {
            let bin = idx.rem_euclid(n_fft as i32) as usize;
            role[bin] = SubcarrierRole::Data;
            data_bins.push(bin);
        }

        let mut pilot_bins = Vec::with_capacity(plan.pilot_carriers().len());
        for &(idx, value) in plan.pilot_carriers() {
            let bin = idx.rem_euclid(n_fft as i32) as usize;
            role[bin] = SubcarrierRole::Pilot;
            pilot_bins.push((bin, value));
        }

        Self {
            n_fft,
            role,
            data_bins,
            pilot_bins,
        }
    }

    pub fn n_fft(&self) -> usize {
        self.n_fft
    }

    /// Role lookup by rustfft bin index, length `n_fft`.
    pub fn role(&self) -> &[SubcarrierRole] {
        &self.role
    }

    /// Data-carrier bins, in the same order as `CarrierPlan::data_carriers`.
    pub fn data_bins(&self) -> &[usize] {
        &self.data_bins
    }

    /// Pilot-carrier bins with their known TX values, in the same order as
    /// `CarrierPlan::pilot_carriers`.
    pub fn pilot_bins(&self) -> &[(usize, C32)] {
        &self.pilot_bins
    }

    pub fn num_data_carriers(&self) -> usize {
        self.data_bins.len()
    }
}

/// TX resource-grid mapper: scatters dense data symbols into a sparse
/// `n_fft`-bin frequency vector (nulls zeroed, pilots inserted from known
/// values).
///
/// One `num_data_carriers()`-sized input chunk → one `n_fft`-sized output
/// chunk per `process()` call; a partial trailing chunk is a no-op.
#[derive(Debug, Clone)]
pub struct GridMap {
    grid: CarrierGrid,
}

impl GridMap {
    pub fn new(grid: CarrierGrid) -> Self {
        Self { grid }
    }

    pub fn num_data_carriers(&self) -> usize {
        self.grid.num_data_carriers()
    }

    pub fn n_fft(&self) -> usize {
        self.grid.n_fft()
    }
}

impl Block for GridMap {
    type In = C32;
    type Out = C32;

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> WorkReport {
        let n_data = self.grid.num_data_carriers();
        let n_fft = self.grid.n_fft();
        if input.len() < n_data || output.len() < n_fft {
            return WorkReport::default();
        }

        for bin in output[..n_fft].iter_mut() {
            *bin = C32::default();
        }
        for (k, &bin) in self.grid.data_bins().iter().enumerate() {
            output[bin] = input[k];
        }
        for &(bin, value) in self.grid.pilot_bins() {
            output[bin] = value;
        }

        WorkReport {
            in_read: n_data,
            out_written: n_fft,
        }
    }
}

/// RX resource-grid extractor: gathers data-carrier bins back into a dense
/// stream. Deliberately ignores pilots/channel estimation — that's a later
/// release's `OfdmEqualizer`, which runs upstream of `GridExtract` in the RX
/// chain.
///
/// One `n_fft`-sized input chunk → one `num_data_carriers()`-sized output
/// chunk per `process()` call; a partial trailing chunk is a no-op.
#[derive(Debug, Clone)]
pub struct GridExtract {
    grid: CarrierGrid,
}

impl GridExtract {
    pub fn new(grid: CarrierGrid) -> Self {
        Self { grid }
    }

    pub fn num_data_carriers(&self) -> usize {
        self.grid.num_data_carriers()
    }

    pub fn n_fft(&self) -> usize {
        self.grid.n_fft()
    }
}

impl Block for GridExtract {
    type In = C32;
    type Out = C32;

    fn process(&mut self, input: &[C32], output: &mut [C32]) -> WorkReport {
        let n_fft = self.grid.n_fft();
        let n_data = self.grid.num_data_carriers();
        if input.len() < n_fft || output.len() < n_data {
            return WorkReport::default();
        }

        for (k, &bin) in self.grid.data_bins().iter().enumerate() {
            output[k] = input[bin];
        }

        WorkReport {
            in_read: n_fft,
            out_written: n_data,
        }
    }
}
