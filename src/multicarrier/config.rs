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
    #[error("carrier index {0} intrudes into the {1}-carrier edge-guard band")]
    InGuardBand(i32, usize),
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
    /// TX symbol-window roll-off in samples (raised-cosine taper per symbol
    /// edge). `0` (the default) means no windowing. Symbol geometry, like
    /// `cp_len`, so every waveform/profile that funnels through `CarrierPlan`
    /// inherits it. See [`with_window_roll_off`](Self::with_window_roll_off).
    window_roll_off: usize,
}

impl CarrierPlan {
    pub fn new(n_fft: usize, cp_len: usize) -> Self {
        Self {
            n_fft,
            cp_len,
            data_carriers: Vec::new(),
            pilot_carriers: Vec::new(),
            window_roll_off: 0,
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

    /// Sets the TX symbol-window roll-off in samples: a raised-cosine taper of
    /// `roll_off` samples at each symbol edge, applied by the modulator to soften
    /// the boundary discontinuity and reduce out-of-band emission. `0` (the
    /// default) disables windowing. The taper is only RX-transparent when the
    /// receiver's window back-off is set so the ramp falls outside the FFT
    /// window (`roll_off ≤ min(cp_len - b, b)`, maximized at `b = cp_len/2`); the
    /// caller is responsible for pairing the two (see `rx_window_backoff`).
    pub fn with_window_roll_off(mut self, roll_off: usize) -> Self {
        self.window_roll_off = roll_off;
        self
    }

    /// TX symbol-window roll-off in samples (`0` = no windowing).
    pub fn window_roll_off(&self) -> usize {
        self.window_roll_off
    }

    /// Fill `data_carriers` with a contiguous span that leaves `edge_guard`
    /// null carriers at each band edge (in addition to the always-null Nyquist
    /// bin at `-(n_fft/2)`). DC (index 0) is skipped unless `include_dc`, and
    /// any index already present in `pilot_carriers` is skipped, so data and
    /// pilots never overlap.
    ///
    /// This narrows the occupied bandwidth and pulls the strongest
    /// `sinc`-skirt generators inward, reducing out-of-band emission for the
    /// caller-owned (COFDM) carrier layout. `edge_guard == 0` with no pilots
    /// reproduces the full-fill span, so it is regression-safe as a default.
    ///
    /// # On `include_dc`
    ///
    /// `false` is the conventional choice and the right default: a
    /// direct-conversion front end puts its LO leakage and DC offset squarely
    /// on bin 0, so most numerologies (802.11, LTE) leave it null and spend the
    /// carrier rather than fight the impairment. `true` is supported — every
    /// stage that needs to know which bins are occupied derives that from
    /// [`occupied_bins`](Self::occupied_bins), including the preamble's
    /// training symbol — and is worth taking when the link does not sit at
    /// baseband DC, e.g. one modulated at `rf_hz = 0.0` and then upconverted
    /// wholesale by a [`Rotator`](crate::dsp::Rotator), where bin 0 lands on
    /// the RF carrier rather than on the receiver's LO.
    ///
    /// Occupying DC does **not** buy DC-offset immunity: nothing here estimates
    /// or removes a receiver-side DC offset, so an impaired front end corrupts
    /// that carrier and the FEC absorbs it or does not.
    ///
    /// Owns the **data**-carrier list only: use it *instead of*
    /// [`with_data_carriers`](Self::with_data_carriers) (both extend the same
    /// vec), but *alongside*
    /// [`with_pilot_carriers`](Self::with_pilot_carriers). Call
    /// `with_pilot_carriers` first so the pilot indices are excluded from the
    /// data fill:
    ///
    /// ```ignore
    /// CarrierPlan::new(n_fft, cp_len)
    ///     .with_pilot_carriers(pilots)
    ///     .with_contiguous_data(edge_guard, /* include_dc */ false)
    /// ```
    pub fn with_contiguous_data(mut self, edge_guard: usize, include_dc: bool) -> Self {
        let (lo, hi) = self.index_bounds();
        let g = edge_guard as i32;
        // The Nyquist bin -(n_fft/2) is representable (so index_bounds includes
        // it) but conventionally null — the canonical full-fill span never
        // occupies it. Start the fill one above it so `edge_guard == 0`
        // reproduces that span exactly; the guard `g` then measures from the
        // lowest *usable* index.
        let start = lo + 1 + g;
        let pilots: std::collections::HashSet<i32> =
            self.pilot_carriers.iter().map(|&(idx, _)| idx).collect();
        for idx in start..=(hi - g) {
            if idx == 0 && !include_dc {
                continue;
            }
            if pilots.contains(&idx) {
                continue;
            }
            self.data_carriers.push(idx);
        }
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

    /// Half-width of the occupied band in subcarriers: the largest `|index|`
    /// over the data and pilot carriers (`0` if the plan is empty). This is the
    /// band edge a TX spectral mask must not cut into — see
    /// [`TxLowpass::for_null_band`](super::TxLowpass::for_null_band).
    pub fn occupied_half_carriers(&self) -> usize {
        let data = self.data_carriers.iter().copied();
        let pilots = self.pilot_carriers.iter().map(|&(idx, _)| idx);
        data.chain(pilots)
            .map(|idx| idx.unsigned_abs() as usize)
            .max()
            .unwrap_or(0)
    }

    /// Every rustfft bin this plan occupies — data and pilot alike — resolved
    /// once via `bin = idx.rem_euclid(n_fft)`, deduplicated and sorted
    /// ascending. Empty when the plan assigns nothing.
    ///
    /// The single answer to "which bins does this link use", so the stages that
    /// need it cannot disagree. [`occupied_half_carriers`](Self::occupied_half_carriers)
    /// answers a narrower question — where the band *ends*, which is what a
    /// spectral mask needs — and cannot express which bins inside that span are
    /// live. Using it as if it could is what let the preamble's training symbol
    /// null DC while [`with_contiguous_data`](Self::with_contiguous_data)
    /// handed DC out as a data carrier: the training symbol never transmitted
    /// the bin whose channel the equalizer then divided by.
    pub fn occupied_bins(&self) -> Vec<usize> {
        if self.n_fft == 0 {
            return Vec::new();
        }
        let n = self.n_fft as i32;
        let mut bins: Vec<usize> = self
            .data_carriers
            .iter()
            .copied()
            .chain(self.pilot_carriers.iter().map(|&(idx, _)| idx))
            .map(|idx| idx.rem_euclid(n) as usize)
            .collect();
        bins.sort_unstable();
        bins.dedup();
        bins
    }

    pub fn pilot_carriers(&self) -> &[(i32, C32)] {
        &self.pilot_carriers
    }

    /// Signed carrier-index bounds representable by `n_fft`: negative
    /// frequencies down to `-(n_fft/2)`, positive up to `(n_fft-1)/2`.
    pub fn index_bounds(&self) -> (i32, i32) {
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

    /// In addition to the [`validate`](Self::validate) checks, confirm that no
    /// data or pilot carrier intrudes into the `edge_guard`-wide null band at
    /// either edge — i.e. every assigned index lies in
    /// `[-(n_fft/2) + edge_guard ..= (n_fft-1)/2 - edge_guard]`.
    ///
    /// Opt-in and non-breaking: [`validate`](Self::validate) is unchanged, so
    /// existing callers are unaffected. Use this to assert a guard band is
    /// actually honored when pilots are placed by hand.
    pub fn validate_edge_guard(&self, edge_guard: usize) -> Result<(), CarrierPlanError> {
        self.validate()?;
        let (lo, hi) = self.index_bounds();
        let g = edge_guard as i32;
        let (glo, ghi) = (lo + g, hi - g);
        for &idx in &self.data_carriers {
            if idx < glo || idx > ghi {
                return Err(CarrierPlanError::InGuardBand(idx, edge_guard));
            }
        }
        for &(idx, _) in &self.pilot_carriers {
            if idx < glo || idx > ghi {
                return Err(CarrierPlanError::InGuardBand(idx, edge_guard));
            }
        }
        Ok(())
    }
}
