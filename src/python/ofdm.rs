// Copyright (c) 2025-2026 G & R Associates LLC
// SPDX-License-Identifier: MIT OR Apache-2.0

// src/python/ofdm.rs — PyO3 bindings for OFDM: config, TX, RX, sync.
//
// `CarrierGrid`/`FftBlock`/`GridMap` are not exposed individually, matching
// how the per-order symbol mappers aren't exposed today: `PyOfdmMod` and
// `PyOfdmDemod` are the two main entry points, plus a free-function
// `ofdm_sync()` mirroring `ft8_sync`'s pattern.

use num_complex::Complex32;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::core::Block;
use crate::demodulate::{EqualizerMethod, OfdmDecider, OfdmEqualizer};
use crate::modulate::{ConstellationOrder, OfdmConfig, OfdmMod};
use crate::multicarrier::{CarrierGrid, CarrierPlan, CyclicPrefixRemove, FftBlock, GridExtract};
use crate::sync::{OfdmPreamble, generate_ofdm_preamble, ofdm_sync as ofdm_sync_fn};

type SoftDemodulateResult<'py> = (Bound<'py, PyArray1<Complex32>>, Bound<'py, PyArray1<u8>>);

fn parse_constellation(s: &str) -> PyResult<ConstellationOrder> {
    match s {
        "bpsk" => Ok(ConstellationOrder::Bpsk),
        "qpsk" => Ok(ConstellationOrder::Qpsk),
        "qam16" => Ok(ConstellationOrder::Qam16),
        "qam64" => Ok(ConstellationOrder::Qam64),
        "qam256" => Ok(ConstellationOrder::Qam256),
        other => Err(PyValueError::new_err(format!(
            "OfdmConfig: unknown constellation {:?} (expected one of: bpsk, qpsk, qam16, qam64, qam256)",
            other
        ))),
    }
}

// ── OfdmConfig ────────────────────────────────────────────────────────────────

/// OFDM waveform configuration: carrier plan (FFT size, cyclic-prefix
/// length, data/pilot carrier layout) plus RF/constellation parameters.
#[pyclass(name = "OfdmConfig", eq, skip_from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PyOfdmConfig(pub(crate) OfdmConfig);

#[pymethods]
impl PyOfdmConfig {
    /// `pilot_carrier_indices`/`pilot_carrier_values` are parallel arrays
    /// (same length): `pilot_carrier_indices[i]` carries the known symbol
    /// `pilot_carrier_values[i]`. Pass empty arrays for no pilots.
    #[new]
    #[pyo3(signature = (n_fft, cp_len, data_carriers, pilot_carrier_indices, pilot_carrier_values, fs, rf_hz, gain, constellation))]
    #[allow(clippy::too_many_arguments)] // mirrors the Python-facing signature
    fn new<'py>(
        n_fft: usize,
        cp_len: usize,
        data_carriers: PyReadonlyArray1<'py, i32>,
        pilot_carrier_indices: PyReadonlyArray1<'py, i32>,
        pilot_carrier_values: PyReadonlyArray1<'py, Complex32>,
        fs: f32,
        rf_hz: f32,
        gain: f32,
        constellation: &str,
    ) -> PyResult<Self> {
        let pilot_indices = pilot_carrier_indices.as_slice()?;
        let pilot_values = pilot_carrier_values.as_slice()?;
        if pilot_indices.len() != pilot_values.len() {
            return Err(PyValueError::new_err(format!(
                "OfdmConfig: pilot_carrier_indices ({}) and pilot_carrier_values ({}) must have the same length",
                pilot_indices.len(),
                pilot_values.len()
            )));
        }
        let order = parse_constellation(constellation)?;
        let pilots: Vec<(i32, Complex32)> = pilot_indices
            .iter()
            .copied()
            .zip(pilot_values.iter().copied())
            .collect();
        let plan = CarrierPlan::new(n_fft, cp_len)
            .with_data_carriers(data_carriers.as_slice()?.iter().copied())
            .with_pilot_carriers(pilots);
        plan.validate()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self(OfdmConfig::new(plan, fs, rf_hz, gain, order)))
    }

    #[getter]
    fn bits_per_ofdm_symbol(&self) -> usize {
        self.0.bits_per_ofdm_symbol()
    }

    #[getter]
    fn samples_per_ofdm_symbol(&self) -> usize {
        self.0.samples_per_ofdm_symbol()
    }
}

// ── OfdmMod ───────────────────────────────────────────────────────────────────

/// OFDM transmitter: fused mapper + resource-grid mapping + IFFT + cyclic
/// prefix + optional RF upconversion.
///
/// Input: uint8 array of bits (LSB of each byte); consumed
/// `bits_per_ofdm_symbol` at a time, zero-padding a final partial symbol.
/// Output: complex64 IQ array, `samples_per_ofdm_symbol` samples per symbol.
#[pyclass(name = "OfdmMod")]
pub struct PyOfdmMod(OfdmMod);

#[pymethods]
impl PyOfdmMod {
    #[new]
    fn new(cfg: &PyOfdmConfig) -> Self {
        Self(OfdmMod::new(&cfg.0))
    }

    fn modulate<'py>(
        &mut self,
        py: Python<'py>,
        bits: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
        let input = bits.as_slice()?;
        let iq = self.0.modulate(input);
        Ok(iq.into_pyarray(py))
    }
}

// ── OfdmDemod ─────────────────────────────────────────────────────────────────

/// OFDM receiver: fused cyclic-prefix removal + FFT + channel equalization
/// + resource-grid extraction + hard-decision decoding.
///
/// `equalizer` selects the channel-estimation method: `"training_symbol"`
/// (the default — one estimate per packet, held constant; call
/// `estimate_channel()` once with a demodulated training symbol before
/// `demodulate()`) or `"pilot_interp"` (re-estimated every symbol from
/// in-band pilots, no separate estimation call needed).
///
/// Input: complex64 IQ array, `samples_per_ofdm_symbol` samples per symbol.
/// Output: uint8 array of bits, `bits_per_ofdm_symbol` per symbol.
#[pyclass(name = "OfdmDemod")]
pub struct PyOfdmDemod {
    cfg: OfdmConfig,
    cp_remove: CyclicPrefixRemove,
    fft: FftBlock,
    equalizer: OfdmEqualizer,
    grid_extract: GridExtract,
    decider: OfdmDecider,
    n_fft: usize,
    samples_per_symbol: usize,
    num_data_carriers: usize,
    bits_per_symbol: usize,
}

#[pymethods]
impl PyOfdmDemod {
    #[new]
    #[pyo3(signature = (cfg, equalizer = "training_symbol"))]
    fn new(cfg: &PyOfdmConfig, equalizer: &str) -> PyResult<Self> {
        let method = match equalizer {
            "training_symbol" => EqualizerMethod::TrainingSymbolHold,
            "pilot_interp" => EqualizerMethod::PerSymbolPilotInterp,
            other => {
                return Err(PyValueError::new_err(format!(
                    "OfdmDemod: unknown equalizer {:?} (expected 'training_symbol' or 'pilot_interp')",
                    other
                )));
            }
        };
        let grid = CarrierGrid::from_plan(&cfg.0.carrier_plan);
        let n_fft = cfg.0.carrier_plan.n_fft();
        let cp_len = cfg.0.carrier_plan.cp_len();
        Ok(Self {
            cp_remove: CyclicPrefixRemove::new(n_fft, cp_len),
            fft: FftBlock::new(n_fft),
            equalizer: OfdmEqualizer::new(&cfg.0, method),
            grid_extract: GridExtract::new(grid.clone()),
            decider: OfdmDecider::new(&cfg.0),
            n_fft,
            samples_per_symbol: cfg.0.samples_per_ofdm_symbol(),
            num_data_carriers: grid.num_data_carriers(),
            bits_per_symbol: cfg.0.bits_per_ofdm_symbol(),
            cfg: cfg.0.clone(),
        })
    }

    /// Estimates and holds the channel from one already-demodulated training
    /// symbol's raw IQ (`samples_per_ofdm_symbol` samples, CP included). Only
    /// meaningful for the `"training_symbol"` equalizer; a no-op under
    /// `"pilot_interp"`.
    fn estimate_channel<'py>(
        &mut self,
        training_iq: PyReadonlyArray1<'py, Complex32>,
    ) -> PyResult<()> {
        let input = training_iq.as_slice()?;
        if input.len() < self.samples_per_symbol {
            return Err(PyValueError::new_err(format!(
                "OfdmDemod.estimate_channel: input too short ({} < {})",
                input.len(),
                self.samples_per_symbol
            )));
        }
        let mut time = vec![Complex32::default(); self.n_fft];
        self.cp_remove
            .process(&input[..self.samples_per_symbol], &mut time);
        let mut freq = vec![Complex32::default(); self.n_fft];
        self.fft.process(&time, &mut freq);
        self.equalizer.estimate_from_training_symbol(&freq);
        Ok(())
    }

    fn demodulate<'py>(
        &mut self,
        py: Python<'py>,
        iq: PyReadonlyArray1<'py, Complex32>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let (_soft, bits) = self.demodulate_inner(iq.as_slice()?)?;
        Ok(bits.into_pyarray(py))
    }

    /// Like `demodulate()`, but also returns the pre-decision soft symbols
    /// (post-equalization, post-grid-extract), for callers that want to
    /// build an [`PyOfdmRxFrame`] via `build_ofdm_rx_frame`.
    fn demodulate_soft<'py>(
        &mut self,
        py: Python<'py>,
        iq: PyReadonlyArray1<'py, Complex32>,
    ) -> PyResult<SoftDemodulateResult<'py>> {
        let (soft, bits) = self.demodulate_inner(iq.as_slice()?)?;
        Ok((soft.into_pyarray(py), bits.into_pyarray(py)))
    }
}

impl PyOfdmDemod {
    fn demodulate_inner(&mut self, input: &[Complex32]) -> PyResult<(Vec<Complex32>, Vec<u8>)> {
        if input.len() < self.samples_per_symbol {
            return Err(PyValueError::new_err(format!(
                "OfdmDemod.demodulate: input too short ({} < {})",
                input.len(),
                self.samples_per_symbol
            )));
        }

        let mut time = vec![Complex32::default(); self.n_fft];
        self.cp_remove
            .process(&input[..self.samples_per_symbol], &mut time);
        let mut freq = vec![Complex32::default(); self.n_fft];
        self.fft.process(&time, &mut freq);
        let mut equalized = vec![Complex32::default(); self.n_fft];
        self.equalizer.process(&freq, &mut equalized);
        let mut soft = vec![Complex32::default(); self.num_data_carriers];
        self.grid_extract.process(&equalized, &mut soft);
        let mut bits = vec![0u8; self.bits_per_symbol];
        self.decider.process(&soft, &mut bits);

        let _ = &self.cfg;
        Ok((soft, bits))
    }
}

// ── OfdmRxFrame ───────────────────────────────────────────────────────────────

/// Per-packet OFDM RX diagnostics. Fields that require acquisition or
/// equalization stay `None` until the caller has actually run those stages.
#[pyclass(name = "OfdmRxFrame")]
pub struct PyOfdmRxFrame {
    bits: Vec<u8>,
    num_symbols: usize,
    evm_db: Option<f32>,
    cfo_hz: Option<f32>,
    timing_offset_samples: Option<i32>,
    channel_mse: Option<f32>,
}

#[pymethods]
impl PyOfdmRxFrame {
    #[getter]
    fn bits<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        self.bits.clone().into_pyarray(py)
    }
    #[getter]
    fn num_symbols(&self) -> usize {
        self.num_symbols
    }
    #[getter]
    fn evm_db(&self) -> Option<f32> {
        self.evm_db
    }
    #[getter]
    fn cfo_hz(&self) -> Option<f32> {
        self.cfo_hz
    }
    #[getter]
    fn timing_offset_samples(&self) -> Option<i32> {
        self.timing_offset_samples
    }
    #[getter]
    fn channel_mse(&self) -> Option<f32> {
        self.channel_mse
    }
}

/// Builds an [`PyOfdmRxFrame`] from demodulated soft symbols and their
/// corresponding hard-decided bits, mirroring
/// `demodulate::ofdm::build_ofdm_rx_frame`.
#[pyfunction]
#[pyo3(name = "build_ofdm_rx_frame")]
fn py_build_ofdm_rx_frame<'py>(
    cfg: &PyOfdmConfig,
    soft_symbols: PyReadonlyArray1<'py, Complex32>,
    bits: PyReadonlyArray1<'py, u8>,
) -> PyResult<PyOfdmRxFrame> {
    let soft = soft_symbols.as_slice()?;
    let bits_vec = bits.as_slice()?.to_vec();
    let frame = crate::demodulate::ofdm::build_ofdm_rx_frame(&cfg.0, soft, bits_vec);
    Ok(PyOfdmRxFrame {
        bits: frame.bits,
        num_symbols: frame.num_symbols,
        evm_db: frame.evm_db,
        cfo_hz: frame.cfo_hz,
        timing_offset_samples: frame.timing_offset_samples,
        channel_mse: frame.channel_mse,
    })
}

// ── ofdm_sync ─────────────────────────────────────────────────────────────────

/// Searches an OFDM IQ buffer for a repeated-segment preamble match.
///
/// Returns a list of dicts, sorted by descending score:
///   {"start_sample": int, "cfo_hz": float, "integer_cfo_bins": int, "score": float}
///
/// `integer_cfo_bins` is only meaningful (nonzero) if `with_training_symbol`
/// was set when generating the preamble; total CFO is
/// `cfo_hz + integer_cfo_bins * (fs / n_fft)`.
#[pyfunction]
#[pyo3(signature = (iq, fs, num_repeats, repeat_len, search_start, search_end, training_n_fft = None, training_cp_len = None))]
#[allow(clippy::too_many_arguments)] // mirrors the Python-facing signature
fn ofdm_sync<'py>(
    py: Python<'py>,
    iq: PyReadonlyArray1<'py, Complex32>,
    fs: f32,
    num_repeats: usize,
    repeat_len: usize,
    search_start: usize,
    search_end: usize,
    training_n_fft: Option<usize>,
    training_cp_len: Option<usize>,
) -> PyResult<Bound<'py, PyList>> {
    let input = iq.as_slice()?;
    let mut preamble = OfdmPreamble::new(num_repeats, repeat_len);
    if let (Some(n_fft), Some(cp_len)) = (training_n_fft, training_cp_len) {
        preamble = preamble.with_training_symbol(n_fft, cp_len);
    }

    let results = ofdm_sync_fn(input, fs, &preamble, search_start, search_end);

    let list = PyList::empty(py);
    for r in results {
        let d = PyDict::new(py);
        d.set_item("start_sample", r.start_sample)?;
        d.set_item("cfo_hz", r.cfo_hz)?;
        d.set_item("integer_cfo_bins", r.integer_cfo_bins)?;
        d.set_item("score", r.score)?;
        list.append(d)?;
    }
    Ok(list)
}

/// Generates a repeated-segment preamble (plus training symbol, if
/// `training_n_fft`/`training_cp_len` are given) for prepending before OFDM
/// data symbols.
#[pyfunction]
#[pyo3(name = "generate_ofdm_preamble")]
#[pyo3(signature = (cfg, num_repeats, repeat_len, training_n_fft = None, training_cp_len = None))]
fn generate_ofdm_preamble_py<'py>(
    py: Python<'py>,
    cfg: &PyOfdmConfig,
    num_repeats: usize,
    repeat_len: usize,
    training_n_fft: Option<usize>,
    training_cp_len: Option<usize>,
) -> Bound<'py, PyArray1<Complex32>> {
    let mut preamble = OfdmPreamble::new(num_repeats, repeat_len);
    if let (Some(n_fft), Some(cp_len)) = (training_n_fft, training_cp_len) {
        preamble = preamble.with_training_symbol(n_fft, cp_len);
    }
    let iq = generate_ofdm_preamble(&preamble, &cfg.0);
    iq.into_pyarray(py)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOfdmConfig>()?;
    m.add_class::<PyOfdmMod>()?;
    m.add_class::<PyOfdmDemod>()?;
    m.add_class::<PyOfdmRxFrame>()?;
    m.add_function(wrap_pyfunction!(py_build_ofdm_rx_frame, m)?)?;
    m.add_function(wrap_pyfunction!(ofdm_sync, m)?)?;
    m.add_function(wrap_pyfunction!(generate_ofdm_preamble_py, m)?)?;
    Ok(())
}
